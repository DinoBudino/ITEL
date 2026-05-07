"""Extract and save an interviewer reference embedding from a single audio file.

Usage (step 1 — inspect speakers):
    python research/scripts/extract_interviewer_reference.py \
        --audio "research/Datos_Extra/INGLES/SLI/A - SLI/479.mp3"

The script diarizes the audio, prints each detected speaker with their
duration and cosine-similarity to every other speaker, then exits so you
can pick the interviewer label.

Usage (step 2 — save the reference):
    python research/scripts/extract_interviewer_reference.py \
        --audio "research/Datos_Extra/INGLES/SLI/A - SLI/479.mp3" \
        --interviewer-speaker SPEAKER_00 \
        --output assets/models/interviewer_reference_embedding.npy

After running step 2, pass the .npy to run_diarization.py:
    python research/scripts/run_diarization.py \
        --all --chi-strategy reference \
        --reference-embedding assets/models/interviewer_reference_embedding.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torchaudio

from research.src.config import CONFIG
from research.src.diarization import (
    build_diarization_pipeline,
    compute_speaker_embedding,
    cosine_similarity,
    diarize_segments,
    extract_speaker_embeddings_from_segments,
    load_embedding_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract interviewer reference embedding from a single audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the reference audio file (e.g. 479.mp3).",
    )
    parser.add_argument(
        "--interviewer-speaker",
        default=None,
        help=(
            "Speaker label to save as the interviewer reference (e.g. SPEAKER_00). "
            "If omitted the script only prints speaker stats and exits."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.model_dir / "interviewer_reference_embedding.npy"),
        help="Where to save the reference embedding (.npy).",
    )
    parser.add_argument(
        "--export-speakers",
        action="store_true",
        help=(
            "Export each speaker's audio as a separate WAV file so you can listen "
            "and confirm who is the interviewer before saving the reference. "
            "Files are saved next to the reference audio as 479_SPEAKER_00.wav, etc."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = ROOT_DIR / audio_path
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\nLoading diarization pipeline…")
    pipeline, device = build_diarization_pipeline(device=args.device)

    print(f"Diarizing: {audio_path.name}")
    segments, waveform, sr = diarize_segments(audio_path, pipeline)

    # Aggregate duration per speaker
    totals: dict[str, float] = {}
    for seg in segments:
        speaker = str(seg["speaker"])
        totals[speaker] = totals.get(speaker, 0.0) + float(seg["end"]) - float(seg["start"])

    total_speech = sum(totals.values())
    print(f"\n{'─' * 50}")
    print(f"{'SPEAKER':<15} {'DURATION':>10}  {'SHARE':>8}")
    print(f"{'─' * 50}")
    for speaker, duration in sorted(totals.items(), key=lambda x: -x[1]):
        share = duration / total_speech if total_speech else 0.0
        print(f"{speaker:<15} {duration:>9.1f}s  {share:>7.1%}")
    print(f"{'─' * 50}")

    print(f"\nLoading ECAPA-TDNN embedding model…")
    embedding_model = load_embedding_model()

    print("Computing per-speaker embeddings…")
    speaker_embeddings = extract_speaker_embeddings_from_segments(waveform, sr, segments, embedding_model)

    # Print pairwise cosine similarities
    speakers = sorted(speaker_embeddings.keys())
    if len(speakers) > 1:
        print(f"\nPairwise cosine similarity between speakers:")
        for i, s1 in enumerate(speakers):
            for s2 in speakers[i + 1:]:
                sim = cosine_similarity(speaker_embeddings[s1], speaker_embeddings[s2])
                print(f"  {s1} ↔ {s2}: {sim:.3f}")

    # Export per-speaker WAV files so you can listen and verify
    if args.export_speakers:
        audio_np = waveform.squeeze().cpu().numpy()
        export_dir = audio_path.parent
        speaker_clips: dict[str, list] = {}
        for seg in segments:
            spk = str(seg["speaker"])
            start = max(int(float(seg["start"]) * sr), 0)
            end = min(int(float(seg["end"]) * sr), len(audio_np))
            if end > start:
                speaker_clips.setdefault(spk, []).append(audio_np[start:end])

        print(f"\nExporting speaker audio files to: {export_dir}")
        for spk, clips in sorted(speaker_clips.items()):
            out_path = export_dir / f"{audio_path.stem}_{spk}.wav"
            concatenated = torch.tensor(
                __import__("numpy").concatenate(clips), dtype=torch.float32
            ).unsqueeze(0)
            torchaudio.save(str(out_path), concatenated, sr)
            print(f"  ✓ {out_path.name}  ({totals.get(spk, 0):.1f}s)")
        print("\nListen to the files, then run again with --interviewer-speaker <LABEL> to save the embedding.")
        return

    if args.interviewer_speaker is None:
        print(
            "\nOptions:"
            "\n  --export-speakers          export WAV per speaker so you can listen first"
            "\n  --interviewer-speaker <X>  save the reference embedding directly\n"
        )
        return

    interviewer_label = args.interviewer_speaker.strip()
    if interviewer_label not in speaker_embeddings:
        available = ", ".join(sorted(speaker_embeddings.keys()))
        raise ValueError(
            f"Speaker '{interviewer_label}' not found in this audio. "
            f"Available speakers: {available}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), speaker_embeddings[interviewer_label])

    print(f"\n✓ Interviewer reference embedding saved to: {output_path}")
    print(
        f"  Speaker : {interviewer_label}"
        f"  Duration: {totals.get(interviewer_label, 0):.1f}s"
        f"  Shape   : {speaker_embeddings[interviewer_label].shape}"
    )
    print(
        "\nNow run diarization with the reference strategy:\n"
        f"  python research/scripts/run_diarization.py \\\n"
        f"      --all --chi-strategy reference \\\n"
        f"      --reference-embedding {output_path}\n"
    )


if __name__ == "__main__":
    main()
