"""Build a robust interviewer reference embedding from multiple diarized files.

Instead of relying on one recording, this script:
  1. Reads existing segment JSONs for completed English files
  2. Identifies the interviewer (= the speaker that is NOT chi_speaker)
  3. Keeps only segments >= min_duration_s (filters boundary errors)
  4. Optionally trims segment boundaries by trim_s to reduce bleed
  5. Computes a wespeaker embedding per file using only those clean segments
  6. L2-normalises each embedding and averages them → robust reference

Usage:
    python research/scripts/build_robust_reference.py ^
        --output assets/models/interviewer_reference_embedding.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from research.src.config import CONFIG
from research.src.diarization import (
    extract_speaker_embeddings_from_segments,
    load_audio_mono_16k,
    load_embedding_model,
)
from research.src.preprocessing import preprocess_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a robust multi-file interviewer reference embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.model_dir / "interviewer_reference_embedding.npy"),
        help="Output path for the reference embedding (.npy).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum segment duration in seconds (default: 2.0). "
             "Shorter segments are more likely to be diarization errors.",
    )
    parser.add_argument(
        "--trim",
        type=float,
        default=0.15,
        help="Seconds to trim from each segment boundary (default: 0.15). "
             "Reduces speech bleed from the other speaker.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files used (default: all completed).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def load_interviewer_segments(
    json_path: Path,
    min_duration: float,
    trim: float,
) -> list[dict]:
    """Return segments belonging to the interviewer with short/boundary ones removed."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chi_speaker = data.get("chi_speaker")
    if not chi_speaker:
        return []

    clean = []
    for seg in data.get("segments", []):
        if str(seg["speaker"]) == chi_speaker:
            continue  # skip child segments
        start = float(seg["start"]) + trim
        end = float(seg["end"]) - trim
        if end - start < min_duration:
            continue  # too short after trimming
        clean.append({"speaker": seg["speaker"], "start": start, "end": end})
    return clean


def main() -> None:
    args = parse_args()

    meta = pd.read_csv(
        CONFIG.paths.manifests_dir / "english_external_metadata.csv"
    ).dropna(subset=["file_name"])
    completed = meta[meta["diarization_status"] == "completed"]

    if args.max_files:
        completed = completed.head(args.max_files)

    diar_dir = CONFIG.paths.diarization_dir

    print(f"\nBuilding robust reference from {len(completed)} files")
    print(f"  min_duration={args.min_duration}s  trim={args.trim}s\n")

    print("Loading embedding model…")
    embedding_model = load_embedding_model()
    print("Model ready.\n")

    embeddings: list[np.ndarray] = []
    skipped = 0

    for _, row in completed.iterrows():
        fname = str(row["file_name"])
        parts = fname.replace("\\", "/").split("/")
        label, stem = parts[0], Path(parts[1]).stem

        json_path = diar_dir / label / f"{stem}_segments.json"
        raw_audio = CONFIG.paths.raw_en_dir / label / parts[1]

        if not json_path.exists():
            print(f"  [skip] {parts[1]}: JSON not found")
            skipped += 1
            continue
        if not raw_audio.exists():
            print(f"  [skip] {parts[1]}: audio not found at {raw_audio}")
            skipped += 1
            continue

        segments = load_interviewer_segments(json_path, args.min_duration, args.trim)
        if not segments:
            print(f"  [skip] {parts[1]}: no clean interviewer segments")
            skipped += 1
            continue

        try:
            waveform, sr = load_audio_mono_16k(raw_audio)

            # Preprocess each interviewer segment before embedding:
            # highpass + trim silence + keep voiced + RMS normalise
            import torch as _torch
            audio_np = waveform.squeeze().cpu().numpy()
            clean_clips = []
            for seg in segments:
                start = max(int(seg["start"] * sr), 0)
                end = min(int(seg["end"] * sr), len(audio_np))
                if end <= start:
                    continue
                clip = audio_np[start:end].astype("float32")
                clip = preprocess_samples(clip, sr=sr, profile="anti_confound")
                if len(clip) > sr // 4:  # skip if < 0.25s after VAD
                    clean_clips.append(clip)

            if not clean_clips:
                print(f"  [skip] {parts[1]}: segmentos vacíos tras preprocesamiento")
                skipped += 1
                continue

            import numpy as _np
            concatenated = _np.concatenate(clean_clips).astype("float32")
            # Build a fake single-segment waveform for embedding extraction
            clean_waveform = _torch.tensor(concatenated).unsqueeze(0)
            fake_segments = [{"speaker": "EXA", "start": 0.0, "end": len(concatenated) / sr}]
            emb_dict = extract_speaker_embeddings_from_segments(
                clean_waveform, sr, fake_segments, embedding_model
            )
            if not emb_dict:
                print(f"  [skip] {parts[1]}: embedding extraction returned nothing")
                skipped += 1
                continue

            # Average across interviewer speaker labels (usually just one)
            vecs = list(emb_dict.values())
            vec = np.mean(vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            embeddings.append(vec)
            total_s = sum(s["end"] - s["start"] for s in segments)
            print(f"  [ok] {parts[1]:12s} ({label:8s})  {len(segments)} segmentos  {total_s:.0f}s speech")

        except Exception as exc:
            print(f"  [error] {parts[1]}: {exc}")
            skipped += 1

    if not embeddings:
        print("\n[ERROR] No embeddings could be extracted.")
        sys.exit(1)

    # Average all per-file embeddings and re-normalise
    reference = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(reference)
    reference = reference / norm if norm > 1e-8 else reference

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), reference)

    print(f"\n{'=' * 55}")
    print(f"✓ Referencia robusta guardada en: {out_path}")
    print(f"  Archivos usados : {len(embeddings)}")
    print(f"  Archivos omitidos: {skipped}")
    print(f"  Shape del embedding: {reference.shape}")
    print(f"{'=' * 55}")
    print("\nAhora vuelve a correr la diarización con la nueva referencia:")
    print("  python research/scripts/run_diarization.py --all --skip-completed --chi-strategy reference --reference-embedding assets/models/interviewer_reference_embedding.npy")


if __name__ == "__main__":
    main()
