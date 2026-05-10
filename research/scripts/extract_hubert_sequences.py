from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from tqdm import tqdm

from research.src.config import CONFIG
from research.src.preprocessing import load_and_preprocess_audio
from research.src.sequence_model import HubertSequenceEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HuBERT last_hidden_state sequences [T,768] and save to .pt files."
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Path to the Czech sample manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CONFIG.paths.processed_dir / "hubert_sequences" / "train_default"),
        help="Directory for .pt files.",
    )
    parser.add_argument(
        "--output-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_train_default.csv"),
        help="Output CSV mapping sample_id to sequence_path, label, subject_id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Auto-detected if not set (cuda / cpu).",
    )
    parser.add_argument(
        "--natural-length",
        action="store_true",
        default=False,
        help="Use the full natural audio duration (no 2s crop/pad).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N samples (debug).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.natural_length:
        duration_label = "natural"
        duration_s = None
        if args.output_manifest == str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_train_default.csv"):
            args.output_manifest = str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural.csv")
        if args.output_dir == str(CONFIG.paths.processed_dir / "hubert_sequences" / "train_default"):
            args.output_dir = str(CONFIG.paths.processed_dir / "hubert_sequences" / "natural")
    else:
        duration_label = "2.0s_fixed"
        duration_s = CONFIG.audio.train_segment_s

    sample_manifest = pd.read_csv(args.sample_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Duration mode: {duration_label}")

    encoder = HubertSequenceEncoder(device=device)

    records: list[dict] = []
    rows = sample_manifest.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    for row in tqdm(rows, desc=f"extracting hubert seq ({duration_label})"):
        sample_id = row["sample_id"]
        audio_path = row["raw_audio_path"]
        pt_path = output_dir / f"{sample_id}.pt"

        if pt_path.exists():
            records.append({
                "sample_id": sample_id,
                "sequence_path": str(pt_path),
                "label": row["label"],
                "subject_id": row["subject_id"],
                "sequence_length": -1,
                "input_dim": 768,
            })
            continue

        try:
            samples, sr = load_and_preprocess_audio(
                audio_path,
                duration_s=duration_s,
                profile=CONFIG.audio.default_profile,
            )
            waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
            seq, mask = encoder.extract_sequence(waveform)

            torch.save(
                {"sequence": seq.float(), "mask": mask.bool()},
                pt_path,
            )

            records.append({
                "sample_id": sample_id,
                "sequence_path": str(pt_path),
                "label": row["label"],
                "subject_id": row["subject_id"],
                "sequence_length": int(mask.sum()),
                "input_dim": int(seq.size(-1)),
            })
        except Exception as exc:
            print(f"ERROR processing {sample_id}: {exc}")

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(args.output_manifest, index=False)
    print(f"Saved {len(manifest_df)} sequence records to {args.output_manifest}")

    if len(manifest_df) > 0:
        lengths = manifest_df.loc[manifest_df["sequence_length"] >= 0, "sequence_length"]
        if len(lengths) > 0:
            print(f"Sequence lengths: min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")


if __name__ == "__main__":
    main()
