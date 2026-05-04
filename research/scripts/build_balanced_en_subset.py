from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced English external subset manifest.")
    parser.add_argument(
        "--input-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name),
        help="Path to the cleaned English sample manifest.",
    )
    parser.add_argument(
        "--output-manifest",
        default=str(CONFIG.paths.manifests_dir / "english_external_samples_balanced40.csv"),
        help="Where to save the balanced subset manifest.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=20,
        help="Number of samples to draw from each label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)

    frame = pd.read_csv(manifest_path)
    subsets: list[pd.DataFrame] = []
    for label in ["Healthy", "Patients"]:
        label_frame = frame.loc[frame["label"] == label].copy()
        if len(label_frame) < args.per_label:
            raise ValueError(f"Not enough rows for label '{label}': requested {args.per_label}, found {len(label_frame)}")
        subsets.append(label_frame.sample(n=args.per_label, random_state=args.seed).sort_values("subject_id"))

    balanced = pd.concat(subsets, ignore_index=True)
    balanced = balanced.sort_values(["label", "subject_id", "sample_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_path, index=False)

    counts = balanced["label"].value_counts().to_dict()
    print(f"Saved balanced English subset manifest to: {output_path}")
    print(f"Rows: {len(balanced)} | label counts: {counts}")


if __name__ == "__main__":
    main()
