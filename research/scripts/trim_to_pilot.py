"""Trim the English metadata CSV to n_per_label completed files per label.

Usage:
    python research/scripts/trim_to_pilot.py          # default: 10 per label
    python research/scripts/trim_to_pilot.py --n 20   # 20 per label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV = ROOT_DIR / "research" / "manifests" / "english_external_metadata.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Files to keep per label (default: 10)")
    args = parser.parse_args()

    df = pd.read_csv(CSV)
    print(f"Before: {len(df)} rows  {df.groupby('label').size().to_dict()}")

    kept = []
    for label in ["Healthy", "Patients"]:
        completed = df[(df["label"] == label) & (df["diarization_status"] == "completed")]
        selected = completed.head(args.n)
        if len(selected) < args.n:
            print(f"  [warn] Only {len(selected)} completed rows for {label} (wanted {args.n})")
        kept.append(selected)
        print(f"  {label}: keeping {len(selected)} files")

    trimmed = pd.concat(kept, ignore_index=True)
    print(f"\nAfter:  {len(trimmed)} rows  {trimmed.groupby('label').size().to_dict()}")

    df.to_csv(CSV.with_suffix(".csv.bak"), index=False)
    print(f"Backup saved to: {CSV.with_suffix('.csv.bak')}")

    trimmed.to_csv(CSV, index=False)
    print(f"Trimmed metadata saved to: {CSV}")


if __name__ == "__main__":
    main()
