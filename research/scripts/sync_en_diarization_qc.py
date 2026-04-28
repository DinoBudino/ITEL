from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG
from research.scripts.bootstrap_en_metadata import METADATA_COLUMNS
from research.scripts.build_en_manifest import _merge_diarization_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill English metadata with diarization QC derived from the summary CSV.")
    parser.add_argument(
        "--metadata",
        default=str(CONFIG.paths.manifests_dir / "english_external_metadata.csv"),
        help="English metadata CSV to update in place.",
    )
    parser.add_argument(
        "--diarization-summary",
        default=str(CONFIG.paths.diarization_dir / "english_external_diarization_summary.csv"),
        help="Diarization summary CSV to merge into metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    summary_path = Path(args.diarization_summary)

    metadata = pd.read_csv(metadata_path)
    merged = _merge_diarization_summary(metadata, summary_path)
    merged = merged.reindex(columns=METADATA_COLUMNS)
    merged.to_csv(metadata_path, index=False)
    print(f"Updated English metadata with diarization QC: {metadata_path}")


if __name__ == "__main__":
    main()
