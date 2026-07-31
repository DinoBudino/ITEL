from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.scripts.evaluate_multilingual_bidirectional import _load_manifest_embeddings
from research.src.config import CONFIG
from research.src.model_promotion import fit_and_export_target_head, write_multilingual_manifest
from research.src.multilingual_evaluation import aggregate_subject_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and export frozen Czech/English heads after model selection.")
    parser.add_argument("--cz-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural_multilingual_light.csv"))
    parser.add_argument("--en-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_v2.csv"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "models" / "registry" / "multilingual-cz-en" / "1.0.0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cz_subjects = aggregate_subject_embeddings(_load_manifest_embeddings(args.cz_sequence_manifest, "last"))
    en_subjects = aggregate_subject_embeddings(_load_manifest_embeddings(args.en_sequence_manifest, "last"))
    metrics = {
        "target_cs": fit_and_export_target_head(en_subjects, cz_subjects, output_dir / "target-cs.npz"),
        "target_en": fit_and_export_target_head(cz_subjects, en_subjects, output_dir / "target-en.npz"),
    }
    manifest = write_multilingual_manifest(output_dir, metrics)
    print(f"Promoted multilingual bundle: {manifest}")


if __name__ == "__main__":
    main()
