from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from research.src.config import CONFIG
from research.src.train import fit_logreg_classifier, fit_mlp_classifier, load_embeddings


TASK_VARIANTS: dict[str, tuple[str, ...]] = {
    "all_tasks": ("controlled_vowel", "controlled_consonant", "controlled_word"),
    "word_only": ("controlled_word",),
    "vowel_only": ("controlled_vowel",),
    "consonant_only": ("controlled_consonant",),
    "speech_sounds_only": ("controlled_vowel", "controlled_consonant"),
}

MODEL_TYPES = ("logreg", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline LogisticRegression model on Czech XLSR embeddings.")
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Path to the Czech sample manifest.",
    )
    parser.add_argument(
        "--embedding-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"),
        help="Path to the Czech embedding manifest.",
    )
    parser.add_argument(
        "--output-model",
        default=str(CONFIG.paths.model_dir / "xlsr_logreg_baseline.joblib"),
        help="Output path for the trained baseline model.",
    )
    parser.add_argument(
        "--task-variant",
        choices=list(TASK_VARIANTS.keys()),
        default="all_tasks",
        help="Subset of Czech task types used for training.",
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default="logreg",
        help="Classifier family trained on top of the Czech embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_manifest = pd.read_csv(args.sample_manifest)
    embedding_manifest = pd.read_csv(args.embedding_manifest)

    task_types = TASK_VARIANTS[args.task_variant]
    sample_manifest = sample_manifest.loc[sample_manifest["task_type"].isin(task_types)].copy()
    allowed_ids = sample_manifest["sample_id"].tolist()
    embedding_manifest = embedding_manifest.loc[embedding_manifest["source_sample_id"].isin(allowed_ids)].copy()

    merged = embedding_manifest.merge(sample_manifest[["sample_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")
    if merged["label"].isna().any():
        missing = merged.loc[merged["label"].isna(), "source_sample_id"].tolist()[:10]
        raise ValueError(f"Missing labels for embedding rows. Example source_sample_id values: {missing}")

    X, _ = load_embeddings(merged)
    y = merged["label"]
    if args.model_type == "logreg":
        model = fit_logreg_classifier(X, y)
    else:
        model = fit_mlp_classifier(X, y)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)
    print(f"Saved baseline model to: {output_model}")
    print(f"Model type: {args.model_type}")
    print(f"Task variant: {args.task_variant} -> {task_types}")
    print(f"Training rows: {len(merged)}")


if __name__ == "__main__":
    main()
