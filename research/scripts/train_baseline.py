from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import mlflow
import pandas as pd

from research.src.config import CONFIG
from research.src.tracking import configure_tracking, describe_model, enable_training_autolog, log_artifact, log_dict, log_params
from research.src.train import MODEL_TRAINERS, RECOMMENDED_MODEL_TYPE, load_embeddings


TASK_VARIANTS: dict[str, tuple[str, ...]] = {
    "all_tasks": ("controlled_vowel", "controlled_consonant", "controlled_word"),
    "word_only": ("controlled_word",),
    "vowel_only": ("controlled_vowel",),
    "consonant_only": ("controlled_consonant",),
    "speech_sounds_only": ("controlled_vowel", "controlled_consonant"),
}

MODEL_TYPES = tuple(MODEL_TRAINERS.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact MLP head on frozen Czech XLS-R embeddings.")
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
        default=str(CONFIG.paths.model_dir / "xlsr300m_mlp_small_cz.joblib"),
        help="Output path for the trained Czech model.",
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
        default=RECOMMENDED_MODEL_TYPE,
        help="Classifier head trained on top of the frozen Czech embeddings.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="itel-baselines",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to a local file store under research_results/mlruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    enable_training_autolog()

    with mlflow.start_run(run_name=args.mlflow_run_name):
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
        model = MODEL_TRAINERS[args.model_type](X, y)

        output_model = Path(args.output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_model)

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "train_baseline.py",
                "dataset_id": CONFIG.datasets.cz_dataset_id,
                "task_variant": args.task_variant,
                "model_type": args.model_type,
            }
        )
        log_params(
            {
                "sample_manifest": args.sample_manifest,
                "embedding_manifest": args.embedding_manifest,
                "output_model": str(output_model),
                "task_variant": args.task_variant,
                "task_types": task_types,
                "model_type": args.model_type,
                "training_rows": len(merged),
                "n_samples": merged["source_sample_id"].nunique(),
                "n_labels": y.nunique(),
                "embedding_dim": X.shape[1],
                "tracking_uri": tracking_uri,
            }
        )
        log_dict(
            "metadata/train_baseline_summary.json",
            {
                "task_types": list(task_types),
                "model": describe_model(model),
                "label_distribution": {str(key): int(value) for key, value in y.value_counts().to_dict().items()},
                "feature_shape": {"rows": int(X.shape[0]), "cols": int(X.shape[1])},
            },
        )
        log_artifact(output_model, artifact_path="exported_model")

        print(f"Saved baseline model to: {output_model}")
        print(f"Model type: {args.model_type}")
        print(f"Task variant: {args.task_variant} -> {task_types}")
        print(f"Training rows: {len(merged)}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
