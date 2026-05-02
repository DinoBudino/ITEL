from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research.src.config import CONFIG
from research.src.evaluate import binary_metrics
from research.src.train import MODEL_TRAINERS, fit_logreg_classifier, load_embeddings


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal Czech cross-validation with subject-disjoint folds.")
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
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by StratifiedGroupKFold.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(REPORT_CSV_DIR / "metrics_cz_internal_cv.json"),
        help="Where to save the CV metrics JSON.",
    )
    parser.add_argument(
        "--sample-predictions-output",
        default=str(REPORT_CSV_DIR / "predictions_cz_internal_cv_samples.csv"),
        help="Where to save out-of-fold sample predictions.",
    )
    parser.add_argument(
        "--subject-predictions-output",
        default=str(REPORT_CSV_DIR / "predictions_cz_internal_cv_subjects.csv"),
        help="Where to save out-of-fold subject predictions.",
    )
    return parser.parse_args()


def _positive_class_index(model) -> int:
    classes = list(model.named_steps["clf"].classes_)
    if "Patients" not in classes:
        raise ValueError(f"Patients class not found in classifier classes: {classes}")
    return classes.index("Patients")


def _summarize_fold_metrics(metrics_by_fold: list[dict[str, float | int | None]]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for metric_name in ["balanced_accuracy", "auc", "f1"]:
        values = [float(row[metric_name]) for row in metrics_by_fold if row.get(metric_name) is not None]
        if not values:
            summary[metric_name] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _subject_level_predictions(sample_predictions: pd.DataFrame) -> pd.DataFrame:
    subject_frame = (
        sample_predictions.groupby(["fold", "subject_id", "label"], as_index=False, dropna=False)["score_patients"]
        .mean()
        .rename(columns={"score_patients": "score_patients_mean"})
    )
    subject_frame["pred_label"] = subject_frame["score_patients_mean"].apply(lambda score: "Patients" if score >= 0.5 else "Healthy")
    return subject_frame


def run_cv(
    sample_manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
    n_splits: int,
    random_state: int,
    fit_fn=fit_logreg_classifier,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    merged = embedding_manifest.merge(
        sample_manifest[["sample_id", "subject_id", "label", "task_type"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_sample"),
    )
    if merged[["subject_id", "label"]].isna().any().any():
        raise ValueError("Merged Czech embeddings contain rows without subject_id/label.")

    X, merged = load_embeddings(merged)
    y = merged["label"].reset_index(drop=True)
    groups = merged["subject_id"].reset_index(drop=True)
    merged = merged.reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    sample_predictions: list[pd.DataFrame] = []
    sample_fold_metrics: list[dict[str, float | int | None]] = []
    subject_fold_metrics: list[dict[str, float | int | None]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = fit_fn(X[train_idx], y.iloc[train_idx])
        positive_index = _positive_class_index(model)

        fold_scores = model.predict_proba(X[test_idx])[:, positive_index]
        fold_pred_labels = np.where(fold_scores >= 0.5, "Patients", "Healthy")

        fold_frame = merged.iloc[test_idx][["sample_id", "source_sample_id", "subject_id", "label", "task_type"]].copy()
        fold_frame = fold_frame.rename(columns={"sample_id": "embedding_sample_id"})
        fold_frame["fold"] = fold_idx
        fold_frame["score_patients"] = fold_scores
        fold_frame["pred_label"] = fold_pred_labels
        sample_predictions.append(fold_frame)

        fold_metrics = binary_metrics(fold_frame["label"], fold_frame["pred_label"], fold_frame["score_patients"])
        fold_metrics["fold"] = fold_idx
        fold_metrics["n_subjects"] = int(fold_frame["subject_id"].nunique())
        sample_fold_metrics.append(fold_metrics)

        subject_frame = _subject_level_predictions(fold_frame)
        subject_metrics = binary_metrics(subject_frame["label"], subject_frame["pred_label"], subject_frame["score_patients_mean"])
        subject_metrics["fold"] = fold_idx
        subject_metrics["n_subjects"] = int(subject_frame["subject_id"].nunique())
        subject_fold_metrics.append(subject_metrics)

    sample_predictions_frame = pd.concat(sample_predictions, ignore_index=True)
    subject_predictions_frame = _subject_level_predictions(sample_predictions_frame)

    sample_metrics = binary_metrics(
        sample_predictions_frame["label"],
        sample_predictions_frame["pred_label"],
        sample_predictions_frame["score_patients"],
    )
    subject_metrics = binary_metrics(
        subject_predictions_frame["label"],
        subject_predictions_frame["pred_label"],
        subject_predictions_frame["score_patients_mean"],
    )

    metrics = {
        "config": {
            "n_splits": n_splits,
            "random_state": random_state,
            "dataset_id": CONFIG.datasets.cz_dataset_id,
            "eval_mode": "train_default",
            "grouping": "subject_id",
            "cv_strategy": "StratifiedGroupKFold",
        },
        "dataset": {
            "n_embedding_rows": int(len(merged)),
            "n_sample_ids": int(merged["source_sample_id"].nunique()),
            "n_subjects": int(merged["subject_id"].nunique()),
            "subjects_by_label": sample_manifest[["subject_id", "label"]].drop_duplicates()["label"].value_counts().to_dict(),
            "samples_by_label": sample_manifest["label"].value_counts().to_dict(),
        },
        "sample_level": {
            **sample_metrics,
            "folds": sample_fold_metrics,
            "fold_summary": _summarize_fold_metrics(sample_fold_metrics),
        },
        "subject_level": {
            **subject_metrics,
            "folds": subject_fold_metrics,
            "fold_summary": _summarize_fold_metrics(subject_fold_metrics),
        },
    }
    return sample_predictions_frame, subject_predictions_frame, metrics


def main() -> None:
    args = parse_args()
    sample_manifest = pd.read_csv(args.sample_manifest)
    embedding_manifest = pd.read_csv(args.embedding_manifest)

    sample_predictions, subject_predictions, metrics = run_cv(
        sample_manifest=sample_manifest,
        embedding_manifest=embedding_manifest,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    sample_output = Path(args.sample_predictions_output)
    subject_output = Path(args.subject_predictions_output)
    metrics_output = Path(args.metrics_output)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    subject_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    sample_predictions.to_csv(sample_output, index=False)
    subject_predictions.to_csv(subject_output, index=False)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved sample-level predictions to: {sample_output}")
    print(f"Saved subject-level predictions to: {subject_output}")
    print(f"Saved internal Czech CV metrics to: {metrics_output}")
    print(
        "Subject-level balanced_accuracy="
        f"{metrics['subject_level']['balanced_accuracy']:.4f} "
        f"AUC={metrics['subject_level']['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
