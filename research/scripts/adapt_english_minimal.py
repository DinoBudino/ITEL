from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_metrics, log_params
from research.src.train import MODEL_TRAINERS, RECOMMENDED_MODEL_TYPE, load_embeddings


REPORT_DIR = CONFIG.paths.reports_dir / "CSV" / "english_adaptation"


def aggregate_subject_scores(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    agg_name = "mean" if method == "mean" else "min"
    grouped = (
        frame.groupby(["subject_id", "label"], as_index=False)["score_patients"]
        .agg(agg_name)
        .rename(columns={"score_patients": "score_patients_subject"})
    )
    return grouped


def best_threshold(train_subjects: pd.DataFrame) -> tuple[float, float]:
    y_true = (train_subjects["label"] == "Patients").astype(int).to_numpy()
    scores = train_subjects["score_patients_subject"].to_numpy()
    unique_thresholds = sorted(set(float(value) for value in scores.tolist()))
    best: tuple[float, float] | None = None
    for threshold in unique_thresholds:
        preds = (scores >= threshold).astype(int)
        score = float(balanced_accuracy_score(y_true, preds))
        if best is None or score > best[1]:
            best = (float(threshold), score)
    if best is None:
        return 0.5, 0.0
    return best


def classify_subjects(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["pred_label"] = output["score_patients_subject"].apply(lambda score: "Patients" if float(score) >= threshold else "Healthy")
    return output


def metrics_from_subjects(frame: pd.DataFrame) -> dict[str, float | int | None]:
    y_true = (frame["label"] == "Patients").astype(int).to_numpy()
    y_pred = (frame["pred_label"] == "Patients").astype(int).to_numpy()
    scores = frame["score_patients_subject"].to_numpy()
    metrics: dict[str, float | int | None] = {
        "n_rows": int(len(frame)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "f1_patients": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return metrics


def model_positive_index(model) -> int:
    classes = list(model.classes_)
    if "Patients" not in classes:
        raise ValueError(f"Expected Patients class, got {classes}")
    return classes.index("Patients")


def _english_scores_from_model(model, X_en: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_en)[:, model_positive_index(model)]


def _fit_czech_plus_english(
    fit_fn: Callable,
    X_cz: np.ndarray,
    y_cz: pd.Series,
    X_en_train: np.ndarray,
    y_en_train: pd.Series,
):
    X_train = np.vstack([X_cz, X_en_train])
    y_train = pd.concat([y_cz.reset_index(drop=True), y_en_train.reset_index(drop=True)], ignore_index=True)
    return fit_fn(X_train, y_train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate minimal English cross-lingual adaptation with frozen 5s embeddings.")
    parser.add_argument("--cz-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name))
    parser.add_argument(
        "--cz-embedding-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"),
    )
    parser.add_argument("--en-sample-manifest", default=str(CONFIG.paths.manifests_dir / "english_test_samples.csv"))
    parser.add_argument(
        "--en-embedding-manifest",
        default=str(CONFIG.paths.processed_dir / "english_test_embeddings_seg_5s.csv"),
    )
    parser.add_argument(
        "--czech-model",
        default=str(CONFIG.paths.model_dir / "xlsr300m_mlp_small_cz.joblib"),
        help="Trained Czech-only model used for threshold-only adaptation.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=[RECOMMENDED_MODEL_TYPE, "logreg"],
        choices=tuple(MODEL_TRAINERS.keys()),
        help="Heads to compare for minimal adaptation. Logistic regression stays as a baseline only.",
    )
    parser.add_argument("--aggregation-methods", nargs="+", default=["mean", "min"], choices=["mean", "min"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-json", default=str(REPORT_DIR / "adaptation_summary.json"))
    parser.add_argument("--subject-predictions-csv", default=str(REPORT_DIR / "adaptation_subject_predictions.csv"))
    parser.add_argument("--mlflow-experiment", default="itel-adaptation")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)

    cz_samples = pd.read_csv(args.cz_sample_manifest)
    cz_embeddings = pd.read_csv(args.cz_embedding_manifest)
    en_samples = pd.read_csv(args.en_sample_manifest)
    en_embeddings = pd.read_csv(args.en_embedding_manifest)

    cz_merged = cz_embeddings.merge(cz_samples[["sample_id", "subject_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")
    en_merged = en_embeddings.merge(en_samples[["sample_id", "subject_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")

    X_cz, cz_merged = load_embeddings(cz_merged)
    X_en, en_merged = load_embeddings(en_merged)
    y_en = en_merged["label"].reset_index(drop=True)
    groups = en_merged["subject_id"].reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    czech_model = joblib.load(args.czech_model)
    baseline_scores = _english_scores_from_model(czech_model, X_en)

    all_predictions: list[pd.DataFrame] = []
    summary: dict[str, object] = {
        "config": {
            "n_splits": args.n_splits,
            "random_state": args.random_state,
            "aggregation_methods": args.aggregation_methods,
            "model_types": args.model_types,
            "czech_model": args.czech_model,
        },
        "dataset": {
            "n_english_rows": int(len(en_merged)),
            "n_english_subjects": int(en_merged["subject_id"].nunique()),
            "english_subjects_by_label": en_samples[["subject_id", "label"]].drop_duplicates()["label"].value_counts().to_dict(),
        },
        "threshold_only": {},
        "mixed_head_refit": {},
    }

    with mlflow.start_run(run_name=args.mlflow_run_name):
        for aggregation in args.aggregation_methods:
            threshold_only_folds: list[dict[str, float | int | None]] = []
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_en, y_en, groups=groups), start=1):
                train_frame = en_merged.iloc[train_idx][["subject_id", "label"]].copy()
                train_frame["score_patients"] = baseline_scores[train_idx]
                train_subjects = aggregate_subject_scores(train_frame, aggregation)
                threshold, train_ba = best_threshold(train_subjects)

                test_frame = en_merged.iloc[test_idx][["subject_id", "label"]].copy()
                test_frame["score_patients"] = baseline_scores[test_idx]
                test_subjects = classify_subjects(aggregate_subject_scores(test_frame, aggregation), threshold)
                metrics = metrics_from_subjects(test_subjects)
                metrics.update({"fold": fold_idx, "threshold": threshold, "train_balanced_accuracy": train_ba})
                threshold_only_folds.append(metrics)

                fold_predictions = test_subjects.copy()
                fold_predictions["strategy"] = "threshold_only"
                fold_predictions["aggregation_method"] = aggregation
                fold_predictions["fold"] = fold_idx
                fold_predictions["threshold"] = threshold
                all_predictions.append(fold_predictions)

            summary["threshold_only"][aggregation] = {
                "folds": threshold_only_folds,
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in threshold_only_folds])),
                "auc_mean": float(np.mean([row["auc"] for row in threshold_only_folds if row["auc"] is not None])),
            }

        y_cz = cz_merged["label"].reset_index(drop=True)
        for model_type in args.model_types:
            fit_fn = MODEL_TRAINERS[model_type]
            summary["mixed_head_refit"][model_type] = {}
            for aggregation in args.aggregation_methods:
                fold_rows: list[dict[str, float | int | None]] = []
                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_en, y_en, groups=groups), start=1):
                    model = _fit_czech_plus_english(
                        fit_fn,
                        X_cz,
                        y_cz,
                        X_en[train_idx],
                        y_en.iloc[train_idx],
                    )
                    train_scores = _english_scores_from_model(model, X_en[train_idx])
                    test_scores = _english_scores_from_model(model, X_en[test_idx])

                    train_frame = en_merged.iloc[train_idx][["subject_id", "label"]].copy()
                    train_frame["score_patients"] = train_scores
                    threshold, train_ba = best_threshold(aggregate_subject_scores(train_frame, aggregation))

                    test_frame = en_merged.iloc[test_idx][["subject_id", "label"]].copy()
                    test_frame["score_patients"] = test_scores
                    test_subjects = classify_subjects(aggregate_subject_scores(test_frame, aggregation), threshold)
                    metrics = metrics_from_subjects(test_subjects)
                    metrics.update({"fold": fold_idx, "threshold": threshold, "train_balanced_accuracy": train_ba})
                    fold_rows.append(metrics)

                    fold_predictions = test_subjects.copy()
                    fold_predictions["strategy"] = f"mixed_head_refit:{model_type}"
                    fold_predictions["aggregation_method"] = aggregation
                    fold_predictions["fold"] = fold_idx
                    fold_predictions["threshold"] = threshold
                    all_predictions.append(fold_predictions)

                summary["mixed_head_refit"][model_type][aggregation] = {
                    "folds": fold_rows,
                    "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
                    "auc_mean": float(np.mean([row["auc"] for row in fold_rows if row["auc"] is not None])),
                }

        report_path = Path(args.report_json)
        predictions_path = Path(args.subject_predictions_csv)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.concat(all_predictions, ignore_index=True).to_csv(predictions_path, index=False)

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "adapt_english_minimal.py",
                "dataset_id": CONFIG.datasets.en_dataset_id,
                "encoder_model": CONFIG.encoder.model_id,
            }
        )
        log_params(
            {
                "cz_sample_manifest": args.cz_sample_manifest,
                "cz_embedding_manifest": args.cz_embedding_manifest,
                "en_sample_manifest": args.en_sample_manifest,
                "en_embedding_manifest": args.en_embedding_manifest,
                "czech_model": args.czech_model,
                "model_types": args.model_types,
                "aggregation_methods": args.aggregation_methods,
                "n_splits": args.n_splits,
                "random_state": args.random_state,
                "tracking_uri": tracking_uri,
            }
        )
        log_metrics(summary, prefix="adaptation.")
        log_dict("metrics/english_minimal_adaptation.json", summary)
        log_artifact(report_path, artifact_path="reports")
        log_artifact(predictions_path, artifact_path="predictions")

        print(f"Saved adaptation report to: {report_path}")
        print(f"Saved adaptation subject predictions to: {predictions_path}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
