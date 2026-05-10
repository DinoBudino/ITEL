from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import mlflow
except ModuleNotFoundError:  # pragma: no cover - optional dependency for analysis-only runs
    mlflow = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from research.src.config import CONFIG
from research.src.evaluate import binary_metrics
from research.src.segmentation import aggregate_probabilities
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_metrics, log_params
from research.src.train import load_embeddings


def _positive_class_index(model) -> int:
    classes = list(model.classes_)
    if "Patients" not in classes:
        raise ValueError(f"Expected 'Patients' in model classes, found: {classes}")
    return classes.index("Patients")


def _is_excluded_flag(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text) and text.startswith("exclude_")


def _subset_frames(per_source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subsets = {"all": per_source}
    if "review_status" in per_source.columns:
        approved = per_source[per_source["review_status"].fillna("").astype(str).str.strip().str.lower() == "approved"].copy()
        if not approved.empty:
            subsets["approved_only"] = approved
    if "auto_qc_flag" in per_source.columns:
        clean_auto_qc = per_source[~per_source["auto_qc_flag"].apply(_is_excluded_flag)].copy()
        if len(clean_auto_qc) != len(per_source):
            subsets["auto_qc_clean"] = clean_auto_qc
        suspect_auto_qc = per_source[per_source["auto_qc_flag"].fillna("").astype(str).str.strip() != ""].copy()
        if not suspect_auto_qc.empty:
            subsets["flagged_auto_qc"] = suspect_auto_qc
    return subsets


def _filter_sample_manifest(sample_manifest: pd.DataFrame, approved_only: bool) -> pd.DataFrame:
    filtered = sample_manifest.copy()
    if approved_only and "review_status" in filtered.columns:
        review_values = filtered["review_status"].fillna("").astype(str).str.strip().str.lower()
        filtered = filtered.loc[review_values == "approved"].copy()
    return filtered


def _filter_segment_rows(merged: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    return merged


def _default_aggregation_method(mode_name: str) -> str:
    if mode_name in {"seg_2s", "seg_5s"}:
        return "median"
    return "mean"


def _aggregate_scores(series: pd.Series, method: str) -> float:
    values = [float(value) for value in series.dropna().tolist()]
    return aggregate_probabilities(values, method=method)


def _threshold_grid() -> list[float]:
    return [round(value, 2) for value in np.arange(0.05, 1.0, 0.05).tolist()]


def _predictions_from_threshold(scores: pd.Series, threshold: float) -> pd.Series:
    return scores.apply(lambda score: "Patients" if float(score) >= threshold else "Healthy")


def _threshold_sweep(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"grid": [], "best_by_balanced_accuracy": None}

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    for threshold in _threshold_grid():
        pred_labels = _predictions_from_threshold(frame["score_patients"], threshold)
        bal_acc = float(balanced_accuracy_score(frame["label"], pred_labels))
        row = {
            "threshold": threshold,
            "balanced_accuracy": bal_acc,
            "accuracy": float((pred_labels == frame["label"]).mean()),
            "n_pred_patients": int((pred_labels == "Patients").sum()),
            "n_pred_healthy": int((pred_labels == "Healthy").sum()),
        }
        rows.append(row)
        if best_row is None or row["balanced_accuracy"] > best_row["balanced_accuracy"]:
            best_row = row

    return {
        "grid": rows,
        "best_by_balanced_accuracy": best_row,
    }


def _evaluate_manifest(
    model,
    sample_manifest: pd.DataFrame,
    embedding_manifest_path: Path,
    *,
    aggregation_method: str | None = None,
    decision_threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    embedding_manifest = pd.read_csv(embedding_manifest_path)
    mode_name = embedding_manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
    aggregation = aggregation_method or _default_aggregation_method(mode_name)
    merged = embedding_manifest.merge(
        sample_manifest[["sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_sample"),
    )
    merged = merged.dropna(subset=["label"]).copy()
    merged = _filter_segment_rows(merged, mode_name)
    if merged.empty:
        empty_predictions = pd.DataFrame(
            columns=[
                "sample_id",
                "subject_id",
                "label",
                "review_status",
                "review_reason",
                "auto_qc_flag",
                "qc_flag",
                "score_patients",
                "aggregation_method",
                "pred_label",
            ]
        )
        metrics = binary_metrics(pd.Series(dtype="object"), pd.Series(dtype="object"), pd.Series(dtype="float64"))
        metrics["aggregation_method"] = aggregation
        metrics["decision_threshold"] = decision_threshold
        metrics["threshold_sweep"] = {"grid": [], "best_by_balanced_accuracy": None}
        metrics["subsets"] = {
            "all": {
                **binary_metrics(pd.Series(dtype="object"), pd.Series(dtype="object"), pd.Series(dtype="float64")),
                "aggregation_method": aggregation,
                "decision_threshold": decision_threshold,
                "threshold_sweep": {"grid": [], "best_by_balanced_accuracy": None},
            }
        }
        return metrics, empty_predictions
    X, frame = load_embeddings(merged)
    positive_index = _positive_class_index(model)
    frame = frame.copy()
    frame["score_patients"] = model.predict_proba(X)[:, positive_index]

    per_source = (
        frame.groupby(
            ["source_sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"],
            as_index=False,
            dropna=False,
        )["score_patients"]
        .agg(lambda values: _aggregate_scores(values, aggregation))
        .rename(columns={"source_sample_id": "sample_id"})
    )
    per_source["aggregation_method"] = aggregation
    if mode_name in {"seg_2s", "seg_5s"}:
        segment_counts = frame.groupby("source_sample_id", dropna=False).size().rename("n_segments_used").reset_index()
        per_source = per_source.merge(segment_counts, left_on="sample_id", right_on="source_sample_id", how="left").drop(columns=["source_sample_id"])
    per_source["decision_threshold"] = decision_threshold
    per_source["pred_label"] = _predictions_from_threshold(per_source["score_patients"], decision_threshold)
    metrics = binary_metrics(per_source["label"], per_source["pred_label"], per_source["score_patients"])
    metrics["aggregation_method"] = aggregation
    metrics["decision_threshold"] = decision_threshold
    metrics["threshold_sweep"] = _threshold_sweep(per_source)
    subset_metrics = {}
    for subset_name, subset_frame in _subset_frames(per_source).items():
        subset_metrics[subset_name] = binary_metrics(subset_frame["label"], subset_frame["pred_label"], subset_frame["score_patients"])
        subset_metrics[subset_name]["aggregation_method"] = aggregation
        subset_metrics[subset_name]["decision_threshold"] = decision_threshold
        subset_metrics[subset_name]["threshold_sweep"] = _threshold_sweep(subset_frame)
    metrics["subsets"] = subset_metrics
    return metrics, per_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Czech baseline on the English external corpus.")
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name),
        help="Path to the English external sample manifest.",
    )
    parser.add_argument(
        "--model",
        default=str(CONFIG.paths.model_dir / "xlsr300m_mlp_small_cz.joblib"),
        help="Path to the trained Czech MLP model.",
    )
    parser.add_argument(
        "--embedding-manifests",
        nargs="+",
        default=[
            str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_2s.csv"),
            str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_5s.csv"),
            str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_full_audio.csv"),
        ],
        help="One or more embedding manifest CSVs to evaluate.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(CONFIG.paths.reports_dir / "CSV"),
        help="Directory where metrics and predictions will be written.",
    )
    parser.add_argument(
        "--approved-only",
        action="store_true",
        help="Evaluate only rows with review_status=approved from the English sample manifest.",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["mean", "median"],
        default=None,
        help="Override per-file score aggregation. Defaults to median for segmented modes and mean for full_audio.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert score_patients into class predictions.",
    )
    parser.add_argument("--mlflow-experiment", default="itel-evaluations", help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to a local file store under research_results/mlruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if mlflow is None:
        raise ModuleNotFoundError("mlflow is required to run evaluate_external.py directly.")
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    sample_manifest = _filter_sample_manifest(pd.read_csv(args.sample_manifest), approved_only=args.approved_only)
    model = joblib.load(args.model)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        summary: dict[str, dict] = {}
        if sample_manifest.empty:
            print("No English rows available after applying the current manifest filter.")
        for manifest_str in args.embedding_manifests:
            manifest_path = Path(manifest_str)
            metrics, predictions = _evaluate_manifest(
                model,
                sample_manifest,
                manifest_path,
                aggregation_method=args.aggregation_method,
                decision_threshold=args.decision_threshold,
            )
            mode_name = manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
            summary[mode_name] = metrics

            predictions_path = reports_dir / f"predictions_{CONFIG.datasets.en_dataset_id}_{mode_name}.csv"
            predictions.to_csv(predictions_path, index=False)
            log_artifact(predictions_path, artifact_path=f"predictions/{mode_name}")
            print(f"Saved predictions to: {predictions_path}")

        metrics_path = reports_dir / f"metrics_{CONFIG.datasets.en_dataset_id}.json"
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "evaluate_external.py",
                "dataset_id": CONFIG.datasets.en_dataset_id,
                "model_path": str(args.model),
            }
        )
        log_params(
            {
                "sample_manifest": args.sample_manifest,
                "model": args.model,
                "embedding_manifests": args.embedding_manifests,
                "reports_dir": str(reports_dir),
                "approved_only": args.approved_only,
                "aggregation_method": args.aggregation_method,
                "decision_threshold": args.decision_threshold,
                "tracking_uri": tracking_uri,
            }
        )
        log_metrics(summary, prefix="external.")
        log_dict("metrics/english_external_metrics.json", summary)
        log_artifact(metrics_path, artifact_path="metrics")

        print(f"Saved metrics summary to: {metrics_path}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
