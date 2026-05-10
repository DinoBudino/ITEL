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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from research.src.config import CONFIG


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Czech baseline analysis report.")
    parser.add_argument("--sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name))
    parser.add_argument("--sample-predictions", default=str(REPORT_CSV_DIR / "predictions_cz_internal_cv_samples.csv"))
    parser.add_argument("--subject-predictions", default=str(REPORT_CSV_DIR / "predictions_cz_internal_cv_subjects.csv"))
    parser.add_argument("--metrics-json", default=str(REPORT_CSV_DIR / "metrics_cz_internal_cv.json"))
    parser.add_argument("--audit-json", default=str(REPORT_CSV_DIR / "audit_cz_recording_confounds.json"))
    parser.add_argument("--report-json", default=str(REPORT_CSV_DIR / "analyze_cz_baseline.json"))
    parser.add_argument("--task-metrics-csv", default=str(REPORT_CSV_DIR / "analyze_cz_baseline_task_metrics.csv"))
    parser.add_argument("--errors-csv", default=str(REPORT_CSV_DIR / "analyze_cz_baseline_subject_errors.csv"))
    return parser.parse_args()


def _safe_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    y_bin = (y_true.astype(str) == "Patients").astype(int)
    if y_bin.nunique() < 2:
        return None
    return float(roc_auc_score(y_bin, scores.astype(float)))


def _confusion_payload(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int | None]:
    labels = ["Healthy", "Patients"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def _threshold_rows(frame: pd.DataFrame) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        threshold = round(float(threshold), 2)
        pred = np.where(frame["score_patients_mean"] >= threshold, "Patients", "Healthy")
        payload = _confusion_payload(frame["label"], pd.Series(pred, index=frame.index))
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": float(balanced_accuracy_score(frame["label"], pred)),
                "accuracy": float(accuracy_score(frame["label"], pred)),
                "f1_patients": float(f1_score(frame["label"], pred, pos_label="Patients")),
                **payload,
            }
        )
    return rows


def _task_metrics(sample_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_type, task_frame in sample_predictions.groupby("task_type", dropna=False):
        subject_frame = (
            task_frame.groupby(["subject_id", "label"], as_index=False)["score_patients"]
            .mean()
            .rename(columns={"score_patients": "score_patients_mean"})
        )
        subject_frame["pred_label"] = np.where(subject_frame["score_patients_mean"] >= 0.5, "Patients", "Healthy")
        rows.append(
            {
                "task_type": task_type,
                "n_samples": int(len(task_frame)),
                "n_subjects": int(task_frame["subject_id"].nunique()),
                "sample_balanced_accuracy": float(balanced_accuracy_score(task_frame["label"], task_frame["pred_label"])),
                "sample_auc": _safe_auc(task_frame["label"], task_frame["score_patients"]),
                "subject_balanced_accuracy": float(balanced_accuracy_score(subject_frame["label"], subject_frame["pred_label"])),
                "subject_auc": _safe_auc(subject_frame["label"], subject_frame["score_patients_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by="subject_balanced_accuracy", ascending=False).reset_index(drop=True)


def _duration_summary(sample_manifest: pd.DataFrame, audit: dict[str, object]) -> dict[str, object]:
    frame = sample_manifest.copy()
    frame["duration_s"] = pd.to_numeric(frame.get("duration_s"), errors="coerce")
    if frame["duration_s"].notna().sum() == 0:
        feature_cache = audit.get("config", {}).get("feature_cache")
        if feature_cache and Path(feature_cache).exists():
            cached = pd.read_csv(feature_cache, usecols=["sample_id", "duration_s"])
            frame = frame.drop(columns=["duration_s"], errors="ignore").merge(cached, on="sample_id", how="left")
    by_label = frame.groupby("label")["duration_s"].agg(["mean", "std", "median", "min", "max"]).round(6)
    by_task_label = (
        frame.groupby(["task_type", "label"])["duration_s"]
        .agg(["mean", "std", "median", "min", "max"])
        .round(6)
        .reset_index()
    )
    return {
        "by_label": by_label.to_dict(orient="index"),
        "by_task_label": by_task_label.to_dict(orient="records"),
    }


def _subject_error_table(subject_predictions: pd.DataFrame, sample_manifest: pd.DataFrame) -> pd.DataFrame:
    subject_stats = (
        sample_manifest.groupby(["subject_id", "label"], as_index=False)
        .agg(n_samples=("sample_id", "count"), n_tasks=("task_type", "nunique"))
    )
    task_mix = (
        sample_manifest.groupby(["subject_id", "task_type"]).size().rename("n_task_samples").reset_index()
    )
    dominant_task = task_mix.sort_values(["subject_id", "n_task_samples", "task_type"], ascending=[True, False, True]).drop_duplicates("subject_id")
    errors = subject_predictions[subject_predictions["label"] != subject_predictions["pred_label"]].copy()
    errors = errors.merge(subject_stats, on=["subject_id", "label"], how="left")
    errors = errors.merge(dominant_task[["subject_id", "task_type", "n_task_samples"]], on="subject_id", how="left")
    errors = errors.rename(columns={"task_type": "dominant_task_type"})
    return errors.sort_values(by="score_patients_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    sample_manifest = pd.read_csv(args.sample_manifest)
    sample_predictions = pd.read_csv(args.sample_predictions)
    subject_predictions = pd.read_csv(args.subject_predictions)
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))

    merged_samples = sample_predictions.merge(
        sample_manifest[["sample_id", "subject_id", "label", "task_type", "duration_s"]],
        left_on=["source_sample_id", "subject_id", "label", "task_type"],
        right_on=["sample_id", "subject_id", "label", "task_type"],
        how="left",
        suffixes=("", "_manifest"),
    )

    task_metrics = _task_metrics(merged_samples)
    threshold_rows = _threshold_rows(subject_predictions)
    best_threshold = max(threshold_rows, key=lambda row: (row["balanced_accuracy"], row["specificity"], row["accuracy"]))
    errors = _subject_error_table(subject_predictions, sample_manifest)

    subject_counts = sample_manifest.groupby(["subject_id", "label"], as_index=False).size().rename(columns={"size": "n_samples"})
    sample_count_summary = subject_counts.groupby("label")["n_samples"].agg(["mean", "std", "median", "min", "max"]).round(4)
    score_summary = subject_predictions.groupby("label")["score_patients_mean"].agg(["mean", "std", "median", "min", "max"]).round(6)

    single_feature_auc = audit.get("feature_summary", {})
    top_proxy_features = []
    for feature_name, payload in single_feature_auc.items():
        top_proxy_features.append(
            {
                "feature": feature_name,
                "auc": payload.get("single_feature_auc"),
                "auc_abs_distance_from_half": payload.get("single_feature_auc_abs_distance_from_half"),
            }
        )
    top_proxy_features = sorted(top_proxy_features, key=lambda row: (row["auc_abs_distance_from_half"] is None, -(row["auc_abs_distance_from_half"] or -1.0)))

    report = {
        "dataset": {
            "n_subjects": int(sample_manifest["subject_id"].nunique()),
            "n_samples": int(len(sample_manifest)),
            "subjects_by_label": sample_manifest[["subject_id", "label"]].drop_duplicates()["label"].value_counts().to_dict(),
            "samples_by_label": sample_manifest["label"].value_counts().to_dict(),
            "samples_per_subject_by_label": sample_count_summary.to_dict(orient="index"),
            "task_counts_by_label": sample_manifest.groupby(["task_type", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
            "duration_summary": _duration_summary(sample_manifest, audit),
        },
        "baseline": {
            "subject_level": metrics["subject_level"],
            "sample_level": metrics["sample_level"],
            "subject_score_summary": score_summary.to_dict(orient="index"),
        },
        "error_analysis": {
            "n_subject_errors": int(len(errors)),
            "false_positive_subjects": errors[errors["label"] == "Healthy"]["subject_id"].tolist(),
            "false_negative_subjects": errors[errors["label"] == "Patients"]["subject_id"].tolist(),
        },
        "threshold_analysis": {
            "default_threshold": 0.5,
            "default_confusion": _confusion_payload(subject_predictions["label"], subject_predictions["pred_label"]),
            "best_subject_threshold_by_balanced_accuracy": best_threshold,
            "grid": threshold_rows,
        },
        "task_analysis": task_metrics.to_dict(orient="records"),
        "recording_confound_audit": {
            "proxy_model_metrics": audit.get("proxy_model_metrics", {}),
            "top_single_proxy_features": top_proxy_features[:5],
        },
    }

    report_path = Path(args.report_json)
    task_metrics_path = Path(args.task_metrics_csv)
    errors_path = Path(args.errors_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    task_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    task_metrics.to_csv(task_metrics_path, index=False)
    errors.to_csv(errors_path, index=False)

    print(f"Saved Czech baseline analysis report to: {report_path}")
    print(f"Saved Czech task metrics to: {task_metrics_path}")
    print(f"Saved Czech subject error table to: {errors_path}")
    print(f"Best subject threshold by balanced accuracy: {best_threshold['threshold']:.2f}")
    print(f"False-positive healthy subjects: {', '.join(report['error_analysis']['false_positive_subjects'])}")


if __name__ == "__main__":
    main()
