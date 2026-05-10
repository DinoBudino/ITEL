from __future__ import annotations

import argparse
from pathlib import Path
import sys

import librosa
import mlflow
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_params


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _load_audio_metrics(audio_path: Path) -> dict[str, float]:
    samples, sr = librosa.load(audio_path, sr=CONFIG.audio.target_sr, mono=True)
    duration_s = float(len(samples) / sr) if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    zcr = float(librosa.feature.zero_crossing_rate(y=samples)[0].mean()) if len(samples) else 0.0
    flatness = float(librosa.feature.spectral_flatness(y=samples)[0].mean()) if len(samples) else 0.0
    silence_ratio = float(np.mean(np.abs(samples) < 0.01)) if len(samples) else 1.0
    return {
        "duration_s": duration_s,
        "raw_rms_mean": rms,
        "raw_zcr_mean": zcr,
        "raw_flatness_mean": flatness,
        "silence_ratio": silence_ratio,
    }


def _thresholds(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "duration_p10": float(frame["duration_s"].quantile(0.10)),
        "duration_p05": float(frame["duration_s"].quantile(0.05)),
        "zcr_p90": float(frame["raw_zcr_mean"].quantile(0.90)),
        "zcr_p95": float(frame["raw_zcr_mean"].quantile(0.95)),
        "flatness_p90": float(frame["raw_flatness_mean"].quantile(0.90)),
        "flatness_p95": float(frame["raw_flatness_mean"].quantile(0.95)),
        "silence_p90": float(frame["silence_ratio"].quantile(0.90)),
        "silence_p95": float(frame["silence_ratio"].quantile(0.95)),
        "rms_p10": float(frame["raw_rms_mean"].quantile(0.10)),
        "rms_p05": float(frame["raw_rms_mean"].quantile(0.05)),
    }


def _merge_scores(frame: pd.DataFrame, predictions_csv: Path | None) -> pd.DataFrame:
    if predictions_csv is None or not predictions_csv.exists():
        frame["score_patients"] = np.nan
        frame["pred_label"] = ""
        return frame
    preds = pd.read_csv(predictions_csv)
    keep = [col for col in ["subject_id", "score_patients", "pred_label"] if col in preds.columns]
    preds = preds[keep].copy()
    preds["subject_id"] = preds["subject_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    return frame.merge(preds, on="subject_id", how="left")


def _decide_qc(row: pd.Series, thr: dict[str, float]) -> tuple[str, str]:
    severe: list[str] = []
    review: list[str] = []

    if row["duration_s"] <= max(30.0, thr["duration_p05"]):
        severe.append("very_short_recording")
    elif row["duration_s"] <= max(60.0, thr["duration_p10"]):
        review.append("short_recording")

    if row["raw_zcr_mean"] >= thr["zcr_p95"]:
        severe.append("extreme_raw_zcr")
    elif row["raw_zcr_mean"] >= thr["zcr_p90"]:
        review.append("high_raw_zcr")

    if row["raw_flatness_mean"] >= thr["flatness_p95"]:
        severe.append("extreme_raw_flatness")
    elif row["raw_flatness_mean"] >= thr["flatness_p90"]:
        review.append("high_raw_flatness")

    if row["silence_ratio"] >= max(0.92, thr["silence_p95"]):
        severe.append("extreme_silence_ratio")
    elif row["silence_ratio"] >= max(0.88, thr["silence_p90"]):
        review.append("high_silence_ratio")

    if row["raw_rms_mean"] <= min(0.008, thr["rms_p05"]):
        severe.append("very_low_rms")
    elif row["raw_rms_mean"] <= min(0.012, thr["rms_p10"]):
        review.append("low_rms")

    label = _clean_text(row.get("label"))
    score = row.get("score_patients")
    if pd.notna(score):
        score = float(score)
        if label == "Healthy" and score >= 0.98:
            review.append("extreme_model_fp")
        elif label == "Patients" and score <= 0.02:
            review.append("extreme_model_fn")

    severe = list(dict.fromkeys(severe))
    review = list(dict.fromkeys(reason for reason in review if reason not in severe))
    if severe:
        return "exclude_clean_v1", "; ".join(severe + review)
    if review:
        return "review_clean_v1", "; ".join(review)
    return "", ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create english_clean_v1 manifest using automatic audio QC.")
    parser.add_argument("--input-manifest", default=str(CONFIG.paths.manifests_dir / "english_test_samples.csv"))
    parser.add_argument("--predictions", default=str(CONFIG.paths.reports_dir / "CSV" / "english_test" / "predictions_english_external_english_test_embeddings_seg_5s.csv"))
    parser.add_argument("--summary-output", default=str(CONFIG.paths.reports_dir / "CSV" / "english_test" / "english_test_clean_v1_summary.csv"))
    parser.add_argument("--output-manifest", default=str(CONFIG.paths.manifests_dir / "english_test_clean_v1.csv"))
    parser.add_argument("--include-review", action="store_true", help="Keep rows flagged review_clean_v1 in the exported manifest. Excluded rows are always removed.")
    parser.add_argument("--mlflow-experiment", default="itel-data-prep")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    manifest_path = Path(args.input_manifest)
    summary_path = Path(args.summary_output)
    output_manifest_path = Path(args.output_manifest)
    predictions_path = Path(args.predictions)

    manifest = pd.read_csv(manifest_path)
    rows = []
    for row in manifest.to_dict(orient="records"):
        audio_path = Path(str(row["raw_audio_path"]))
        metrics = _load_audio_metrics(audio_path)
        rows.append({
            "sample_id": row["sample_id"],
            "subject_id": str(row["subject_id"]),
            "label": row["label"],
            "raw_audio_path": str(audio_path),
            **metrics,
        })
    qc_frame = pd.DataFrame(rows)
    qc_frame = _merge_scores(qc_frame, predictions_path if predictions_path.exists() else None)
    thr = _thresholds(qc_frame)
    qc_flags = qc_frame.apply(lambda row: _decide_qc(row, thr), axis=1)
    qc_frame["suggested_qc_flag"] = [flag for flag, _ in qc_flags]
    qc_frame["suggested_qc_reason"] = [reason for _, reason in qc_flags]

    updated = manifest.merge(qc_frame[["sample_id", "duration_s", "raw_rms_mean", "raw_zcr_mean", "raw_flatness_mean", "silence_ratio", "score_patients", "pred_label", "suggested_qc_flag", "suggested_qc_reason"]], on="sample_id", how="left")
    updated["qc_flag"] = updated["suggested_qc_flag"].fillna("")
    updated["review_reason"] = updated["suggested_qc_reason"].replace("", pd.NA).fillna(updated.get("review_reason"))
    updated["review_status"] = np.where(updated["qc_flag"] == "", "approved", "needs_review")
    updated["duration_s"] = updated["duration_s"]
    updated["notes"] = updated["notes"].fillna("").astype(str).str.strip()
    updated.loc[updated["notes"] == "", "notes"] = "english_clean_v1_auto_qc"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    qc_frame.to_csv(summary_path, index=False)

    keep_mask = ~updated["qc_flag"].fillna("").astype(str).str.startswith("exclude_")
    if not args.include_review:
        keep_mask &= ~updated["qc_flag"].fillna("").astype(str).str.startswith("review_")
    clean_manifest = updated.loc[keep_mask, manifest.columns].copy()
    clean_manifest.to_csv(output_manifest_path, index=False)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.set_tags({"stage": "research", "script": "build_en_clean_manifest.py", "dataset_id": CONFIG.datasets.en_dataset_id, "cleaning_version": "english_clean_v1"})
        log_params({"input_manifest": str(manifest_path), "predictions": str(predictions_path) if predictions_path.exists() else None, "summary_output": str(summary_path), "output_manifest": str(output_manifest_path), "include_review": args.include_review, "tracking_uri": tracking_uri, **thr})
        log_dict(
            "metadata/english_clean_v1_summary.json",
            {
                "n_input_rows": int(len(manifest)),
                "n_output_rows": int(len(clean_manifest)),
                "n_excluded": int((updated["qc_flag"] == "exclude_clean_v1").sum()),
                "n_review": int((updated["qc_flag"] == "review_clean_v1").sum()),
                "kept_labels": {} if clean_manifest.empty else {str(k): int(v) for k, v in clean_manifest["label"].value_counts().to_dict().items()},
            },
        )
        log_artifact(summary_path, artifact_path="qc")
        log_artifact(output_manifest_path, artifact_path="manifests")

    print(f"Saved QC summary to: {summary_path}")
    print(f"Saved clean manifest to: {output_manifest_path}")
    print(f"Excluded rows: {(updated['qc_flag'] == 'exclude_clean_v1').sum()}")
    print(f"Review rows: {(updated['qc_flag'] == 'review_clean_v1').sum()}")


if __name__ == "__main__":
    main()
