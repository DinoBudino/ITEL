from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from research.src.config import CONFIG
from research.src.preprocessing import load_and_preprocess_audio


METRIC_COLUMNS = [
    "raw_duration_s",
    "channels",
    "raw_rms_mean",
    "raw_rms_std",
    "raw_zcr_mean",
    "raw_flatness_mean",
    "proc_duration_s",
    "proc_retention_ratio",
]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _is_excluded_flag(value: object) -> bool:
    text = _clean_text(value).lower()
    return bool(text) and text.startswith("exclude_")


def _is_auto_generated_qc_flag(value: object) -> bool:
    text = _clean_text(value).lower()
    return text in {"", "needs_label", "exclude_auto_noise", "review_auto_noise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-flag noisy English external recordings.")
    parser.add_argument(
        "--metadata",
        default=str(CONFIG.paths.manifests_dir / "english_external_metadata.csv"),
        help="English metadata CSV to update.",
    )
    parser.add_argument(
        "--predictions",
        default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_english_external_seg_5s.csv"),
        help="Optional predictions CSV used to spot extreme model-label disagreement.",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(CONFIG.paths.raw_en_dir),
        help="Root directory for English audio files.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "english_external_auto_qc_summary.csv"),
        help="Where to save the QC summary CSV.",
    )
    parser.add_argument(
        "--force-overwrite-existing",
        action="store_true",
        help="Allow replacing existing qc_flag values. By default manual flags are preserved.",
    )
    return parser.parse_args()


def _extract_metrics(audio_path: Path) -> dict[str, float | int]:
    info = sf.info(str(audio_path))
    raw_samples, raw_sr = librosa.load(audio_path, sr=CONFIG.audio.target_sr, mono=True)
    proc_samples, proc_sr = load_and_preprocess_audio(audio_path, duration_s=None, profile="anti_confound")

    raw_rms = librosa.feature.rms(y=raw_samples)[0] if len(raw_samples) else np.array([0.0], dtype=np.float32)
    raw_zcr = librosa.feature.zero_crossing_rate(y=raw_samples)[0] if len(raw_samples) else np.array([0.0], dtype=np.float32)
    raw_flatness = librosa.feature.spectral_flatness(y=raw_samples)[0] if len(raw_samples) else np.array([0.0], dtype=np.float32)

    raw_duration_s = float(len(raw_samples) / raw_sr) if raw_sr else 0.0
    proc_duration_s = float(len(proc_samples) / proc_sr) if proc_sr else 0.0
    retention_ratio = proc_duration_s / raw_duration_s if raw_duration_s > 0 else 0.0

    return {
        "raw_duration_s": raw_duration_s,
        "channels": int(info.channels),
        "raw_rms_mean": float(np.mean(raw_rms)),
        "raw_rms_std": float(np.std(raw_rms)),
        "raw_zcr_mean": float(np.mean(raw_zcr)),
        "raw_flatness_mean": float(np.mean(raw_flatness)),
        "proc_duration_s": proc_duration_s,
        "proc_retention_ratio": retention_ratio,
    }


def _build_thresholds(feature_frame: pd.DataFrame) -> dict[str, float]:
    return {
        "raw_flatness_mean_p95": float(feature_frame["raw_flatness_mean"].quantile(0.95)),
        "raw_flatness_mean_p97": float(feature_frame["raw_flatness_mean"].quantile(0.97)),
        "raw_zcr_mean_p95": float(feature_frame["raw_zcr_mean"].quantile(0.95)),
        "raw_zcr_mean_p99": float(feature_frame["raw_zcr_mean"].quantile(0.99)),
        "proc_retention_ratio_p10": float(feature_frame["proc_retention_ratio"].quantile(0.10)),
        "proc_retention_ratio_p05": float(feature_frame["proc_retention_ratio"].quantile(0.05)),
    }


def _merge_predictions(feature_frame: pd.DataFrame, predictions_csv: Path) -> pd.DataFrame:
    if not predictions_csv.exists():
        feature_frame["score_patients"] = np.nan
        feature_frame["pred_label"] = ""
        return feature_frame
    predictions = pd.read_csv(predictions_csv)
    keep_cols = ["subject_id", "score_patients", "pred_label"]
    predictions = predictions[keep_cols].copy()
    predictions["subject_id"] = predictions["subject_id"].astype(str)
    return feature_frame.merge(predictions, on="subject_id", how="left")


def _decide_qc(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str, str]:
    severe_acoustic: list[str] = []
    review_reasons: list[str] = []

    if row["raw_zcr_mean"] >= thresholds["raw_zcr_mean_p99"]:
        severe_acoustic.append("extreme_raw_zcr")
    elif row["raw_zcr_mean"] >= thresholds["raw_zcr_mean_p95"]:
        review_reasons.append("high_raw_zcr")

    if row["raw_flatness_mean"] >= thresholds["raw_flatness_mean_p97"]:
        severe_acoustic.append("extreme_raw_flatness")
    elif row["raw_flatness_mean"] >= thresholds["raw_flatness_mean_p95"]:
        review_reasons.append("high_raw_flatness")

    if row["proc_retention_ratio"] <= thresholds["proc_retention_ratio_p05"]:
        severe_acoustic.append("very_low_retention")
    elif row["proc_retention_ratio"] <= thresholds["proc_retention_ratio_p10"]:
        review_reasons.append("low_retention")

    extreme_model_disagreement = False
    score = row.get("score_patients")
    label = _clean_text(row.get("label"))
    if pd.notna(score):
        score = float(score)
        if label == "Healthy" and score >= 0.95:
            extreme_model_disagreement = True
            review_reasons.append("extreme_model_fp")
        elif label == "Patients" and score <= 0.05:
            review_reasons.append("extreme_model_fn")
        elif (label == "Healthy" and score >= 0.80) or (label == "Patients" and score <= 0.20):
            review_reasons.append("strong_model_disagreement")

    unique_severe = list(dict.fromkeys(severe_acoustic))
    unique_review_reasons = [reason for reason in dict.fromkeys(review_reasons) if reason not in unique_severe]

    if unique_severe and (len(unique_severe) >= 2 or extreme_model_disagreement):
        return "exclude_auto_noise", "; ".join(unique_severe), "; ".join(unique_review_reasons)
    if unique_severe:
        return "review_auto_noise", "; ".join(unique_severe), "; ".join(unique_review_reasons)
    if unique_review_reasons:
        return "review_auto_noise", "; ".join(unique_review_reasons), ""
    return "", "", ""


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    predictions_path = Path(args.predictions)
    audio_dir = Path(args.audio_dir)
    summary_path = Path(args.summary_output)

    metadata = pd.read_csv(metadata_path)
    feature_rows: list[dict[str, object]] = []
    for row in metadata.to_dict(orient="records"):
        file_name = _clean_text(row.get("file_name"))
        audio_path = audio_dir / file_name
        feature_row = {
            "file_name": file_name,
            "subject_id": _clean_text(row.get("subject_id")),
            "label": _clean_text(row.get("label")),
        }
        feature_row.update(_extract_metrics(audio_path))
        feature_rows.append(feature_row)

    feature_frame = pd.DataFrame(feature_rows)
    feature_frame = _merge_predictions(feature_frame, predictions_path)
    thresholds = _build_thresholds(feature_frame)

    summaries: list[dict[str, object]] = []
    updated_rows: list[dict[str, object]] = []
    for row in metadata.to_dict(orient="records"):
        file_name = _clean_text(row.get("file_name"))
        features = feature_frame.loc[feature_frame["file_name"] == file_name].iloc[0]
        suggested_flag, qc_reason, review_hint = _decide_qc(features, thresholds)
        existing_flag = _clean_text(row.get("qc_flag"))
        existing_notes = _clean_text(row.get("notes"))

        applied_flag = existing_flag
        can_update_flag = args.force_overwrite_existing or _is_auto_generated_qc_flag(existing_flag)
        if can_update_flag:
            if suggested_flag:
                applied_flag = suggested_flag
            elif existing_flag in {"exclude_auto_noise", "review_auto_noise"}:
                applied_flag = ""

        if can_update_flag:
            row["qc_flag"] = applied_flag

        auto_note_parts = []
        if qc_reason:
            auto_note_parts.append(f"auto_qc_reason={qc_reason}")
        if review_hint:
            auto_note_parts.append(f"auto_qc_review={review_hint}")
        note_text = " | ".join(auto_note_parts)
        if note_text and note_text not in existing_notes:
            row["notes"] = f"{existing_notes} | {note_text}".strip(" |")

        summaries.append(
            {
                "file_name": file_name,
                "subject_id": features["subject_id"],
                "label": features["label"],
                **{column: features[column] for column in METRIC_COLUMNS},
                "score_patients": features.get("score_patients"),
                "pred_label": features.get("pred_label", ""),
                "suggested_qc_flag": suggested_flag,
                "suggested_qc_reason": qc_reason,
                "review_hint": review_hint,
                "applied_qc_flag": applied_flag,
            }
        )
        updated_rows.append(row)

    updated = pd.DataFrame(updated_rows)
    updated.to_csv(metadata_path, index=False)

    summary = pd.DataFrame(summaries)
    for key, value in thresholds.items():
        summary[key] = value
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    applied = summary.loc[summary["applied_qc_flag"].astype(str).str.startswith("exclude_")].copy()
    review = summary.loc[summary["applied_qc_flag"].astype(str).str.startswith("review_")].copy()
    print(f"Updated metadata: {metadata_path}")
    print(f"Saved QC summary: {summary_path}")
    print(f"Auto-excluded rows: {len(applied)}")
    print(f"Auto-review rows: {len(review)}")
    if not applied.empty:
        print(applied[["subject_id", "file_name", "applied_qc_flag", "suggested_qc_reason"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
