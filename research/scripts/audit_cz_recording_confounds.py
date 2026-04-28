from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from research.src.config import CONFIG
from research.src.evaluate import binary_metrics
from research.src.preprocessing import load_and_preprocess_audio


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the Czech baseline may exploit recording-condition differences.")
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Path to the Czech sample manifest.",
    )
    parser.add_argument(
        "--feature-cache",
        default=str(CONFIG.paths.processed_manifest_dir / "cz_main_recording_proxy_features.csv"),
        help="CSV cache for extracted recording proxy features.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(REPORT_CSV_DIR / "audit_cz_recording_confounds.json"),
        help="Where to save the audit metrics JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        default=str(REPORT_CSV_DIR / "audit_cz_recording_confounds_subject_predictions.csv"),
        help="Where to save subject-level predictions for each proxy model.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of subject-disjoint folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute cached proxy features even if the cache exists.",
    )
    parser.add_argument(
        "--preproc-profile",
        default="anti_confound",
        choices=["standard", "anti_confound"],
        help="Preprocessing profile applied before extracting audit proxy features.",
    )
    return parser.parse_args()


def _safe_auc(y_true: pd.Series, values: pd.Series) -> float | None:
    try:
        return float(roc_auc_score((y_true == "Patients").astype(int), values))
    except ValueError:
        return None


def _extract_audio_features(audio_path: str, preproc_profile: str) -> dict[str, float]:
    samples, sr = load_and_preprocess_audio(audio_path, duration_s=None, profile=preproc_profile)
    duration_s = float(len(samples) / sr) if sr else 0.0
    peak_abs = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = librosa.feature.rms(y=samples)[0]
    zcr = librosa.feature.zero_crossing_rate(y=samples)[0]
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=samples)[0]
    silence_ratio = float(np.mean(np.abs(samples) < 0.01)) if len(samples) else 1.0

    return {
        "duration_s": duration_s,
        "peak_abs": peak_abs,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "silence_ratio": silence_ratio,
    }


def _resolve_audio_path(processed_audio_path: str, raw_audio_path: str) -> str:
    processed = Path(processed_audio_path)
    if processed.exists():
        return str(processed)
    return raw_audio_path


def load_or_build_feature_cache(
    sample_manifest: pd.DataFrame,
    feature_cache: Path,
    force_recompute: bool,
    preproc_profile: str,
) -> pd.DataFrame:
    if feature_cache.exists() and not force_recompute:
        cached = pd.read_csv(feature_cache)
        if "preproc_profile" in cached.columns and cached["preproc_profile"].eq(preproc_profile).all():
            return cached

    rows: list[dict[str, object]] = []
    for row in tqdm(sample_manifest.itertuples(index=False), total=len(sample_manifest), desc="audit:cz_features"):
        feature_row = {
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "label": row.label,
            "task_type": row.task_type,
            "processed_audio_path": row.processed_audio_path,
            "raw_audio_path": row.raw_audio_path,
            "preproc_profile": preproc_profile,
        }
        feature_row["feature_audio_path"] = _resolve_audio_path(row.processed_audio_path, row.raw_audio_path)
        feature_row.update(_extract_audio_features(feature_row["feature_audio_path"], preproc_profile=preproc_profile))
        rows.append(feature_row)

    feature_frame = pd.DataFrame(rows)
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(feature_cache, index=False)
    return feature_frame


def summarize_features(feature_frame: pd.DataFrame, numeric_features: list[str]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    y = feature_frame["label"]
    for feature in numeric_features:
        grouped = feature_frame.groupby("label")[feature].agg(["mean", "std", "median", "min", "max"]) 
        summary[feature] = {
            "by_label": grouped.round(6).to_dict(orient="index"),
            "single_feature_auc": _safe_auc(y, feature_frame[feature]),
            "single_feature_auc_abs_distance_from_half": None,
        }
        auc = summary[feature]["single_feature_auc"]
        if auc is not None:
            summary[feature]["single_feature_auc_abs_distance_from_half"] = float(abs(auc - 0.5))
    return summary


def evaluate_proxy_model(
    feature_frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    n_splits: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y = feature_frame["label"].reset_index(drop=True)
    groups = feature_frame["subject_id"].reset_index(drop=True)
    X = feature_frame[numeric_features + categorical_features].reset_index(drop=True)

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    sample_predictions: list[pd.DataFrame] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        positive_index = list(model.named_steps["clf"].classes_).index("Patients")
        scores = model.predict_proba(X.iloc[test_idx])[:, positive_index]
        pred_labels = np.where(scores >= 0.5, "Patients", "Healthy")
        fold_frame = feature_frame.iloc[test_idx][["sample_id", "subject_id", "label"]].copy()
        fold_frame["fold"] = fold_idx
        fold_frame["score_patients"] = scores
        fold_frame["pred_label"] = pred_labels
        sample_predictions.append(fold_frame)

    sample_predictions_frame = pd.concat(sample_predictions, ignore_index=True)
    subject_predictions = (
        sample_predictions_frame.groupby(["subject_id", "label"], as_index=False, dropna=False)["score_patients"]
        .mean()
        .rename(columns={"score_patients": "score_patients_mean"})
    )
    subject_predictions["pred_label"] = subject_predictions["score_patients_mean"].apply(lambda score: "Patients" if score >= 0.5 else "Healthy")

    metrics = {
        "sample_level": binary_metrics(
            sample_predictions_frame["label"],
            sample_predictions_frame["pred_label"],
            sample_predictions_frame["score_patients"],
        ),
        "subject_level": binary_metrics(
            subject_predictions["label"],
            subject_predictions["pred_label"],
            subject_predictions["score_patients_mean"],
        ),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
    }
    return metrics, subject_predictions


def main() -> None:
    args = parse_args()
    sample_manifest = pd.read_csv(args.sample_manifest)
    feature_cache = Path(args.feature_cache)
    feature_frame = load_or_build_feature_cache(
        sample_manifest,
        feature_cache,
        args.force_recompute,
        args.preproc_profile,
    )

    numeric_features = [
        "duration_s",
        "peak_abs",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "spectral_flatness_mean",
        "silence_ratio",
    ]
    categorical_features = ["task_type"]

    model_specs = {
        "task_type_only": {"numeric": [], "categorical": ["task_type"]},
        "duration_only": {"numeric": ["duration_s"], "categorical": []},
        "recording_proxy_only": {"numeric": numeric_features, "categorical": []},
        "recording_proxy_plus_task": {"numeric": numeric_features, "categorical": categorical_features},
    }

    model_metrics: dict[str, object] = {}
    subject_prediction_frames: list[pd.DataFrame] = []
    for model_name, spec in model_specs.items():
        metrics, subject_predictions = evaluate_proxy_model(
            feature_frame,
            numeric_features=spec["numeric"],
            categorical_features=spec["categorical"],
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        model_metrics[model_name] = metrics
        subject_predictions = subject_predictions.copy()
        subject_predictions.insert(0, "model_name", model_name)
        subject_prediction_frames.append(subject_predictions)

    report = {
        "config": {
            "n_splits": args.n_splits,
            "random_state": args.random_state,
            "grouping": "subject_id",
            "cv_strategy": "StratifiedGroupKFold",
            "feature_cache": str(feature_cache),
            "preproc_profile": args.preproc_profile,
        },
        "dataset": {
            "n_rows": int(len(feature_frame)),
            "n_subjects": int(feature_frame["subject_id"].nunique()),
            "subjects_by_label": feature_frame[["subject_id", "label"]].drop_duplicates()["label"].value_counts().to_dict(),
            "samples_by_label": feature_frame["label"].value_counts().to_dict(),
        },
        "feature_summary": summarize_features(feature_frame, numeric_features),
        "proxy_model_metrics": model_metrics,
    }

    metrics_output = Path(args.metrics_output)
    predictions_output = Path(args.predictions_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.concat(subject_prediction_frames, ignore_index=True).to_csv(predictions_output, index=False)

    print(f"Saved Czech recording-confound audit metrics to: {metrics_output}")
    print(f"Saved Czech recording-confound subject predictions to: {predictions_output}")
    for model_name, metrics in model_metrics.items():
        subj = metrics["subject_level"]
        print(
            f"{model_name}: subject_bal_acc={subj['balanced_accuracy']:.4f} "
            f"subject_auc={subj['auc']:.4f}"
        )


if __name__ == "__main__":
    main()
