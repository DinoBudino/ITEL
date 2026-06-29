from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.scripts.audit_cz_recording_confounds import (
    _extract_audio_features,
    _resolve_audio_path,
    _safe_auc,
    evaluate_proxy_model,
    load_or_build_feature_cache,
)
from research.src.config import CONFIG
from research.src.evaluate import binary_metrics


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"

NUMERIC_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Czech-derived acoustic markers against English external labels.")
    parser.add_argument("--cz-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name))
    parser.add_argument("--en-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name))
    parser.add_argument(
        "--cz-feature-cache",
        default=str(CONFIG.paths.processed_manifest_dir / "cz_main_recording_proxy_features.csv"),
    )
    parser.add_argument(
        "--en-feature-cache",
        default=str(CONFIG.paths.processed_manifest_dir / "english_external_recording_proxy_features.csv"),
    )
    parser.add_argument(
        "--cz-audit-report",
        default=str(REPORT_CSV_DIR / "audit_cz_recording_confounds.json"),
    )
    parser.add_argument("--preproc-profile", default="anti_confound", choices=["standard", "anti_confound"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metrics-output",
        default=str(REPORT_CSV_DIR / "compare_cz_english_markers.json"),
    )
    parser.add_argument(
        "--predictions-output",
        default=str(REPORT_CSV_DIR / "compare_cz_english_markers_predictions.csv"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def _build_english_feature_cache(
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
    for row in sample_manifest.itertuples(index=False):
        feature_audio_path = _resolve_audio_path(
            getattr(row, "processed_audio_path", "") or "",
            getattr(row, "raw_audio_path", "") or "",
        )
        feature_row: dict[str, object] = {
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "label": row.label,
            "processed_audio_path": getattr(row, "processed_audio_path", "") or "",
            "raw_audio_path": getattr(row, "raw_audio_path", "") or "",
            "preproc_profile": preproc_profile,
            "feature_audio_path": feature_audio_path,
        }
        feature_row.update(_extract_audio_features(feature_audio_path, preproc_profile=preproc_profile))
        rows.append(feature_row)

    feature_frame = pd.DataFrame(rows)
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(feature_cache, index=False)
    return feature_frame


def _top_cz_features(cz_audit_report: Path, top_k: int) -> list[str]:
    if cz_audit_report.exists():
        report = json.loads(cz_audit_report.read_text(encoding="utf-8"))
        summary = report.get("feature_summary", {})
        ranked = sorted(
            (
                (
                    feature,
                    details.get("single_feature_auc_abs_distance_from_half")
                    if details.get("single_feature_auc_abs_distance_from_half") is not None
                    else 0.0,
                )
                for feature, details in summary.items()
                if feature in NUMERIC_FEATURES
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [feature for feature, _ in ranked[:top_k]]
    fallback = ["spectral_rolloff_mean", "spectral_bandwidth_mean", "rms_std", "duration_s", "spectral_centroid_mean"]
    return fallback[:top_k]


def _single_feature_summary(cz_features: pd.DataFrame, en_features: pd.DataFrame, selected_features: list[str]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for feature in selected_features:
        cz_auc = _safe_auc(cz_features["label"], cz_features[feature])
        en_auc = _safe_auc(en_features["label"], en_features[feature])
        summary[feature] = {
            "cz_auc": cz_auc,
            "english_auc": en_auc,
            "cz_auc_abs_distance_from_half": None if cz_auc is None else float(abs(cz_auc - 0.5)),
            "english_auc_abs_distance_from_half": None if en_auc is None else float(abs(en_auc - 0.5)),
            "direction_matches": None if cz_auc is None or en_auc is None else bool((cz_auc >= 0.5) == (en_auc >= 0.5)),
        }
    return summary


def _fit_marker_transfer_model(cz_features: pd.DataFrame, selected_features: list[str]) -> Pipeline:
    X = cz_features[selected_features]
    y = cz_features["label"]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def _evaluate_marker_transfer(model: Pipeline, en_features: pd.DataFrame, selected_features: list[str]) -> tuple[dict[str, object], pd.DataFrame]:
    X = en_features[selected_features]
    positive_index = list(model.named_steps["clf"].classes_).index("Patients")
    scores = model.predict_proba(X)[:, positive_index]
    pred_labels = pd.Series(scores).apply(lambda score: "Patients" if score >= 0.5 else "Healthy")
    predictions = en_features[["sample_id", "subject_id", "label"]].copy()
    predictions["score_patients"] = scores
    predictions["pred_label"] = pred_labels
    metrics = binary_metrics(predictions["label"], predictions["pred_label"], predictions["score_patients"])
    return metrics, predictions


def main() -> None:
    args = parse_args()
    cz_sample_manifest = pd.read_csv(args.cz_sample_manifest)
    en_sample_manifest = pd.read_csv(args.en_sample_manifest)

    cz_features = load_or_build_feature_cache(
        cz_sample_manifest,
        Path(args.cz_feature_cache),
        args.force_recompute,
        args.preproc_profile,
    )
    en_features = _build_english_feature_cache(
        en_sample_manifest,
        Path(args.en_feature_cache),
        args.force_recompute,
        args.preproc_profile,
    )

    selected_features = _top_cz_features(Path(args.cz_audit_report), args.top_k)
    feature_transfer = _single_feature_summary(cz_features, en_features, selected_features)

    english_cv_metrics, english_cv_predictions = evaluate_proxy_model(
        en_features,
        numeric_features=selected_features,
        categorical_features=[],
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    transfer_model = _fit_marker_transfer_model(cz_features, selected_features)
    english_transfer_metrics, english_transfer_predictions = _evaluate_marker_transfer(transfer_model, en_features, selected_features)

    report = {
        "config": {
            "preproc_profile": args.preproc_profile,
            "top_k": args.top_k,
            "selected_features": selected_features,
            "n_splits": args.n_splits,
            "random_state": args.random_state,
        },
        "datasets": {
            "cz_rows": int(len(cz_features)),
            "cz_subjects": int(cz_features["subject_id"].nunique()),
            "english_rows": int(len(en_features)),
            "english_subjects": int(en_features["subject_id"].nunique()),
            "english_labels": en_features["label"].value_counts().to_dict(),
        },
        "single_feature_transfer": feature_transfer,
        "english_marker_only_cv": english_cv_metrics,
        "cz_trained_marker_transfer_to_english": english_transfer_metrics,
    }

    metrics_output = Path(args.metrics_output)
    predictions_output = Path(args.predictions_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    english_cv_predictions = english_cv_predictions.copy()
    english_cv_predictions.insert(0, "evaluation", "english_marker_only_cv")
    english_transfer_predictions = english_transfer_predictions.copy()
    english_transfer_predictions.insert(0, "evaluation", "cz_trained_marker_transfer_to_english")
    pd.concat([english_cv_predictions, english_transfer_predictions], ignore_index=True).to_csv(predictions_output, index=False)

    print(f"Saved marker comparison report to: {metrics_output}")
    print(f"Saved marker predictions to: {predictions_output}")
    print(f"Selected features: {selected_features}")
    print(
        "Marker-only English CV: "
        f"bal_acc={english_cv_metrics['subject_level']['balanced_accuracy']:.4f} "
        f"auc={english_cv_metrics['subject_level']['auc']:.4f}"
    )
    print(
        "Czech-trained marker transfer: "
        f"bal_acc={english_transfer_metrics['balanced_accuracy']:.4f} "
        f"auc={english_transfer_metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
