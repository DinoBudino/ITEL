from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.src.multilingual_evaluation import (
    best_threshold,
    fit_model,
    healthy_centroid,
    positive_class_index,
    run_target_loso,
    shift_source_toward_target,
)


def fit_and_export_target_head(
    source_subjects: pd.DataFrame,
    target_subjects: pd.DataFrame,
    output_path: Path,
) -> dict[str, float]:
    _, oof_predictions = run_target_loso(source_subjects, target_subjects, "logreg_pca", "centroid_shift")
    y_true = (oof_predictions["label"] == "Patients").astype(int).to_numpy()
    threshold, threshold_metrics = best_threshold(y_true, oof_predictions["score_patients"].to_numpy())

    shifted_source = shift_source_toward_target(source_subjects, target_subjects)
    training = pd.concat([shifted_source, target_subjects], ignore_index=True)
    model = fit_model("logreg_pca", np.vstack(training["embedding"].to_list()), training["label"])
    if positive_class_index(model) != 1:
        raise ValueError("Export expects Patients to be classifier class index 1.")

    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]
    classifier = model.named_steps["clf"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_healthy_centroid=healthy_centroid(source_subjects),
        target_healthy_centroid=healthy_centroid(target_subjects),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        classifier_coef=classifier.coef_,
        classifier_intercept=classifier.intercept_,
        threshold=np.asarray([threshold], dtype=np.float64),
    )
    return {"threshold": threshold, "oof_balanced_accuracy": float(threshold_metrics["balanced_accuracy"])}


def write_multilingual_manifest(output_dir: Path, metrics: dict[str, dict[str, float]]) -> Path:
    artifacts = []
    for file_name in ("target-cs.npz", "target-en.npz"):
        artifacts.append({"file": file_name, "sha256": sha256_file(output_dir / file_name)})
    payload = {
        "schema_version": 1,
        "model_id": "multilingual-cz-en",
        "display_name": "Unified Multilingual CZ/EN",
        "version": "1.0.0",
        "family": "multilingual_logreg_pca",
        "status": "promoted",
        "prediction_unit": "subject",
        "languages": ["cs", "en"],
        "classes": ["Healthy", "Patients"],
        "positive_class": "Patients",
        "encoder": {"bundle": "HUBERT_BASE", "layer_policy": "last", "hidden_size": 768},
        "feature_recipe": {"id": "hubert-base-last-temporal-stats-v1", "dimension": 4608, "aggregation": "subject_mean"},
        "preprocessing": {"sample_rate": 16000, "highpass_hz": 100, "target_rms": 0.08, "max_peak": 0.99, "min_duration_seconds": 0.3},
        "language_heads": {"cs": "target-cs.npz", "en": "target-en.npz"},
        "artifacts": artifacts,
        "validation": metrics,
        "limitations": [
            "Supervised adaptation for LANNA Czech and ENNI English only",
            "Not validated as zero-shot transfer",
            "Threshold selected from out-of-fold predictions; no independent deployment cohort",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
