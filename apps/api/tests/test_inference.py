from __future__ import annotations

import json

import numpy as np

from packages.model_runtime.manifest import load_manifest
from packages.model_runtime.multilingual import MultilingualRuntime


class DummyPreprocessor:
    def process_bytes(self, payload):
        return payload


class DummyExtractor:
    device = "cpu"

    def extract_last_layer(self, processed):
        return np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)


def test_multilingual_runtime_uses_frozen_numpy_head(tmp_path):
    head_path = tmp_path / "target-en.npz"
    np.savez(
        head_path,
        source_healthy_centroid=np.zeros(12),
        target_healthy_centroid=np.zeros(12),
        scaler_mean=np.zeros(12),
        scaler_scale=np.ones(12),
        pca_mean=np.zeros(12),
        pca_components=np.eye(12)[:1],
        classifier_coef=np.ones((1, 1)),
        classifier_intercept=np.zeros(1),
        threshold=np.array([0.5]),
    )
    manifest_payload = {
        "schema_version": 1,
        "model_id": "multilingual-cz-en",
        "version": "test",
        "family": "multilingual_logreg_pca",
        "prediction_unit": "subject",
        "languages": ["en"],
        "classes": ["Healthy", "Patients"],
        "encoder": {"bundle": "HUBERT_BASE"},
        "feature_recipe": {"id": "test-stats", "dimension": 12},
        "preprocessing": {"sample_rate": 16000, "highpass_hz": 100, "target_rms": 0.08, "max_peak": 0.99, "min_duration_seconds": 0.3},
        "language_heads": {"en": "target-en.npz"},
        "artifacts": [{"file": "target-en.npz", "sha256": None}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    runtime = MultilingualRuntime(load_manifest(tmp_path / "manifest.json"))
    runtime.preprocessor = DummyPreprocessor()
    runtime.extractor = DummyExtractor()

    result = runtime.predict_subject([b"one", b"two"], "en")

    assert result.predicted_label == "Patients"
    assert result.file_count == 2
    assert result.probabilities["Patients"] > 0.5
