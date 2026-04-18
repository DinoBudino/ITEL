from __future__ import annotations

import numpy as np

from packages.inference_engine.config import Settings
from packages.inference_engine.inference import ITELInferenceService
from packages.inference_engine.preprocessing import ProcessedAudio


class DummyScaler:
    def transform(self, array):
        return array


class DummyClassifier:
    def predict_logits(self, embedding_scaled):
        return np.array([[0.1, 0.9]], dtype=np.float32)


class DummyExtractor:
    def extract_embedding(self, processed_audio):
        return np.ones(4608, dtype=np.float32)


def test_predict_from_bytes_returns_patients(monkeypatch):
    service = ITELInferenceService(Settings())
    service.scaler = DummyScaler()
    service.classifier = DummyClassifier()
    service.embedding_extractor = DummyExtractor()
    service.hparams = {"hidden": 256}

    monkeypatch.setattr(
        service.preprocessor,
        "process_bytes",
        lambda audio_bytes: ProcessedAudio(samples=np.zeros(32000, dtype=np.float32), sample_rate=16000, duration_seconds=2.0),
    )

    result = service.predict_from_bytes(b"wav-bytes", filename="demo.wav")

    assert result.predicted_label == "Patients"
    assert result.metadata["embedding_dim"] == 4608
    assert set(result.probabilities) == {"Healthy", "Patients"}
