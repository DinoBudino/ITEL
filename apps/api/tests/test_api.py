from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from packages.inference_engine.inference import InferenceResult


class DummyService:
    class settings:
        model_version = "test-version"
        cors_origins = []

    def predict_from_bytes(self, audio_bytes: bytes, filename: str):
        return InferenceResult(
            predicted_label="Healthy",
            confidence=0.88,
            probabilities={"Healthy": 0.88, "Patients": 0.12},
            model_version="test-version",
            timings_ms={"preprocessing": 1.0, "embedding": 2.0, "classification": 3.0, "total": 6.0},
            metadata={"filename": filename, "sample_rate": 16000, "duration_seconds": 2.0, "embedding_dim": 4608},
        )


def test_predict_endpoint_accepts_wav(monkeypatch):
    import apps.api.app.main as main_module

    main_module.get_service.cache_clear()
    monkeypatch.setattr(main_module, "get_service", lambda: DummyService())
    monkeypatch.setattr(main_module, "get_settings", lambda: DummyService.settings)

    client = TestClient(create_app())
    response = client.post("/predict", files={"file": ("audio.wav", b"fake-bytes", "audio/wav")})

    assert response.status_code == 200
    assert response.json()["predicted_label"] == "Healthy"


def test_predict_endpoint_rejects_non_wav(monkeypatch):
    import apps.api.app.main as main_module

    monkeypatch.setattr(main_module, "get_service", lambda: DummyService())
    monkeypatch.setattr(main_module, "get_settings", lambda: DummyService.settings)

    client = TestClient(create_app())
    response = client.post("/predict", files={"file": ("audio.mp3", b"fake-bytes", "audio/mpeg")})

    assert response.status_code == 400
