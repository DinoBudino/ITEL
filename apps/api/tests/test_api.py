from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.app.main import create_app
from apps.api.app.config import ApiSettings
from packages.model_runtime.result import SubjectPrediction


class DummyRuntime:
    def predict_subject(self, audio_files, language, progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback({
                "type": "progress", "stage": "cache_lookup", "progress": 50,
                "indeterminate": False, "message": "Checking cache", "file_index": 1,
                "file_count": len(audio_files), "elapsed_ms": 1.0, "cache_hits": 1,
                "cache_misses": 0,
            })
        return SubjectPrediction(
            model_id="multilingual-cz-en",
            model_version="test",
            model_digest="abc123",
            language=language,
            file_count=len(audio_files),
            predicted_label="Healthy",
            confidence=0.8,
            probabilities={"Healthy": 0.8, "Patients": 0.2},
            threshold=0.5,
            timings_ms={"total": 2.0},
            warnings=[],
        )


def test_default_upload_limits_cover_natural_english_wavs():
    settings = ApiSettings()

    assert settings.max_upload_size_mb == 30
    assert settings.max_total_upload_size_mb == 50


class DummyRegistry:
    def __init__(self, errors=None):
        self.errors = errors or {}

    def readiness_errors(self):
        return self.errors

    def catalog(self):
        return [{
            "id": "multilingual-cz-en",
            "name": "Multilingual",
            "version": "test",
            "languages": ["cs", "en"],
            "prediction_unit": "subject",
            "available": True,
            "validation_errors": [],
        }]

    def get_runtime(self, model_id):
        if model_id != "multilingual-cz-en":
            raise KeyError(model_id)
        return DummyRuntime()


def _client(monkeypatch, registry=None):
    import apps.api.app.main as main_module

    monkeypatch.setattr(main_module, "get_registry", lambda: registry or DummyRegistry())
    return TestClient(create_app())


def test_live_and_model_catalog_do_not_load_model(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/live").json() == {"status": "ok"}
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["models"][0]["id"] == "multilingual-cz-en"


def test_ready_reports_missing_artifacts(monkeypatch):
    client = _client(monkeypatch, DummyRegistry({"czech-layermix": ["Missing artifact: model.pt"]}))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_subject_prediction_accepts_multiple_wavs(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/predictions/subject",
        data={"model_id": "multilingual-cz-en", "language": "en"},
        files=[
            ("files", ("one.wav", b"first", "audio/wav")),
            ("files", ("two.wav", b"second", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    assert response.json()["file_count"] == 2
    assert response.json()["model_id"] == "multilingual-cz-en"


def test_subject_prediction_rejects_non_wav(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/predictions/subject",
        data={"model_id": "multilingual-cz-en", "language": "en"},
        files={"files": ("audio.mp3", b"data", "audio/mpeg")},
    )

    assert response.status_code == 400


def test_streamed_prediction_emits_progress_and_result(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/v1/predictions/subject/stream",
        data={"model_id": "multilingual-cz-en", "language": "en"},
        files={"files": ("audio.wav", b"data", "audio/wav")},
    )

    assert response.status_code == 200
    events = [__import__("json").loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["progress", "progress", "result"]
    assert events[1]["stage"] == "cache_lookup"
    assert events[-1]["data"]["model_id"] == "multilingual-cz-en"
