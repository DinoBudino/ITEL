from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from packages.model_runtime.result import SubjectPrediction


class DemoRuntime:
    def predict_subject(self, audio_files, language, progress_callback=None, cancel_event=None):
        if progress_callback:
            progress_callback({
                "type": "progress", "stage": "cache_hit", "progress": 80,
                "indeterminate": False, "message": "Cached", "file_index": len(audio_files),
                "file_count": len(audio_files), "elapsed_ms": 1, "cache_hits": len(audio_files),
                "cache_misses": 0,
            })
        return SubjectPrediction(
            model_id="multilingual-cz-en", model_version="test", model_digest="digest",
            language=language, file_count=len(audio_files), predicted_label="Healthy",
            confidence=0.9, probabilities={"Healthy": 0.9, "Patients": 0.1}, threshold=0.5,
            timings_ms={"total": 1.0}, warnings=[], feature_source="precomputed_cache",
            cache_hits=len(audio_files), cache_misses=0, device="cpu",
        )


class DemoRegistry:
    def __init__(self):
        self.manifests = {
            "multilingual-cz-en": SimpleNamespace(languages=["cs", "en"]),
            "czech-layermix": SimpleNamespace(languages=["cs"]),
        }

    def get_runtime(self, model_id):
        if model_id not in self.manifests:
            raise KeyError(model_id)
        return DemoRuntime()

    def catalog(self):
        return []

    def readiness_errors(self):
        return {}


def _demo_client(monkeypatch, tmp_path: Path) -> TestClient:
    import apps.api.app.main as main_module

    czech_root = tmp_path / "czech"
    english_root = tmp_path / "english"
    audio_path = czech_root / "private-subject" / "recording.wav"
    audio_path.parent.mkdir(parents=True)
    english_root.mkdir()
    audio_path.write_bytes(b"private-wave-data")
    catalog = {
        "schema_version": 1,
        "subject_count": 1,
        "audio_count": 1,
        "subjects": [{
            "token": "opaque-subject-token",
            "display_id": "CS-DEMO-001",
            "language": "cs",
            "reference_label": "Healthy",
            "audio_count": 1,
            "task_types": ["controlled_word"],
            "total_duration_seconds": 1.0,
            "audios": [{
                "token": "opaque-audio-token",
                "display_name": "Grabacion 01",
                "task_type": "controlled_word",
                "duration_seconds": 1.0,
                "relative_path": "czech/private-subject/recording.wav",
            }],
        }],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setenv("ITEL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("ITEL_ENABLE_DEMO_LIBRARY", "true")
    monkeypatch.setenv("ITEL_DEMO_CATALOG_FILE", "catalog.json")
    monkeypatch.setenv("ITEL_DEMO_CZECH_DIR", "czech")
    monkeypatch.setenv("ITEL_DEMO_ENGLISH_DIR", "english")
    main_module.get_settings.cache_clear()
    main_module.get_demo_library.cache_clear()
    monkeypatch.setattr(main_module, "get_registry", lambda: DemoRegistry())
    return TestClient(main_module.create_app())


def test_demo_listing_and_detail_expose_reference_without_private_paths(monkeypatch, tmp_path):
    client = _demo_client(monkeypatch, tmp_path)

    assert client.get("/v1/capabilities").json()["demo_library_enabled"] is True
    listing = client.get("/v1/demo/subjects", params={"model_id": "multilingual-cz-en", "language": "cs"})
    detail = client.get("/v1/demo/subjects/opaque-subject-token")

    assert listing.status_code == 200 and listing.json()["total"] == 1
    serialized = json.dumps({"listing": listing.json(), "detail": detail.json()})
    assert listing.json()["items"][0]["reference_label"] == "Healthy"
    assert detail.json()["reference_label"] == "Healthy"
    assert "private-subject" not in serialized
    assert "recording.wav" not in serialized
    assert detail.json()["audios"][0]["display_name"] == "Grabacion 01"


def test_demo_listing_filters_by_reference_label(monkeypatch, tmp_path):
    client = _demo_client(monkeypatch, tmp_path)

    healthy = client.get(
        "/v1/demo/subjects",
        params={"model_id": "multilingual-cz-en", "language": "cs", "reference_label": "Healthy"},
    )
    patients = client.get(
        "/v1/demo/subjects",
        params={"model_id": "multilingual-cz-en", "language": "cs", "reference_label": "Patients"},
    )

    assert healthy.status_code == 200 and healthy.json()["total"] == 1
    assert patients.status_code == 200 and patients.json()["total"] == 0


def test_demo_prediction_reveals_reference_only_in_final_event(monkeypatch, tmp_path):
    client = _demo_client(monkeypatch, tmp_path)
    response = client.post(
        "/v1/demo/subjects/opaque-subject-token/predict/stream",
        data={"model_id": "multilingual-cz-en"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert all("reference_label" not in event for event in events[:-1])
    assert events[-1]["data"]["reference_label"] == "Healthy"
    assert events[-1]["data"]["reference_match"] is True
    assert events[-1]["data"]["demo_subject_display_id"] == "CS-DEMO-001"


def test_demo_rejects_language_incompatible_model(monkeypatch, tmp_path):
    client = _demo_client(monkeypatch, tmp_path)

    response = client.get(
        "/v1/demo/subjects",
        params={"model_id": "czech-layermix", "language": "en"},
    )

    assert response.status_code == 400


def test_demo_audio_stream_uses_anonymous_filename(monkeypatch, tmp_path):
    client = _demo_client(monkeypatch, tmp_path)

    response = client.get("/v1/demo/audios/opaque-audio-token/stream")

    assert response.status_code == 200
    assert response.content == b"private-wave-data"
    assert "grabacion-01.wav" in response.headers["content-disposition"]
    assert "private-subject" not in response.headers["content-disposition"]
