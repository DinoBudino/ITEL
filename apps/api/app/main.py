from __future__ import annotations

import base64
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from packages.inference_engine.config import Settings
from packages.inference_engine.exceptions import ITELServingError, ValidationError
from packages.inference_engine.inference import ITELInferenceService
from packages.inference_engine.schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictResponse, SampleAudioItem, SampleAudioListResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("itel.api")


MAIN_SAMPLE_PATTERN = re.compile(r"_[aeiou]{1,2}\d+$", re.IGNORECASE)
SAMPLE_AUDIO_MANIFEST = ".sample_audio_catalog.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_service() -> ITELInferenceService:
    service = ITELInferenceService(get_settings())
    service.warmup()
    return service


def make_sample_id(relative_path: Path) -> str:
    return base64.urlsafe_b64encode(relative_path.as_posix().encode("utf-8")).decode("ascii").rstrip("=")


def decode_sample_id(sample_id: str) -> Path:
    padding = '=' * (-len(sample_id) % 4)
    return Path(base64.urlsafe_b64decode(f"{sample_id}{padding}").decode("utf-8"))


@lru_cache(maxsize=1)
def get_sample_audio_catalog() -> tuple[list[SampleAudioItem], dict[str, Path]]:
    settings = get_settings()
    base_dir = settings.sample_audio_dir

    if not base_dir.exists():
        LOGGER.warning("Sample audio directory does not exist: %s", base_dir)
        return [], {}

    manifest_path = base_dir / SAMPLE_AUDIO_MANIFEST
    manifest_catalog = load_sample_audio_catalog_from_manifest(base_dir, manifest_path)
    if manifest_catalog is not None:
        return manifest_catalog

    catalog = build_sample_audio_catalog(base_dir)
    write_sample_audio_manifest(manifest_path, catalog[0])
    return catalog


def load_sample_audio_catalog_from_manifest(base_dir: Path, manifest_path: Path) -> tuple[list[SampleAudioItem], dict[str, Path]] | None:
    if not manifest_path.exists():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = [SampleAudioItem.model_validate(item) for item in payload]
    except Exception as exc:
        LOGGER.warning("Failed to load sample audio manifest %s: %s", manifest_path, exc)
        return None

    path_map = {item.id: base_dir / item.relative_path for item in items}
    LOGGER.info("Loaded %s sample audios from manifest %s", len(items), manifest_path)
    return items, path_map


def write_sample_audio_manifest(manifest_path: Path, items: list[SampleAudioItem]) -> None:
    payload = [item.model_dump() for item in items]

    try:
        manifest_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        LOGGER.info("Wrote sample audio manifest %s with %s items", manifest_path, len(items))
    except OSError as exc:
        LOGGER.warning("Failed to write sample audio manifest %s: %s", manifest_path, exc)


def build_sample_audio_catalog(base_dir: Path) -> tuple[list[SampleAudioItem], dict[str, Path]]:
    items: list[SampleAudioItem] = []
    path_map: dict[str, Path] = {}

    for audio_path in sorted(base_dir.rglob("*.wav")):
        if not audio_path.is_file():
            continue

        if MAIN_SAMPLE_PATTERN.search(audio_path.stem):
            continue

        try:
            relative_path = audio_path.relative_to(base_dir)
        except ValueError:
            continue

        parts = relative_path.parts
        if len(parts) < 4:
            continue

        class_label, subject_id, task_code = parts[0], parts[1], parts[2]
        sample_id = make_sample_id(relative_path)
        item = SampleAudioItem(
            id=sample_id,
            class_label=class_label,
            subject_id=subject_id,
            task_code=task_code,
            filename=audio_path.name,
            relative_path=relative_path.as_posix(),
        )
        items.append(item)
        path_map[sample_id] = audio_path

    LOGGER.info("Indexed %s sample audios from %s", len(items), base_dir)
    return items, path_map


def get_sample_audio_path(sample_id: str) -> Path:
    settings = get_settings()
    _, path_map = get_sample_audio_catalog()
    sample_path = path_map.get(sample_id)
    if not sample_path:
        raise HTTPException(status_code=404, detail="No se encontro la muestra solicitada.")

    try:
        sample_path.relative_to(settings.sample_audio_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="La muestra solicitada no es valida.") from exc

    return sample_path


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ITEL Audio TEL API",
        version="1.0.0",
        description="API de inferencia para clasificacion de voz Healthy vs Patients.",
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service = get_service()
        return HealthResponse(model_version=service.settings.model_version)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_version=settings.model_version,
            model_path=str(settings.final_model_path),
            scaler_path=str(settings.final_scaler_path),
            hubert_bundle=settings.hubert_bundle,
            target_sample_rate=settings.target_sr,
            target_duration_seconds=settings.target_duration,
        )

    @app.get("/sample-audios", response_model=SampleAudioListResponse)
    def list_sample_audios(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        class_label: str | None = Query(default=None),
        query: str | None = Query(default=None),
    ) -> SampleAudioListResponse:
        catalog, _ = get_sample_audio_catalog()
        filtered = catalog

        if class_label:
            filtered = [item for item in filtered if item.class_label.lower() == class_label.lower()]

        if query:
            normalized_query = query.strip().lower()
            filtered = [
                item
                for item in filtered
                if normalized_query in item.filename.lower()
                or normalized_query in item.subject_id.lower()
                or normalized_query in item.task_code.lower()
            ]

        total = len(filtered)
        total_pages = math.ceil(total / page_size) if total else 0
        start = (page - 1) * page_size
        end = start + page_size
        items = filtered[start:end] if start < total else []

        return SampleAudioListResponse(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    @app.get("/sample-audios/{sample_id}/stream")
    def stream_sample_audio(sample_id: str) -> FileResponse:
        sample_path = get_sample_audio_path(sample_id)
        return FileResponse(sample_path, media_type="audio/wav", filename=sample_path.name)

    @app.post("/predict-sample/{sample_id}", response_model=PredictResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    def predict_sample(sample_id: str) -> PredictResponse:
        sample_path = get_sample_audio_path(sample_id)

        try:
            payload = sample_path.read_bytes()
            result = get_service().predict_from_bytes(payload, filename=sample_path.name)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ITELServingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail="Ocurrio un error inesperado durante la inferencia.") from exc

        return PredictResponse(
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            model_version=result.model_version,
            timings_ms=result.timings_ms,
            metadata=result.metadata,
        )

    @app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Debes enviar un archivo WAV.")
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="Solo se aceptan archivos .wav.")

        payload = await file.read()
        try:
            result = get_service().predict_from_bytes(payload, filename=file.filename)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ITELServingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail="Ocurrio un error inesperado durante la inferencia.") from exc

        return PredictResponse(
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            model_version=result.model_version,
            timings_ms=result.timings_ms,
            metadata=result.metadata,
        )

    return app


app = create_app()
