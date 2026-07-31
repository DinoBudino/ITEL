from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from packages.model_runtime import ModelRegistry

from .config import ApiSettings
from .demo_library import DemoLibrary, DemoSubject
from .sample_library import load_catalog
from .schemas import (
    CapabilitiesResponse,
    DemoAudioSummary,
    DemoSubjectDetail,
    DemoSubjectListResponse,
    DemoSubjectSummary,
    LiveResponse,
    ModelListResponse,
    ReadyResponse,
    SampleAudioListResponse,
    SubjectPredictionResponse,
)


LOGGER = logging.getLogger("itel.api")


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    settings = get_settings()
    return ModelRegistry(
        settings.registry_path,
        feature_cache_dir=settings.feature_cache_path,
        memory_cache_entries=settings.memory_cache_entries,
    )


@lru_cache(maxsize=1)
def get_demo_library() -> DemoLibrary:
    settings = get_settings()
    return DemoLibrary(settings.demo_catalog_path, settings.demo_czech_path, settings.demo_english_path)


async def read_upload_payloads(files: list[UploadFile], settings: ApiSettings) -> list[bytes]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one WAV file is required.")
    if len(files) > settings.max_files_per_subject:
        raise HTTPException(status_code=400, detail=f"A maximum of {settings.max_files_per_subject} files is allowed.")
    payloads: list[bytes] = []
    total_size = 0
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="All files must use the .wav extension.")
        payload = await file.read(settings.max_upload_size_bytes + 1)
        if len(payload) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail=f"Each file must be at most {settings.max_upload_size_mb} MB.")
        total_size += len(payload)
        if total_size > settings.max_total_upload_size_bytes:
            raise HTTPException(status_code=413, detail="Combined upload is too large.")
        payloads.append(payload)
    return payloads


def resolve_runtime(model_id: str):
    try:
        return get_registry().get_runtime(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def prediction_event_stream(runtime, payloads: list[bytes], language: str, demo_subject: DemoSubject | None = None):
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()

    def enqueue(event: dict | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_prediction() -> None:
        try:
            result = runtime.predict_subject(
                payloads,
                language,
                progress_callback=enqueue,
                cancel_event=cancel_event,
            )
            data = asdict(result)
            if demo_subject is not None:
                data.update({
                    "reference_label": demo_subject.reference_label,
                    "reference_match": data["predicted_label"] == demo_subject.reference_label,
                    "demo_subject_display_id": demo_subject.display_id,
                })
            enqueue({"type": "result", "progress": 100, "data": data})
        except InterruptedError as exc:
            enqueue({"type": "error", "detail": str(exc)})
        except ValueError as exc:
            enqueue({"type": "error", "detail": str(exc)})
        except Exception:
            LOGGER.exception("Streamed subject prediction failed")
            enqueue({"type": "error", "detail": "Prediction failed unexpectedly."})
        finally:
            enqueue(None)

    worker = asyncio.create_task(asyncio.to_thread(run_prediction))
    try:
        yield json.dumps({
            "type": "progress", "stage": "accepted", "progress": 5,
            "indeterminate": False, "message": "Archivos recibidos y validados.",
            "file_index": 0, "file_count": len(payloads), "elapsed_ms": 0,
            "cache_hits": 0, "cache_misses": 0,
        }) + "\n"
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=True) + "\n"
    finally:
        cancel_event.set()
        if not worker.done():
            worker.cancel()


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        registry = get_registry()
        if hasattr(registry, "warmup"):
            await asyncio.to_thread(registry.warmup)
        yield

    app = FastAPI(
        title="ITEL Subject-level Audio API",
        version="2.0.0",
        description="Research API for subject-level Czech and English speech analysis. Not a diagnostic device.",
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get("/live", response_model=LiveResponse)
    @app.get("/health", response_model=LiveResponse, include_in_schema=False)
    def live() -> LiveResponse:
        return LiveResponse()

    @app.get("/ready", response_model=ReadyResponse)
    def ready():
        errors = get_registry().readiness_errors()
        response = ReadyResponse(status="ready" if not errors else "not_ready", models=errors)
        return response if not errors else JSONResponse(status_code=503, content=response.model_dump())

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        return ModelListResponse(models=get_registry().catalog())

    @app.get("/v1/models/{model_id}")
    def model_info(model_id: str):
        models = {model["id"]: model for model in get_registry().catalog()}
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Unknown model.")
        return models[model_id]

    @app.get("/v1/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(demo_library_enabled=settings.enable_demo_library)

    @app.post("/v1/predictions/subject", response_model=SubjectPredictionResponse)
    async def predict_subject(
        model_id: str = Form(...),
        language: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> SubjectPredictionResponse:
        model_id = model_id.strip()
        language = language.strip().lower()
        payloads = await read_upload_payloads(files, settings)
        runtime = resolve_runtime(model_id)
        try:
            result = await asyncio.to_thread(runtime.predict_subject, payloads, language)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Subject prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc
        return SubjectPredictionResponse(**asdict(result))

    @app.post("/v1/predictions/subject/stream")
    async def predict_subject_stream(
        model_id: str = Form(...),
        language: str = Form(...),
        files: list[UploadFile] = File(...),
    ):
        model_id = model_id.strip()
        language = language.strip().lower()
        payloads = await read_upload_payloads(files, settings)
        runtime = resolve_runtime(model_id)
        return StreamingResponse(
            prediction_event_stream(runtime, payloads, language),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    if settings.enable_demo_library:
        library = get_demo_library()

        def ensure_model_language(model_id: str, language: str) -> None:
            registry = get_registry()
            manifest = registry.manifests.get(model_id)
            if manifest is None:
                raise HTTPException(status_code=404, detail="Unknown model.")
            if language not in manifest.languages:
                raise HTTPException(status_code=400, detail=f"Model {model_id} does not support language {language}.")

        def subject_summary(subject: DemoSubject) -> DemoSubjectSummary:
            return DemoSubjectSummary(
                token=subject.token,
                display_id=subject.display_id,
                language=subject.language,
                reference_label=subject.reference_label,
                audio_count=subject.audio_count,
                task_types=list(subject.task_types),
                total_duration_seconds=subject.total_duration_seconds,
            )

        @app.get("/v1/demo/subjects", response_model=DemoSubjectListResponse)
        def list_demo_subjects(
            model_id: str = Query(...),
            language: str = Query(..., pattern="^(cs|en)$"),
            reference_label: str | None = Query(default=None, pattern="^(Healthy|Patients)$"),
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=20, ge=1, le=100),
        ) -> DemoSubjectListResponse:
            ensure_model_language(model_id, language)
            subjects = library.list_subjects(language)
            if reference_label is not None:
                subjects = [subject for subject in subjects if subject.reference_label == reference_label]
            total = len(subjects)
            start = (page - 1) * page_size
            return DemoSubjectListResponse(
                items=[subject_summary(subject) for subject in subjects[start:start + page_size]],
                page=page,
                page_size=page_size,
                total=total,
                total_pages=math.ceil(total / page_size) if total else 0,
            )

        @app.get("/v1/demo/subjects/{subject_token}", response_model=DemoSubjectDetail)
        def get_demo_subject(subject_token: str) -> DemoSubjectDetail:
            try:
                subject = library.get_subject(subject_token)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return DemoSubjectDetail(
                **subject_summary(subject).model_dump(),
                audios=[
                    DemoAudioSummary(
                        token=audio.token,
                        display_name=audio.display_name,
                        task_type=audio.task_type,
                        duration_seconds=audio.duration_seconds,
                        stream_url=f"/v1/demo/audios/{audio.token}/stream",
                    )
                    for audio in subject.audios
                ],
            )

        @app.get("/v1/demo/audios/{audio_token}/stream")
        def stream_demo_audio(audio_token: str) -> FileResponse:
            try:
                audio = library.get_audio(audio_token)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return FileResponse(
                audio.path,
                media_type="audio/wav",
                filename=f"{audio.display_name.replace(' ', '-').lower()}.wav",
                headers={"Cache-Control": "private, no-store"},
            )

        @app.post("/v1/demo/subjects/{subject_token}/predict/stream")
        async def predict_demo_subject(subject_token: str, model_id: str = Form(...)):
            try:
                subject = library.get_subject(subject_token)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            model_id = model_id.strip()
            ensure_model_language(model_id, subject.language)
            runtime = resolve_runtime(model_id)
            payloads = await asyncio.gather(*(asyncio.to_thread(audio.path.read_bytes) for audio in subject.audios))
            return StreamingResponse(
                prediction_event_stream(runtime, list(payloads), subject.language, demo_subject=subject),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

    if settings.enable_sample_library:
        @app.get("/sample-audios", response_model=SampleAudioListResponse)
        def list_sample_audios(
            page: int = Query(default=1, ge=1),
            page_size: int = Query(default=25, ge=1, le=100),
            class_label: str | None = Query(default=None),
            query: str | None = Query(default=None),
        ) -> SampleAudioListResponse:
            items, _ = load_catalog(settings.sample_audio_path)
            filtered = items
            if class_label:
                filtered = [item for item in filtered if item.class_label.lower() == class_label.lower()]
            if query:
                term = query.strip().lower()
                filtered = [item for item in filtered if term in item.filename.lower() or term in item.subject_id.lower() or term in item.task_code.lower()]
            total = len(filtered)
            start = (page - 1) * page_size
            return SampleAudioListResponse(
                items=filtered[start:start + page_size],
                page=page,
                page_size=page_size,
                total=total,
                total_pages=math.ceil(total / page_size) if total else 0,
            )

        @app.get("/sample-audios/{sample_id}/stream")
        def stream_sample(sample_id: str) -> FileResponse:
            _, paths = load_catalog(settings.sample_audio_path)
            path = paths.get(sample_id)
            if path is None:
                raise HTTPException(status_code=404, detail="Sample not found.")
            return FileResponse(path, media_type="audio/wav", filename=path.name)

    return app


app = create_app()
