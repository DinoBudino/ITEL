from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from packages.inference_engine.config import Settings
from packages.inference_engine.exceptions import ITELServingError, ValidationError
from packages.inference_engine.inference import ITELInferenceService
from packages.inference_engine.schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_service() -> ITELInferenceService:
    service = ITELInferenceService(get_settings())
    service.warmup()
    return service


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
