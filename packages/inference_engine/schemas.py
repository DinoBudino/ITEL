from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_path: str
    scaler_path: str
    hubert_bundle: str
    target_sample_rate: int
    target_duration_seconds: float


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    timings_ms: dict[str, float]
    metadata: dict[str, object]
