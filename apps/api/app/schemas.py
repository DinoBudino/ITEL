from __future__ import annotations

from pydantic import BaseModel, Field


class LiveResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    models: dict[str, list[str]] = Field(default_factory=dict)


class ModelSummary(BaseModel):
    id: str
    name: str
    version: str
    languages: list[str]
    prediction_unit: str
    available: bool
    validation_errors: list[str]


class ModelListResponse(BaseModel):
    models: list[ModelSummary]


class CapabilitiesResponse(BaseModel):
    demo_library_enabled: bool


class DemoSubjectSummary(BaseModel):
    token: str
    display_id: str
    language: str
    reference_label: str
    audio_count: int
    task_types: list[str]
    total_duration_seconds: float


class DemoAudioSummary(BaseModel):
    token: str
    display_name: str
    task_type: str
    duration_seconds: float
    stream_url: str


class DemoSubjectDetail(DemoSubjectSummary):
    audios: list[DemoAudioSummary]


class DemoSubjectListResponse(BaseModel):
    items: list[DemoSubjectSummary]
    page: int
    page_size: int
    total: int
    total_pages: int


class SubjectPredictionResponse(BaseModel):
    model_id: str
    model_version: str
    model_digest: str
    language: str
    file_count: int = Field(..., ge=1)
    predicted_label: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    threshold: float = Field(..., ge=0, le=1)
    timings_ms: dict[str, float]
    warnings: list[str]
    feature_source: str = "computed"
    cache_hits: int = 0
    cache_misses: int = 0
    device: str = "unknown"


class SampleAudioItem(BaseModel):
    id: str
    class_label: str
    subject_id: str
    task_code: str
    filename: str
    relative_path: str


class SampleAudioListResponse(BaseModel):
    items: list[SampleAudioItem]
    page: int
    page_size: int
    total: int
    total_pages: int
