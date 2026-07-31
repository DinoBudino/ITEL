from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SubjectPrediction:
    model_id: str
    model_version: str
    model_digest: str
    language: str
    file_count: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    threshold: float
    timings_ms: dict[str, float]
    warnings: list[str]
    feature_source: str = "computed"
    cache_hits: int = 0
    cache_misses: int = 0
    device: str = "unknown"
