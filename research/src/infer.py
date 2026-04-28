from __future__ import annotations

from pathlib import Path

import numpy as np

from .encoders import XLSREncoder
from .preprocessing import load_and_preprocess_audio
from .segmentation import aggregate_probabilities, create_fixed_segments


def predict_single_window(audio_path: str | Path, classifier, duration_s: float = 2.0, encoder: XLSREncoder | None = None) -> float:
    encoder = encoder or XLSREncoder()
    samples, sr = load_and_preprocess_audio(audio_path, duration_s=duration_s)
    embedding = encoder.encode(samples, sr).embedding.reshape(1, -1)
    return float(classifier.predict_proba(embedding)[0, 1])


def predict_full_audio(audio_id: str, audio_path: str | Path, classifier, chunk_duration_s: float = 5.0, encoder: XLSREncoder | None = None) -> float:
    encoder = encoder or XLSREncoder()
    samples, sr = load_and_preprocess_audio(audio_path, duration_s=None)
    segments = create_fixed_segments(audio_id, samples, sr, chunk_duration_s, allow_overlap=False)
    probabilities: list[float] = []
    for segment in segments:
        embedding = encoder.encode(segment.samples, sr).embedding.reshape(1, -1)
        probabilities.append(float(classifier.predict_proba(embedding)[0, 1]))
    if not probabilities:
        raise ValueError(f"No segments were produced for audio: {audio_path}")
    return aggregate_probabilities(probabilities, method="mean")
