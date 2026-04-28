from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CONFIG


@dataclass(frozen=True)
class Segment:
    segment_id: str
    parent_audio_id: str
    start_s: float
    end_s: float
    duration_s: float
    samples: np.ndarray


@dataclass(frozen=True)
class SegmentQC:
    status: str
    reason: str | None
    rms_mean: float
    silence_ratio: float


def create_fixed_segments(
    audio_id: str,
    samples: np.ndarray,
    sr: int,
    segment_duration_s: float,
    allow_overlap: bool = False,
) -> list[Segment]:
    if allow_overlap:
        raise ValueError("The current blueprint only supports non-overlapping segments.")

    segment_size = int(sr * segment_duration_s)
    if segment_size <= 0:
        raise ValueError("segment_duration_s must be positive")

    segments: list[Segment] = []
    for index, start in enumerate(range(0, len(samples), segment_size)):
        end = min(start + segment_size, len(samples))
        chunk = samples[start:end]
        if len(chunk) == 0:
            continue
        segment_id = f"{audio_id}_seg_{index:04d}"
        segments.append(
            Segment(
                segment_id=segment_id,
                parent_audio_id=audio_id,
                start_s=start / sr,
                end_s=end / sr,
                duration_s=(end - start) / sr,
                samples=chunk,
            )
        )
    return segments


def aggregate_probabilities(probabilities: list[float], method: str = "mean") -> float:
    if not probabilities:
        raise ValueError("At least one probability is required for aggregation.")
    values = np.array(probabilities, dtype=np.float32)
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    raise ValueError(f"Unsupported aggregation method: {method}")


def compute_segment_qc(segments: list[Segment], target_duration_s: float) -> dict[str, SegmentQC]:
    if not segments:
        return {}

    rms_by_segment: dict[str, float] = {}
    silence_by_segment: dict[str, float] = {}
    eligible_rms: list[float] = []

    for segment in segments:
        samples = np.asarray(segment.samples, dtype=np.float32)
        rms_mean = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        silence_ratio = float(np.mean(np.abs(samples) < 0.01)) if len(samples) else 1.0
        rms_by_segment[segment.segment_id] = rms_mean
        silence_by_segment[segment.segment_id] = silence_ratio
        if segment.duration_s >= target_duration_s * CONFIG.audio.min_segment_duration_ratio:
            eligible_rms.append(rms_mean)

    median_rms = float(np.median(eligible_rms)) if eligible_rms else 0.0
    min_relative_rms = median_rms * CONFIG.audio.relative_segment_rms_factor
    min_required_rms = max(CONFIG.audio.min_segment_rms, min_relative_rms)

    qc_by_segment: dict[str, SegmentQC] = {}
    for segment in segments:
        reasons: list[str] = []
        rms_mean = rms_by_segment[segment.segment_id]
        silence_ratio = silence_by_segment[segment.segment_id]
        if segment.duration_s < target_duration_s * CONFIG.audio.min_segment_duration_ratio:
            reasons.append("short_segment")
        if silence_ratio > CONFIG.audio.max_segment_silence_ratio:
            reasons.append("high_silence")
        if rms_mean < min_required_rms:
            reasons.append("low_rms")
        qc_by_segment[segment.segment_id] = SegmentQC(
            status="passed" if not reasons else "excluded",
            reason="; ".join(reasons) if reasons else None,
            rms_mean=rms_mean,
            silence_ratio=silence_ratio,
        )
    return qc_by_segment
