from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

import numpy as np


def highpass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: int = 100, order: int = 5) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, a = butter(order, cutoff_hz / (0.5 * sample_rate), btype="high", analog=False)
    try:
        return filtfilt(b, a, audio)
    except ValueError as exc:
        raise ValueError("Audio is too short for high-pass filtering.") from exc


def normalize_rms(audio: np.ndarray, target_rms: float = 0.08, max_peak: float = 0.99) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    if rms <= 1e-8:
        return audio
    normalized = audio * (target_rms / rms)
    peak = float(np.max(np.abs(normalized)))
    return normalized * (max_peak / peak) if peak > max_peak > 0 else normalized


@dataclass(frozen=True, slots=True)
class AudioProfile:
    sample_rate: int = 16000
    highpass_hz: int = 100
    target_rms: float = 0.08
    max_peak: float = 0.99
    min_duration_seconds: float = 0.3

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not 0 < self.highpass_hz < self.sample_rate / 2:
            raise ValueError("highpass_hz must be below Nyquist.")
        if self.target_rms <= 0 or not 0 < self.max_peak <= 1:
            raise ValueError("Invalid normalization parameters.")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive.")


@dataclass(slots=True)
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    source_duration_seconds: float


class AudioPreprocessor:
    def __init__(self, profile: AudioProfile | None = None):
        self.profile = profile or AudioProfile()
        self.profile.validate()

    def process_bytes(self, audio_bytes: bytes) -> ProcessedAudio:
        if not audio_bytes:
            raise ValueError("Audio file is empty.")
        try:
            import soundfile as sf

            audio, sample_rate = cast(tuple[Any, Any], sf.read(BytesIO(audio_bytes), always_2d=False))
        except Exception as exc:
            raise ValueError("Audio file could not be decoded.") from exc
        return self.process_array(np.asarray(audio), int(sample_rate))

    def process_array(self, audio: np.ndarray, sample_rate: int) -> ProcessedAudio:
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 2:
            values = values.mean(axis=1)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Audio must be a non-empty mono or stereo signal.")
        if sample_rate <= 0 or not np.isfinite(values).all():
            raise ValueError("Audio has invalid samples or sample rate.")

        source_duration = values.size / float(sample_rate)
        if source_duration < self.profile.min_duration_seconds:
            raise ValueError(
                f"Audio must be at least {self.profile.min_duration_seconds:.1f} seconds long."
            )
        if sample_rate != self.profile.sample_rate:
            import librosa

            values = librosa.resample(values, orig_sr=sample_rate, target_sr=self.profile.sample_rate)
            sample_rate = self.profile.sample_rate

        values = highpass_filter(values, sample_rate, self.profile.highpass_hz)
        values = normalize_rms(values, self.profile.target_rms, self.profile.max_peak)
        if not np.isfinite(values).all():
            raise ValueError("Processed audio contains non-finite values.")
        return ProcessedAudio(values.astype(np.float32), sample_rate, source_duration)
