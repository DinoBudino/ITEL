from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt

from .config import CONFIG


def highpass_filter(samples: np.ndarray, sr: int, cutoff_hz: int | None = None, order: int = 5) -> np.ndarray:
    cutoff = cutoff_hz or CONFIG.audio.highpass_hz
    b, a = butter(order, cutoff / (0.5 * sr), btype="high", analog=False)
    return filtfilt(b, a, samples)


def normalize_peak(samples: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    return samples / peak


def normalize_rms(samples: np.ndarray, target_rms: float | None = None) -> np.ndarray:
    target = target_rms or CONFIG.audio.target_rms
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    if rms <= 1e-8:
        return samples
    normalized = samples * (target / rms)
    peak = float(np.max(np.abs(normalized))) if len(normalized) else 0.0
    max_peak = CONFIG.audio.max_peak_after_rms
    if peak > max_peak > 0:
        normalized = normalized * (max_peak / peak)
    return normalized


def trim_silence(samples: np.ndarray, top_db: float | None = None) -> np.ndarray:
    if len(samples) == 0:
        return samples
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db or CONFIG.audio.trim_top_db)
    return trimmed if len(trimmed) else samples


def keep_voiced_regions(samples: np.ndarray, top_db: float | None = None) -> np.ndarray:
    if len(samples) == 0:
        return samples
    intervals = librosa.effects.split(samples, top_db=top_db or CONFIG.audio.voiced_top_db)
    if len(intervals) == 0:
        return samples
    voiced = [samples[start:end] for start, end in intervals if end > start]
    if not voiced:
        return samples
    return np.concatenate(voiced)


def preprocess_samples(samples: np.ndarray, sr: int, profile: str = "standard") -> np.ndarray:
    if profile == "standard":
        profile = CONFIG.audio.default_profile
    samples = highpass_filter(samples, sr=sr)
    if profile == "multilingual_light":
        return normalize_rms(samples)
    if profile == "anti_confound":
        samples = trim_silence(samples)
        samples = keep_voiced_regions(samples)
        samples = normalize_rms(samples)
        return samples
    raise ValueError(f"Unsupported preprocessing profile: {profile}")


def fit_to_duration(samples: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    target_length = int(sr * duration_s)
    if len(samples) == target_length:
        return samples
    if len(samples) > target_length:
        start = max((len(samples) - target_length) // 2, 0)
        return samples[start : start + target_length]

    pad_total = target_length - len(samples)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(samples, (pad_left, pad_right), mode="constant")


def load_and_preprocess_audio(
    audio_path: str | Path,
    duration_s: float | None = None,
    profile: str = "standard",
) -> tuple[np.ndarray, int]:
    audio_path = Path(audio_path)
    samples, sr = librosa.load(audio_path, sr=None, mono=True)
    sr = int(sr)
    if sr != CONFIG.audio.target_sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=CONFIG.audio.target_sr)
        sr = CONFIG.audio.target_sr

    samples = preprocess_samples(samples, sr=sr, profile=profile)
    if duration_s is not None:
        samples = fit_to_duration(samples, sr=sr, duration_s=duration_s)
    return samples.astype(np.float32), sr
