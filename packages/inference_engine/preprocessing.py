from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

import numpy as np

from .exceptions import ValidationError


@dataclass(slots=True)
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000, hp_cutoff: int = 100, target_duration: float = 2.0):
        self.target_sr = target_sr
        self.hp_cutoff = hp_cutoff
        self.target_duration = target_duration

    def process_bytes(self, audio_bytes: bytes) -> ProcessedAudio:
        if not audio_bytes:
            raise ValidationError("El archivo de audio esta vacio.")

        try:
            import soundfile as sf

            read_result = cast(tuple[Any, Any], sf.read(BytesIO(audio_bytes), always_2d=False))
            audio = read_result[0]
            sample_rate = int(read_result[1])
        except Exception as exc:
            raise ValidationError("No se pudo leer el archivo WAV.") from exc

        return self.process_array(audio, sample_rate)

    def process_array(self, audio: np.ndarray, sample_rate: int) -> ProcessedAudio:
        if audio is None or sample_rate is None:
            raise ValidationError("No se recibio audio valido.")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValidationError("El audio debe ser mono o convertible a mono.")
        if audio.size == 0:
            raise ValidationError("El audio no contiene muestras.")

        if sample_rate != self.target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sr)
            sample_rate = self.target_sr

        audio = self._highpass_filter(audio, sample_rate)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        audio = self._adjust_duration(audio, sample_rate)
        duration_seconds = len(audio) / float(sample_rate)
        if not np.isfinite(audio).all():
            raise ValidationError("El audio procesado contiene valores no validos.")

        return ProcessedAudio(samples=audio.astype(np.float32), sample_rate=sample_rate, duration_seconds=duration_seconds)

    def _highpass_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * sample_rate
        normalized_cutoff = self.hp_cutoff / nyquist
        b, a = butter(5, normalized_cutoff, btype="high", analog=False)
        return filtfilt(b, a, audio)

    def _adjust_duration(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        target_len = int(sample_rate * self.target_duration)
        if len(audio) > target_len:
            start = (len(audio) - target_len) // 2
            return audio[start : start + target_len]

        pad = (target_len - len(audio)) // 2
        return np.pad(audio, (pad, target_len - len(audio) - pad))
