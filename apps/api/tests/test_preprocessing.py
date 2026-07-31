from __future__ import annotations

from io import BytesIO
import wave

import numpy as np
import pytest

from packages.audio_pipeline.preprocessing import AudioPreprocessor


def make_wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 22050) -> bytes:
    timeline = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    waveform = (0.3 * np.sin(2 * np.pi * 220 * timeline) * 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())
    return buffer.getvalue()


def test_process_bytes_uses_natural_duration_and_rms_profile():
    processed = AudioPreprocessor().process_bytes(make_wav_bytes())

    assert processed.sample_rate == 16000
    assert 7900 <= len(processed.samples) <= 8100
    assert processed.source_duration_seconds == pytest.approx(0.5)
    assert np.sqrt(np.mean(processed.samples.astype(np.float64) ** 2)) == pytest.approx(0.08, abs=0.002)


def test_process_bytes_rejects_too_short_audio():
    with pytest.raises(ValueError, match="at least"):
        AudioPreprocessor().process_bytes(make_wav_bytes(duration_seconds=0.1))
