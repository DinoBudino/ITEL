from __future__ import annotations

from io import BytesIO
import wave

import numpy as np

from packages.inference_engine.preprocessing import AudioPreprocessor


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


def test_process_bytes_normalizes_audio_to_target_shape():
    preprocessor = AudioPreprocessor(target_sr=16000, target_duration=2.0)
    processed = preprocessor.process_bytes(make_wav_bytes())

    assert processed.sample_rate == 16000
    assert len(processed.samples) == 32000
    assert processed.duration_seconds == 2.0
    assert np.max(np.abs(processed.samples)) <= 1.0
