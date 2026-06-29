# LEGACY: Spectrogram utilities for the deprecated CNN baseline.
# This module is retained for historical reference only.
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .data_manifest import load_manifest, save_manifest
from .preprocessing import fit_to_duration, load_and_preprocess_audio
from .segmentation import compute_segment_qc, create_fixed_segments


@dataclass(frozen=True)
class SpectrogramRecord:
    sample_id: str
    source_sample_id: str
    dataset_id: str
    eval_mode: str
    feature_type: str
    n_mels: int
    n_frames: int
    preproc_version: str
    chunk_policy: str
    feature_path: str
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    segment_qc_status: str | None = None
    segment_qc_reason: str | None = None
    segment_rms_mean: float | None = None
    segment_silence_ratio: float | None = None
    created_at: str | None = None
    notes: str | None = None


SPECTROGRAM_MANIFEST_COLUMNS = list(SpectrogramRecord.__dataclass_fields__.keys())


def spectrogram_records_to_frame(records: Iterable[SpectrogramRecord]) -> pd.DataFrame:
    rows = [asdict(record) for record in records]
    if not rows:
        return pd.DataFrame(columns=SPECTROGRAM_MANIFEST_COLUMNS)
    return pd.DataFrame(rows)


def save_spectrogram_manifest(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_manifest(frame, output_path)


def _feature_type() -> str:
    return f"logmel_{CONFIG.spectrogram.n_mels}x{CONFIG.spectrogram.target_frames_5s}"


def _mel_spectrogram(samples: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(samples, dtype=np.float32),
        sr=sr,
        n_fft=CONFIG.spectrogram.n_fft,
        hop_length=CONFIG.spectrogram.hop_length,
        win_length=CONFIG.spectrogram.win_length,
        n_mels=CONFIG.spectrogram.n_mels,
        fmin=CONFIG.spectrogram.fmin,
        fmax=CONFIG.spectrogram.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=CONFIG.spectrogram.top_db)
    if mel_db.shape[1] != CONFIG.spectrogram.target_frames_5s:
        mel_db = librosa.util.fix_length(mel_db, size=CONFIG.spectrogram.target_frames_5s, axis=1)
    if CONFIG.spectrogram.normalize_per_sample:
        mean = float(mel_db.mean())
        std = float(mel_db.std())
        if std > 1e-8:
            mel_db = (mel_db - mean) / std
        else:
            mel_db = mel_db - mean
    return mel_db.astype(np.float32)


def build_spectrogram_records(
    sample_manifest_path: str | Path,
    output_dir: str | Path,
    eval_mode: str = "seg_5s",
    skip_existing: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    if eval_mode != "seg_5s":
        raise ValueError("The current spectrogram baseline supports only eval_mode='seg_5s'.")

    manifest = load_manifest(sample_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(manifest.itertuples(index=False))
    if limit is not None:
        rows = rows[:limit]

    records: list[SpectrogramRecord] = []
    segment_duration = CONFIG.audio.eval_segment_5s
    preproc_version = f"{CONFIG.preproc_version}_{CONFIG.audio.default_profile}"

    for row in tqdm(rows, desc=f"spectrograms:{eval_mode}"):
        sample_id = str(getattr(row, "sample_id"))
        dataset_id = str(getattr(row, "dataset_id"))
        raw_audio_path = str(getattr(row, "raw_audio_path"))
        samples, sr = load_and_preprocess_audio(raw_audio_path, duration_s=None, profile=CONFIG.audio.default_profile)
        segments = create_fixed_segments(
            audio_id=sample_id,
            samples=samples,
            sr=sr,
            segment_duration_s=segment_duration,
            allow_overlap=False,
        )
        segment_qc = compute_segment_qc(segments, target_duration_s=segment_duration) if dataset_id == CONFIG.datasets.en_dataset_id else {}
        for segment in segments:
            qc = segment_qc.get(segment.segment_id)
            feature_path = output_dir / f"{segment.segment_id}.npy"
            if not (skip_existing and feature_path.exists()):
                segment_samples = fit_to_duration(segment.samples, sr, segment_duration)
                mel_db = _mel_spectrogram(segment_samples, sr)
                np.save(feature_path, mel_db)
            records.append(
                SpectrogramRecord(
                    sample_id=segment.segment_id,
                    source_sample_id=sample_id,
                    dataset_id=dataset_id,
                    eval_mode=eval_mode,
                    feature_type=_feature_type(),
                    n_mels=CONFIG.spectrogram.n_mels,
                    n_frames=CONFIG.spectrogram.target_frames_5s,
                    preproc_version=preproc_version,
                    chunk_policy=f"fixed_non_overlapping_{segment_duration:.1f}s",
                    feature_path=str(feature_path),
                    segment_start_s=segment.start_s,
                    segment_end_s=segment.end_s,
                    segment_qc_status=qc.status if qc else None,
                    segment_qc_reason=qc.reason if qc else None,
                    segment_rms_mean=qc.rms_mean if qc else None,
                    segment_silence_ratio=qc.silence_ratio if qc else None,
                )
            )
    return spectrogram_records_to_frame(records)
