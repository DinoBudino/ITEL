from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset_id: str
    language: str
    subject_id: str
    session_id: str | None
    label: str
    task_type: str
    speaker_role: str
    split_role: str
    eval_mode: str
    raw_audio_path: str
    processed_audio_path: str
    interviewer_audio_path: str | None = None
    duration_s: float | None = None
    sr_original: int | None = None
    sr_final: int | None = None
    source_format: str | None = None
    diarization_status: str = "not_required"
    review_status: str | None = None
    review_reason: str | None = None
    auto_qc_flag: str | None = None
    chi_duration_s: float | None = None
    total_speech_s: float | None = None
    chi_speech_share: float | None = None
    n_detected_speakers: int | None = None
    qc_flag: str | None = None
    notes: str | None = None
    articulatory_group: str | None = None


SAMPLE_MANIFEST_COLUMNS = list(SampleRecord.__dataclass_fields__.keys())


@dataclass(frozen=True)
class EmbeddingRecord:
    sample_id: str
    source_sample_id: str
    dataset_id: str
    eval_mode: str
    embedding_model_id: str
    embedding_dim: int
    preproc_version: str
    pooling_recipe: str
    chunk_policy: str
    embedding_path: str
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    segment_qc_status: str | None = None
    segment_qc_reason: str | None = None
    segment_rms_mean: float | None = None
    segment_silence_ratio: float | None = None
    created_at: str | None = None
    notes: str | None = None


EMBEDDING_MANIFEST_COLUMNS = list(EmbeddingRecord.__dataclass_fields__.keys())


def records_to_frame(records: Iterable[SampleRecord | EmbeddingRecord]) -> pd.DataFrame:
    rows = [asdict(record) for record in records]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def save_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    return manifest_path


def ensure_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
