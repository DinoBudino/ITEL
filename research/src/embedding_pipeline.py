from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CONFIG
from .data_manifest import EmbeddingRecord, load_manifest, records_to_frame, save_manifest
from .encoders import XLSREncoder
from .preprocessing import fit_to_duration, load_and_preprocess_audio
from .segmentation import compute_segment_qc, create_fixed_segments


def _is_excluded_flag(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(text) and text.startswith("exclude_")


def _preprocessing_profile_for_dataset(dataset_id: str) -> str:
    if dataset_id in {CONFIG.datasets.cz_dataset_id, CONFIG.datasets.en_dataset_id}:
        return CONFIG.audio.default_profile
    return "standard"


def _preproc_version_for_dataset(dataset_id: str) -> str:
    profile = _preprocessing_profile_for_dataset(dataset_id)
    return f"{CONFIG.preproc_version}_{profile}"


def build_embedding_records(
    sample_manifest_path: str | Path,
    output_dir: str | Path,
    eval_mode: str,
    encoder: XLSREncoder | None = None,
    skip_existing: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    encoder = encoder or XLSREncoder()
    manifest = load_manifest(sample_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[EmbeddingRecord] = []
    rows = list(manifest.itertuples(index=False))
    if limit is not None:
        rows = rows[:limit]

    for row in tqdm(rows, desc=f"embeddings:{eval_mode}"):
        sample_id = str(getattr(row, "sample_id"))
        dataset_id = str(getattr(row, "dataset_id"))
        raw_audio_path = str(getattr(row, "raw_audio_path"))
        created_at = datetime.now(timezone.utc).isoformat()
        input_audio_path = raw_audio_path
        preprocessing_profile = _preprocessing_profile_for_dataset(dataset_id)
        preproc_version = _preproc_version_for_dataset(dataset_id)

        if dataset_id == CONFIG.datasets.en_dataset_id:
            qc_flag = getattr(row, "qc_flag", None)
            if _is_excluded_flag(qc_flag):
                continue
            if not Path(input_audio_path).exists():
                raise FileNotFoundError(
                    f"Raw English audio not found for sample '{sample_id}': {input_audio_path}"
                )

        if eval_mode == "train_default":
            samples, sr = load_and_preprocess_audio(
                input_audio_path,
                duration_s=CONFIG.audio.train_segment_s,
                profile=preprocessing_profile,
            )
            encoded = encoder.encode(samples, sr)
            embedding_path = output_dir / f"{sample_id}_{eval_mode}.npy"
            if skip_existing and embedding_path.exists():
                existing_embedding = np.load(embedding_path)
                embedding_dim = int(existing_embedding.shape[0])
                model_id = encoder.model_id
                pooling_recipe = CONFIG.encoder.pooling_recipe
            else:
                np.save(embedding_path, encoded.embedding)
                embedding_dim = encoded.embedding_dim
                model_id = encoded.model_id
                pooling_recipe = encoded.pooling_recipe
            records.append(
                EmbeddingRecord(
                    sample_id=sample_id,
                    source_sample_id=sample_id,
                    dataset_id=dataset_id,
                    eval_mode=eval_mode,
                    embedding_model_id=model_id,
                    embedding_dim=embedding_dim,
                    preproc_version=preproc_version,
                    pooling_recipe=pooling_recipe,
                    chunk_policy="single_window_2s",
                    embedding_path=str(embedding_path),
                    created_at=created_at,
                )
            )
            continue

        if eval_mode in {"seg_2s", "seg_5s"}:
            segment_duration = CONFIG.audio.eval_segment_2s if eval_mode == "seg_2s" else CONFIG.audio.eval_segment_5s
            samples, sr = load_and_preprocess_audio(input_audio_path, duration_s=None, profile=preprocessing_profile)
            segments = create_fixed_segments(
                audio_id=sample_id,
                samples=samples,
                sr=sr,
                segment_duration_s=segment_duration,
                allow_overlap=False,
            )
            segment_qc = {}
            if dataset_id == CONFIG.datasets.en_dataset_id:
                segment_qc = compute_segment_qc(segments, target_duration_s=segment_duration)
            for segment in segments:
                embedding_path = output_dir / f"{segment.segment_id}.npy"
                qc = segment_qc.get(segment.segment_id)
                if skip_existing and embedding_path.exists():
                    existing_embedding = np.load(embedding_path)
                    embedding_dim = int(existing_embedding.shape[0])
                    model_id = encoder.model_id
                    pooling_recipe = CONFIG.encoder.pooling_recipe
                else:
                    segment_samples = fit_to_duration(segment.samples, sr, segment_duration)
                    encoded = encoder.encode(segment_samples, sr)
                    np.save(embedding_path, encoded.embedding)
                    embedding_dim = encoded.embedding_dim
                    model_id = encoded.model_id
                    pooling_recipe = encoded.pooling_recipe
                records.append(
                    EmbeddingRecord(
                        sample_id=segment.segment_id,
                        source_sample_id=sample_id,
                        dataset_id=dataset_id,
                        eval_mode=eval_mode,
                        embedding_model_id=model_id,
                        embedding_dim=embedding_dim,
                        preproc_version=preproc_version,
                        pooling_recipe=pooling_recipe,
                        chunk_policy=f"fixed_non_overlapping_{segment_duration:.1f}s",
                        embedding_path=str(embedding_path),
                        segment_start_s=segment.start_s,
                        segment_end_s=segment.end_s,
                        segment_qc_status=qc.status if qc else None,
                        segment_qc_reason=qc.reason if qc else None,
                        segment_rms_mean=qc.rms_mean if qc else None,
                        segment_silence_ratio=qc.silence_ratio if qc else None,
                        created_at=created_at,
                    )
                )
            continue

        if eval_mode == "full_audio":
            samples, sr = load_and_preprocess_audio(input_audio_path, duration_s=None, profile=preprocessing_profile)
            embedding_path = output_dir / f"{sample_id}_{eval_mode}.npy"
            if skip_existing and embedding_path.exists():
                existing_embedding = np.load(embedding_path)
                embedding_dim = int(existing_embedding.shape[0])
                model_id = encoder.model_id
                pooling_recipe = CONFIG.encoder.pooling_recipe
            else:
                encoded = encoder.encode(samples, sr)
                np.save(embedding_path, encoded.embedding)
                embedding_dim = encoded.embedding_dim
                model_id = encoded.model_id
                pooling_recipe = encoded.pooling_recipe
            records.append(
                EmbeddingRecord(
                    sample_id=sample_id,
                    source_sample_id=sample_id,
                    dataset_id=dataset_id,
                    eval_mode=eval_mode,
                    embedding_model_id=model_id,
                    embedding_dim=embedding_dim,
                    preproc_version=preproc_version,
                    pooling_recipe=pooling_recipe,
                    chunk_policy="full_audio_single_embedding",
                    embedding_path=str(embedding_path),
                    created_at=created_at,
                )
            )
            continue

        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    return records_to_frame(records)


def save_embedding_manifest(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_manifest(frame, output_path)
