"""Offline experiment comparing chunked HuBERT inference with full-sequence embeddings.

This script is deliberately separate from serving and only reads manifests, audio,
precomputed sequences, and an optional frozen head. It never writes model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from packages.audio_pipeline import (
    AudioPreprocessor,
    AudioProfile,
    HubertFeatureExtractor,
    ProcessedAudio,
    statistical_sequence_embedding,
)
from research.src.multilingual_features import load_statistical_sequence_embedding


DEFAULT_OUTPUT_DIR = ROOT_DIR / "research" / "research_results" / "serving_benchmarks"
MODEL_REGISTRY_DIR = (ROOT_DIR / "models" / "registry").resolve()


def chunk_boundaries(total_samples: int, chunk_samples: int, overlap_samples: int = 0) -> list[tuple[int, int]]:
    """Return complete, ordered chunk boundaries using half-open sample offsets."""
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be non-negative and smaller than chunk_samples.")

    boundaries: list[tuple[int, int]] = []
    step = chunk_samples - overlap_samples
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        boundaries.append((start, end))
        if end == total_samples:
            break
        start += step
    return boundaries


def embedding_distances(reference: Sequence[float], candidate: Sequence[float]) -> tuple[float, float]:
    """Return cosine similarity and L2 distance for two finite vectors."""
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError("Embeddings must be non-empty vectors with matching shapes.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Embeddings must contain only finite values.")

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        cosine = 1.0
    elif left_norm == 0.0 or right_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        cosine = float(np.clip(cosine, -1.0, 1.0))
    return cosine, float(np.linalg.norm(left - right))


def score_frozen_head(embedding: Sequence[float], head: dict[str, np.ndarray]) -> tuple[float, str]:
    """Apply the same frozen scaler/PCA/logistic path as multilingual serving."""
    vector = np.asarray(embedding, dtype=np.float64)
    scaled = (vector - head["scaler_mean"]) / head["scaler_scale"]
    projected = (scaled - head["pca_mean"]) @ head["pca_components"].T
    logit = float(projected @ head["classifier_coef"].reshape(-1) + head["classifier_intercept"].reshape(-1)[0])
    probability = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
    threshold = float(head["threshold"].reshape(-1)[0])
    return probability, "Patients" if probability >= threshold else "Healthy"


def _resolve_data_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    return manifest_relative if manifest_relative.exists() else ROOT_DIR / path


def _load_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    sequence_manifest_path = Path(args.sequence_manifest).expanduser().resolve()
    rows = pd.read_csv(sequence_manifest_path)
    required = {"sample_id", "sequence_path"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Sequence manifest is missing columns: {', '.join(missing)}")

    if args.audio_manifest:
        audio_manifest_path = Path(args.audio_manifest).expanduser().resolve()
        audio = pd.read_csv(audio_manifest_path)
        if "sample_id" not in audio or args.audio_path_column not in audio:
            raise ValueError(f"Audio manifest must contain sample_id and {args.audio_path_column}.")
        if audio["sample_id"].duplicated().any():
            raise ValueError("Audio manifest sample_id values must be unique.")
        rows = rows.drop(columns=[args.audio_path_column], errors="ignore").merge(
            audio[["sample_id", args.audio_path_column]], on="sample_id", how="left", validate="many_to_one"
        )
        audio_base = audio_manifest_path
    else:
        audio_base = sequence_manifest_path

    if args.audio_path_column not in rows:
        raise ValueError(
            f"No {args.audio_path_column} column found. Supply --audio-manifest to join audio paths by sample_id."
        )
    if args.sample_ids:
        requested = set(args.sample_ids)
        rows = rows[rows["sample_id"].astype(str).isin(requested)]
        found = set(rows["sample_id"].astype(str))
        if requested - found:
            raise ValueError(f"Unknown sample IDs: {', '.join(sorted(requested - found))}")
    if args.limit is not None:
        rows = rows.iloc[: args.limit]
    if rows.empty:
        raise ValueError("No manifest rows were selected.")
    if rows[args.audio_path_column].isna().any():
        raise ValueError("Selected rows contain missing audio paths.")
    return rows.reset_index(drop=True), audio_base


def _load_head(path: str | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    with np.load(Path(path).expanduser().resolve()) as payload:
        return {key: payload[key] for key in payload.files}


def _extract_chunk(
    extractor: HubertFeatureExtractor,
    audio: ProcessedAudio,
    precision: str,
) -> np.ndarray:
    import torch

    model = extractor._load_model()  # Shared loader; this experimental path needs dtype control.
    dtype = torch.float16 if precision == "fp16" else torch.float32
    waveform = torch.from_numpy(audio.samples).unsqueeze(0).to(device=extractor.device, dtype=dtype)
    with torch.inference_mode():
        feature_list, _ = model.extract_features(waveform)
    return feature_list[-1].squeeze(0).float().cpu().numpy()


def _extract_chunked_sequence(
    extractor: HubertFeatureExtractor,
    processed: ProcessedAudio,
    chunk_seconds: float,
    overlap_seconds: float,
    precision: str,
) -> tuple[np.ndarray, int]:
    sample_rate = processed.sample_rate
    overlap_samples = round(overlap_seconds * sample_rate)
    boundaries = chunk_boundaries(len(processed.samples), round(chunk_seconds * sample_rate), overlap_samples)
    sequences: list[np.ndarray] = []
    for index, (start, end) in enumerate(boundaries):
        chunk = ProcessedAudio(
            samples=processed.samples[start:end],
            sample_rate=sample_rate,
            source_duration_seconds=(end - start) / sample_rate,
        )
        sequence = _extract_chunk(extractor, chunk, precision)
        if index and overlap_samples:
            trim = min(sequence.shape[0] - 1, round(overlap_samples / (end - start) * sequence.shape[0]))
            sequence = sequence[trim:]
        sequences.append(sequence)
    return np.concatenate(sequences, axis=0), len(boundaries)


def _profile_one(
    extractor: HubertFeatureExtractor,
    processed: ProcessedAudio,
    baseline: np.ndarray,
    head: dict[str, np.ndarray] | None,
    chunk_seconds: float,
    overlap_seconds: float,
    precision: str,
) -> dict[str, Any]:
    import torch

    uses_cuda = extractor.device is not None and extractor.device.type == "cuda"
    if uses_cuda:
        torch.cuda.synchronize(extractor.device)
        torch.cuda.reset_peak_memory_stats(extractor.device)
    started = time.perf_counter()
    sequence, chunk_count = _extract_chunked_sequence(
        extractor, processed, chunk_seconds, overlap_seconds, precision
    )
    embedding = statistical_sequence_embedding(sequence)
    if uses_cuda:
        torch.cuda.synchronize(extractor.device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    cuda_max_memory_bytes = int(torch.cuda.max_memory_allocated(extractor.device)) if uses_cuda else None
    cosine, l2 = embedding_distances(baseline, embedding)
    result: dict[str, Any] = {
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "precision": precision,
        "chunk_count": chunk_count,
        "latency_ms": latency_ms,
        "cuda_max_memory_bytes": cuda_max_memory_bytes,
        "embedding_cosine": cosine,
        "embedding_l2": l2,
    }
    if head is not None:
        baseline_score, baseline_label = score_frozen_head(baseline, head)
        chunked_score, chunked_label = score_frozen_head(embedding, head)
        result.update(
            baseline_target_en_score=baseline_score,
            chunked_target_en_score=chunked_score,
            target_en_score_delta=chunked_score - baseline_score,
            baseline_target_en_label=baseline_label,
            chunked_target_en_label=chunked_label,
            target_en_label_agreement=chunked_label == baseline_label,
        )
    return result


def _safe_output_dir(value: str | Path) -> Path:
    output_dir = Path(value).expanduser().resolve()
    if output_dir == MODEL_REGISTRY_DIR or MODEL_REGISTRY_DIR in output_dir.parents:
        raise ValueError("Benchmark output cannot be written inside models/registry.")
    return output_dir


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "samples": ("sample_id", "count"),
        "duration_seconds_mean": ("duration_seconds", "mean"),
        "latency_ms_mean": ("latency_ms", "mean"),
        "latency_ms_p95": ("latency_ms", lambda values: float(values.quantile(0.95))),
        "embedding_cosine_mean": ("embedding_cosine", "mean"),
        "embedding_cosine_min": ("embedding_cosine", "min"),
        "embedding_l2_mean": ("embedding_l2", "mean"),
    }
    if frame["cuda_max_memory_bytes"].notna().any():
        aggregations["cuda_max_memory_bytes_max"] = ("cuda_max_memory_bytes", "max")
    if "target_en_score_delta" in frame:
        aggregations["target_en_score_delta_mean"] = ("target_en_score_delta", "mean")
        aggregations["target_en_score_delta_abs_max"] = (
            "target_en_score_delta",
            lambda values: float(values.abs().max()),
        )
        aggregations["target_en_label_agreement_rate"] = ("target_en_label_agreement", "mean")
    return frame.groupby(["chunk_seconds", "overlap_seconds", "precision"], as_index=False).agg(**aggregations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OFFLINE/EXPERIMENTAL: benchmark chunked HuBERT against precomputed full sequences."
    )
    parser.add_argument("--sequence-manifest", required=True, help="CSV containing sample_id and sequence_path.")
    parser.add_argument("--audio-manifest", help="Optional CSV joined by sample_id to obtain audio paths.")
    parser.add_argument("--audio-path-column", default="raw_audio_path")
    parser.add_argument("--sample-ids", nargs="+", help="Optional explicit sample IDs.")
    parser.add_argument("--limit", type=int, help="Use only the first N selected rows.")
    parser.add_argument("--chunk-seconds", nargs="+", type=float, default=[10.0, 20.0, 30.0])
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"], default=["fp32"])
    parser.add_argument("--device", help="Torch device; auto-selects CUDA when available.")
    parser.add_argument("--bundle", default="HUBERT_BASE")
    parser.add_argument("--head-path", help="Optional frozen target-English .npz head (read-only).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", help="Output stem; defaults to a UTC timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if any(value <= 0 for value in args.chunk_seconds):
        raise ValueError("Chunk sizes must be positive.")
    if args.overlap_seconds < 0 or any(args.overlap_seconds >= value for value in args.chunk_seconds):
        raise ValueError("Overlap must be non-negative and smaller than every chunk size.")

    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if "fp16" in args.precisions and device.type != "cuda":
        raise ValueError("fp16 benchmarking requires a CUDA device.")

    rows, audio_manifest_path = _load_rows(args)
    sequence_manifest_path = Path(args.sequence_manifest).expanduser().resolve()
    head = _load_head(args.head_path)
    preprocessor = AudioPreprocessor(AudioProfile())
    extractor = HubertFeatureExtractor(args.bundle, device=str(device))
    extractor.preload()
    model = extractor._load_model()
    records: list[dict[str, Any]] = []

    for precision in args.precisions:
        dtype = torch.float16 if precision == "fp16" else torch.float32
        model.to(dtype=dtype)
        for row in rows.to_dict("records"):
            sample_id = str(row["sample_id"])
            sequence_path = _resolve_data_path(row["sequence_path"], sequence_manifest_path)
            audio_path = _resolve_data_path(row[args.audio_path_column], audio_manifest_path)
            baseline = load_statistical_sequence_embedding(sequence_path, "last")
            processed = preprocessor.process_bytes(audio_path.read_bytes())
            for chunk_seconds in args.chunk_seconds:
                result = _profile_one(
                    extractor,
                    processed,
                    baseline,
                    head,
                    chunk_seconds,
                    args.overlap_seconds,
                    precision,
                )
                result.update(
                    sample_id=sample_id,
                    duration_seconds=processed.source_duration_seconds,
                    sequence_path=str(sequence_path),
                    audio_path=str(audio_path),
                )
                records.append(result)
            del baseline, processed

    details = pd.DataFrame(records)
    summary = _aggregate(details)
    output_dir = _safe_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or time.strftime("chunked_%Y%m%dT%H%M%SZ", time.gmtime())
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be a file-name stem, not a path.")
    details_path = output_dir / f"{run_name}_samples.csv"
    summary_path = output_dir / f"{run_name}_summary.csv"
    json_path = output_dir / f"{run_name}_summary.json"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        "experimental_offline_only": True,
        "config": {
            "sequence_manifest": str(sequence_manifest_path),
            "audio_manifest": str(audio_manifest_path),
            "chunk_seconds": args.chunk_seconds,
            "overlap_seconds": args.overlap_seconds,
            "precisions": args.precisions,
            "device": str(device),
            "bundle": args.bundle,
            "layer_policy": "last",
            "head_path": str(Path(args.head_path).resolve()) if args.head_path else None,
        },
        "selected_samples": len(rows),
        "aggregate": summary.where(pd.notna(summary), None).to_dict("records"),
        "sample_results_csv": str(details_path),
        "aggregate_csv": str(summary_path),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Sample results: {details_path}")
    print(f"Aggregate CSV: {summary_path}")
    print(f"Aggregate JSON: {json_path}")


if __name__ == "__main__":
    main()
