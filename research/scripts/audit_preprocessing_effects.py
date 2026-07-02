from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from research.src.config import CONFIG
from research.src.preprocessing import load_and_preprocess_audio


PROFILES = ("multilingual_light", "anti_confound")


def _audio_stats(samples: np.ndarray, sr: int) -> dict[str, float]:
    if len(samples) == 0:
        return {"duration_s": 0.0, "rms": 0.0, "peak_abs": 0.0, "silence_ratio": 1.0}
    return {
        "duration_s": float(len(samples) / sr) if sr else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(samples)))),
        "peak_abs": float(np.max(np.abs(samples))),
        "silence_ratio": float(np.mean(np.abs(samples) < 0.01)),
    }


def _load_raw(path: str) -> tuple[np.ndarray, int]:
    samples, sr = librosa.load(path, sr=None, mono=True)
    return samples.astype(np.float32), int(sr)


def build_audit(manifest_path: str, output: str, limit: int | None = None) -> Path:
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)

    rows: list[dict[str, object]] = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="audit:preprocessing"):
        audio_path = str(getattr(row, "raw_audio_path"))
        try:
            raw_samples, raw_sr = _load_raw(audio_path)
            raw_stats = _audio_stats(raw_samples, raw_sr)
        except Exception as exc:
            rows.append(
                {
                    "sample_id": getattr(row, "sample_id"),
                    "subject_id": getattr(row, "subject_id"),
                    "label": getattr(row, "label"),
                    "task_type": getattr(row, "task_type"),
                    "source_block_code": getattr(row, "notes", None),
                    "error": str(exc),
                }
            )
            continue

        base: dict[str, object] = {
            "sample_id": getattr(row, "sample_id"),
            "subject_id": getattr(row, "subject_id"),
            "label": getattr(row, "label"),
            "task_type": getattr(row, "task_type"),
            "source_block_code": getattr(row, "notes", None),
            "raw_sr": raw_sr,
            "raw_duration_s": raw_stats["duration_s"],
            "raw_rms": raw_stats["rms"],
            "raw_peak_abs": raw_stats["peak_abs"],
            "raw_silence_ratio": raw_stats["silence_ratio"],
        }

        for profile in PROFILES:
            try:
                proc_samples, proc_sr = load_and_preprocess_audio(audio_path, duration_s=None, profile=profile)
                stats = _audio_stats(proc_samples, proc_sr)
                prefix = profile
                base[f"{prefix}_sr"] = proc_sr
                base[f"{prefix}_duration_s"] = stats["duration_s"]
                base[f"{prefix}_duration_retained_ratio"] = stats["duration_s"] / max(raw_stats["duration_s"], 1e-8)
                base[f"{prefix}_rms"] = stats["rms"]
                base[f"{prefix}_peak_abs"] = stats["peak_abs"]
                base[f"{prefix}_silence_ratio"] = stats["silence_ratio"]
            except Exception as exc:
                base[f"{profile}_error"] = str(exc)

        rows.append(base)

    frame = pd.DataFrame(rows)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def summarize(output_csv: Path, summary_output: Path) -> Path:
    frame = pd.read_csv(output_csv)
    summary_rows: list[dict[str, object]] = []
    group_cols = ["label", "task_type", "source_block_code"]
    value_cols = [
        "raw_duration_s",
        "raw_silence_ratio",
        "multilingual_light_duration_s",
        "multilingual_light_silence_ratio",
        "anti_confound_duration_s",
        "anti_confound_duration_retained_ratio",
        "anti_confound_silence_ratio",
    ]
    for group_col in group_cols:
        if group_col not in frame.columns:
            continue
        for group_value, group in frame.groupby(group_col, dropna=False):
            row: dict[str, object] = {"group_col": group_col, "group": group_value, "n_samples": int(len(group))}
            for col in value_cols:
                if col not in group.columns:
                    continue
                values = group[col].dropna().astype(float)
                if values.empty:
                    continue
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_median"] = float(values.median())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    return summary_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit how much each preprocessing profile changes LANNA audio.")
    parser.add_argument(
        "--manifest",
        default=str(CONFIG.paths.manifests_dir / "cz_main_cleaned.csv"),
        help="Input sample manifest.",
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "preprocessing_effects_audit.csv"),
        help="Output per-sample audit CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "preprocessing_effects_summary.csv"),
        help="Output grouped summary CSV.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_audit(args.manifest, args.output, limit=args.limit)
    summary = summarize(output, Path(args.summary_output))
    print(f"Saved audit:   {output}")
    print(f"Saved summary: {summary}")


if __name__ == "__main__":
    main()
