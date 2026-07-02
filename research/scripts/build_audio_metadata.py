from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import wave

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from research.src.config import CONFIG

REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def _read_wav_header(audio_path: str) -> dict[str, object]:
    info: dict[str, object] = {
        "sr_original": None,
        "channels": None,
        "subtype": None,
        "source_format": None,
        "n_frames_header": None,
    }
    path = Path(audio_path)
    info["source_format"] = path.suffix.lstrip(".").lower() if path.suffix else None

    if not path.is_file():
        return info

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                info["sr_original"] = wf.getframerate()
                info["channels"] = wf.getnchannels()
                info["subtype"] = f"PCM_{wf.getsampwidth() * 8}"
                info["n_frames_header"] = wf.getnframes()
        except Exception:
            pass
    else:
        try:
            import soundfile as sf
            sf_info = sf.info(str(path))
            info["sr_original"] = sf_info.samplerate
            info["channels"] = sf_info.channels
            info["subtype"] = sf_info.subtype
            info["n_frames_header"] = sf_info.frames
        except Exception:
            pass

    return info


def _extract_acoustic_features(samples: np.ndarray, sr: int) -> dict[str, float]:
    if len(samples) == 0:
        return {k: 0.0 for k in _acoustic_feature_names()}

    samples_f32 = samples.astype(np.float32)
    duration_s = float(len(samples_f32)) / sr if sr else 0.0
    peak_abs = float(np.max(np.abs(samples_f32)))
    rms = float(np.sqrt(np.mean(np.square(samples_f32))))
    rms_frame = librosa.feature.rms(y=samples_f32, frame_length=2048, hop_length=512)[0]
    silence_ratio = float(np.mean(np.abs(samples_f32) < 0.01))

    zcr = librosa.feature.zero_crossing_rate(y=samples_f32, frame_length=2048, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=samples_f32, sr=sr, n_fft=2048, hop_length=512)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=samples_f32, sr=sr, n_fft=2048, hop_length=512)[0]
    rolloff = librosa.feature.spectral_rolloff(y=samples_f32, sr=sr, n_fft=2048, hop_length=512)[0]
    flatness = librosa.feature.spectral_flatness(y=samples_f32, n_fft=2048, hop_length=512)[0]

    return {
        "duration_s": round(duration_s, 6),
        "peak_abs": round(peak_abs, 6),
        "rms_mean": round(float(np.mean(rms_frame)), 6),
        "rms": round(rms, 6),
        "silence_ratio": round(silence_ratio, 6),
        "zcr_mean": round(float(np.mean(zcr)), 6),
        "zcr_std": round(float(np.std(zcr)), 6),
        "spectral_centroid_mean": round(float(np.mean(centroid)), 4),
        "spectral_bandwidth_mean": round(float(np.mean(bandwidth)), 4),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 4),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 8),
    }


def _acoustic_feature_names() -> list[str]:
    return [
        "duration_s", "peak_abs", "rms_mean", "rms", "silence_ratio",
        "zcr_mean", "zcr_std",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "spectral_rolloff_mean", "spectral_flatness_mean",
    ]


def _wav_header_columns() -> list[str]:
    return ["sr_original", "channels", "subtype", "source_format", "n_frames_header"]


def _resolve_audio_path(row) -> str:
    processed = getattr(row, "processed_audio_path", None)
    if processed and Path(str(processed)).exists():
        return str(processed)
    return str(getattr(row, "raw_audio_path"))


def build_metadata(manifest_path: str, output_csv: str, force: bool = False) -> Path:
    manifest = pd.read_csv(manifest_path)
    output_path = Path(output_csv)

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path)
        if "sample_id" in existing.columns and len(existing) == len(manifest):
            print(f"Metadata cache exists with {len(existing)} rows. Use --force to rebuild.")
            return output_path

    metadata_columns = [
        "sample_id", "subject_id", "label", "language", "task_type",
    ]

    extra_info_cols = []
    for col in ["source_block_code", "articulatory_group", "notes"]:
        if col in manifest.columns:
            extra_info_cols.append(col)
            metadata_columns.append(col)

    metadata_columns += _wav_header_columns()
    metadata_columns += _acoustic_feature_names()

    rows: list[dict[str, object]] = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="metadata:audio"):
        audio_path = _resolve_audio_path(row)
        header = _read_wav_header(audio_path)

        try:
            samples, sr = librosa.load(audio_path, sr=None, mono=True)
            sr = int(sr)
        except Exception:
            samples, sr = np.array([], dtype=np.float32), 0

        acoustic = _extract_acoustic_features(samples, sr if sr else 1)
        acoustic["duration_s"] = float(len(samples)) / sr if sr else 0.0
        acoustic["duration_s"] = round(acoustic["duration_s"], 6)

        record = {
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "label": row.label,
            "language": getattr(row, "language", "unknown"),
            "task_type": getattr(row, "task_type", "unknown"),
        }
        for col in extra_info_cols:
            record[col] = getattr(row, col, None)

        record.update(header)
        record.update(acoustic)
        rows.append(record)

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved audio metadata to: {output_path} ({len(frame)} rows)")
    return output_path


def generate_summary(frame: pd.DataFrame, output_json: str) -> Path:
    by_label = {}
    for label in sorted(frame["label"].unique()):
        sub = frame[frame["label"] == label]
        by_label[label] = {
            "n_samples": int(len(sub)),
            "n_subjects": int(sub["subject_id"].nunique()),
        }
        for feat in _acoustic_feature_names():
            if feat not in sub.columns:
                continue
            values = sub[feat].dropna()
            if len(values) == 0:
                continue
            by_label[label][feat] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std()), 6),
                "median": round(float(values.median()), 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }

    if "channels" in frame.columns:
        for label in sorted(frame["label"].unique()):
            sub = frame[frame["label"] == label]
            by_label[label]["channel_distribution"] = (
                sub["channels"].value_counts().to_dict()
            )

    if "source_format" in frame.columns:
        for label in sorted(frame["label"].unique()):
            sub = frame[frame["label"] == label]
            by_label[label]["format_distribution"] = (
                sub["source_format"].value_counts().to_dict()
            )

    summary = {
        "manifest_summary": {
            "n_total_samples": int(len(frame)),
            "n_total_subjects": int(frame["subject_id"].nunique()),
            "samples_by_label": {str(k): int(v) for k, v in frame["label"].value_counts().to_dict().items()},
        },
        "by_label": by_label,
    }

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary to: {json_path}")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build master audio metadata table from a sample manifest.")
    parser.add_argument(
        "--manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Path to the sample manifest CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(REPORT_CSV_DIR / "cz_audio_metadata_master.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-summary",
        default=str(REPORT_CSV_DIR / "cz_audio_metadata_summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if cache exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_metadata(args.manifest, args.output, force=args.force)
    frame = pd.read_csv(out)
    generate_summary(frame, args.output_summary)


if __name__ == "__main__":
    main()
