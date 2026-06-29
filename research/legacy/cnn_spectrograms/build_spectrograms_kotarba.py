from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import librosa

from research.src.config import CONFIG
from research.src.preprocessing import fit_to_duration, load_and_preprocess_audio


def log_power_spectrogram(samples: np.ndarray, sr: int, target_size: int = 180) -> np.ndarray:
    D = librosa.stft(y=samples.astype(np.float64), n_fft=512, hop_length=160, win_length=400, window="hann")
    power = np.abs(D) ** 2
    log_power = librosa.power_to_db(power, ref=np.max, top_db=80.0)
    spec = np.array(Image.fromarray(log_power).resize((target_size, target_size), Image.LANCZOS), dtype=np.float32)
    spec = (spec - spec.min()) / max(spec.max() - spec.min(), 1e-8)
    return spec


def parse_args():
    p = argparse.ArgumentParser(description="Build Kotarba-style log-power spectrograms (180x180).")
    p.add_argument("--manifest", default=str(CONFIG.paths.manifests_dir / "cz_main_cleaned.csv"))
    p.add_argument("--output-dir", default=str(CONFIG.paths.processed_dir / "spectrograms_kotarba"))
    p.add_argument("--output-manifest", default=str(CONFIG.paths.processed_manifest_dir / "cz_main_spectrograms_kotarba.csv"))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    samples_df = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(samples_df.itertuples(index=False))
    if args.limit: rows = rows[:args.limit]

    records = []
    for row in tqdm(rows, desc="kotarba specs"):
        sample_id = row.sample_id
        out_path = output_dir / f"{sample_id}.npy"

        if out_path.exists():
            records.append({"sample_id": sample_id, "feature_path": str(out_path),
                            "label": row.label, "subject_id": row.subject_id, "shape": "180x180"})
            continue

        try:
            samples, sr = load_and_preprocess_audio(
                row.raw_audio_path, duration_s=CONFIG.audio.train_segment_s,
                profile=CONFIG.audio.default_profile)
            samples = fit_to_duration(samples, sr, CONFIG.audio.train_segment_s)
            spec = log_power_spectrogram(samples, sr, target_size=180)
            np.save(out_path, spec)
            records.append({"sample_id": sample_id, "feature_path": str(out_path),
                            "label": row.label, "subject_id": row.subject_id, "shape": f"{spec.shape[0]}x{spec.shape[1]}"})
        except Exception as e:
            print(f"ERROR {sample_id}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(args.output_manifest, index=False)
    print(f"Saved {len(df)} records to {args.output_manifest}")


if __name__ == "__main__":
    main()
