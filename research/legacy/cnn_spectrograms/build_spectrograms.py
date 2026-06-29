# LEGACY: CNN spectrogram baseline deprecated in favor of HuBERT + Bi-LSTM + Attention.
# This script is retained for historical reference only; no active pipeline depends on it.
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.spectrograms import build_spectrogram_records, save_spectrogram_manifest
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5-second log-Mel spectrogram features.")
    parser.add_argument("--manifest", required=True, help="Sample manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to store spectrogram .npy files.")
    parser.add_argument("--output-manifest", required=True, help="CSV manifest for generated spectrograms.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--mlflow-experiment", default="itel-spectrograms")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        frame = build_spectrogram_records(
            sample_manifest_path=args.manifest,
            output_dir=args.output_dir,
            eval_mode="seg_5s",
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
        )
        saved_manifest = save_spectrogram_manifest(frame, args.output_manifest)
        mlflow.set_tags({"stage": "research", "script": "build_spectrograms.py", "feature_type": "logmel", "eval_mode": "seg_5s"})
        log_params({"manifest": args.manifest, "output_dir": args.output_dir, "output_manifest": str(saved_manifest), "limit": args.limit, "skip_existing": not args.no_skip_existing, "tracking_uri": tracking_uri})
        log_dict(
            "metadata/build_spectrograms_summary.json",
            {
                "n_rows": int(len(frame)),
                "n_source_samples": 0 if frame.empty else int(frame["source_sample_id"].nunique()),
                "feature_type": None if frame.empty else str(frame["feature_type"].iloc[0]),
                "n_mels": None if frame.empty else int(frame["n_mels"].iloc[0]),
                "n_frames": None if frame.empty else int(frame["n_frames"].iloc[0]),
            },
        )
        log_artifact(saved_manifest, artifact_path="spectrogram_manifests")
        print(f"Saved spectrogram manifest to: {saved_manifest}")
        print(f"Generated {len(frame)} spectrogram rows in: {args.output_dir}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
