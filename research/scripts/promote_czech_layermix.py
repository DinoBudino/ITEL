from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a final, non-fold Czech LayerMix checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "models" / "registry" / "czech-layermix" / "1.0.0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model_config")
    if not model_config or not model_config.get("use_layer_mix"):
        raise ValueError("Checkpoint must include the final LayerMix model_config.")
    threshold = args.threshold if args.threshold is not None else checkpoint.get("best_threshold")
    if threshold is None or not 0 <= float(threshold) <= 1:
        raise ValueError("A valid frozen threshold is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "model.pt"
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    preprocessing = checkpoint.get("preprocessing", {})
    payload = {
        "schema_version": 1,
        "model_id": "czech-layermix",
        "display_name": "Czech V3 LayerMix",
        "version": "1.0.0",
        "family": "czech_layermix",
        "status": "promoted",
        "prediction_unit": "subject",
        "languages": ["cs"],
        "classes": ["Healthy", "Patients"],
        "positive_class": "Patients",
        "threshold": float(threshold),
        "encoder": {"bundle": "HUBERT_BASE", "layer_policy": "all_12", "hidden_size": 768},
        "preprocessing": {
            "sample_rate": int(preprocessing.get("target_sr", 16000)),
            "highpass_hz": int(preprocessing.get("highpass_hz", 100)),
            "target_rms": float(preprocessing.get("target_rms", 0.08)),
            "max_peak": float(preprocessing.get("max_peak_after_rms", 0.99)),
            "min_duration_seconds": 0.3,
        },
        "model_config": model_config,
        "checkpoint_file": "model.pt",
        "artifacts": [{"file": "model.pt", "sha256": digest}],
        "limitations": ["Validated on the Czech LANNA domain", "Not a clinical diagnostic device"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Promoted Czech LayerMix bundle: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
