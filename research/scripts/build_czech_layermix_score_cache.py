from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from packages.model_runtime.torch_models import AttentionBiLSTMClassifierV3
from research.src.sequence_model import SequenceV3Dataset, collate_sequences


CONTRACT_ID = "hubert-base-all-12-layermix-score-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a private Czech LayerMix score cache from final model sequences.")
    parser.add_argument("--audio-manifest", default=str(ROOT_DIR / "research" / "manifests" / "cz_main_cleaned.csv"))
    parser.add_argument("--sequence-manifest", default=str(ROOT_DIR / "research" / "processed_data" / "manifests" / "hubert_sequences_cleaned_natural_12layers.csv"))
    parser.add_argument("--model-manifest", default=str(ROOT_DIR / "models" / "registry" / "czech-layermix" / "1.0.0" / "manifest.json"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "artifacts" / "serving_cache" / "czech-layermix" / "1.0.0"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    audio_manifest = pd.read_csv(args.audio_manifest)
    sequence_manifest = pd.read_csv(args.sequence_manifest)
    audio_by_sample = dict(zip(audio_manifest["sample_id"].astype(str), audio_manifest["raw_audio_path"].astype(str), strict=True))
    if set(sequence_manifest["sample_id"].astype(str)) != set(audio_by_sample):
        raise ValueError("Audio and sequence manifests do not contain the same sample IDs.")

    model_manifest_path = Path(args.model_manifest)
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = model_manifest_path.parent / model_manifest["checkpoint_file"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_config") and checkpoint["model_config"] != model_manifest["model_config"]:
        raise ValueError("Checkpoint and model manifest configurations differ.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AttentionBiLSTMClassifierV3(**model_manifest["model_config"])
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model.eval().to(device)

    loader = DataLoader(
        SequenceV3Dataset(sequence_manifest), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_sequences, num_workers=0,
    )
    score_by_sample: dict[str, float] = {}
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["sequence"].to(device), batch["mask"].to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            score_by_sample.update(zip(batch["sample_id"], (float(value) for value in scores), strict=True))

    scores_by_hash: dict[str, list[float]] = {}
    for sample_id, score in score_by_sample.items():
        audio_path = Path(audio_by_sample[sample_id])
        digest = sha256_file(audio_path)
        scores_by_hash.setdefault(digest, []).append(score)

    unambiguous_scores: dict[str, float] = {}
    ambiguous_hashes = 0
    for digest, scores in scores_by_hash.items():
        if max(scores) - min(scores) > 1e-6:
            ambiguous_hashes += 1
            continue
        unambiguous_scores[digest] = float(np.mean(scores))
    hashes = list(unambiguous_scores)
    values = np.asarray([[unambiguous_scores[digest]] for digest in hashes], dtype=np.float64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "known-scores.npz"
    np.savez_compressed(artifact_path, hashes=np.asarray(hashes, dtype="U64"), embeddings=values)
    cache_manifest = {
        "schema_version": 1,
        "model_id": model_manifest["model_id"],
        "model_version": model_manifest["version"],
        "feature_contract": CONTRACT_ID,
        "embedding_dimension": 1,
        "entry_count": len(hashes),
        "source_sample_count": len(score_by_sample),
        "ambiguous_hash_count": ambiguous_hashes,
        "contains_audio": False,
        "contains_labels": False,
        "artifact": {"file": artifact_path.name, "sha256": sha256_file(artifact_path)},
    }
    manifest_path = output_dir / "cache-manifest.json"
    manifest_path.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    print(f"Czech LayerMix score cache: {manifest_path}")
    print(f"Device: {device}  samples={len(score_by_sample)}  cached_hashes={len(hashes)}  ambiguous_hashes={ambiguous_hashes}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
