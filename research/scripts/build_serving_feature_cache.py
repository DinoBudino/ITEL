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

from packages.audio_pipeline.pooling import FEATURE_RECIPE_ID
from research.src.multilingual_features import load_statistical_sequence_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a private content-addressed serving cache from existing HuBERT sequences.")
    parser.add_argument("--audio-manifest", default=str(ROOT_DIR / "research" / "manifests" / "english_test_samples.csv"))
    parser.add_argument("--sequence-manifest", default=str(ROOT_DIR / "research" / "processed_data" / "manifests" / "hubert_sequences_en_test_v2.csv"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "artifacts" / "serving_cache" / "multilingual-cz-en" / "1.0.0"))
    parser.add_argument("--model-id", default="multilingual-cz-en")
    parser.add_argument("--model-version", default="1.0.0")
    return parser.parse_args()


def build_cache(audio_manifest_path: Path, sequence_manifest_path: Path, output_dir: Path, model_id: str, model_version: str) -> Path:
    audio_manifest = pd.read_csv(audio_manifest_path)
    sequence_manifest = pd.read_csv(sequence_manifest_path)
    required_audio = {"sample_id", "raw_audio_path"}
    required_sequence = {"sample_id", "sequence_path"}
    if missing := required_audio - set(audio_manifest.columns):
        raise ValueError(f"Audio manifest is missing columns: {sorted(missing)}")
    if missing := required_sequence - set(sequence_manifest.columns):
        raise ValueError(f"Sequence manifest is missing columns: {sorted(missing)}")

    joined = audio_manifest[["sample_id", "raw_audio_path"]].merge(
        sequence_manifest[["sample_id", "sequence_path"]], on="sample_id", how="inner", validate="one_to_one"
    )
    if len(joined) != len(audio_manifest) or len(joined) != len(sequence_manifest):
        raise ValueError("Audio and sequence manifests do not contain the same sample IDs.")

    embedding_by_hash: dict[str, np.ndarray] = {}
    for row in joined.itertuples(index=False):
        audio_path = Path(row.raw_audio_path)
        sequence_path = Path(row.sequence_path)
        if not audio_path.is_file() or not sequence_path.is_file():
            raise FileNotFoundError(f"Missing source for sample {row.sample_id}")
        digest = sha256_file(audio_path)
        embedding = load_statistical_sequence_embedding(sequence_path, "last")
        previous = embedding_by_hash.get(digest)
        if previous is not None and not np.allclose(previous, embedding, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Duplicate audio hash has incompatible embeddings: {row.sample_id}")
        embedding_by_hash[digest] = embedding

    hashes = list(embedding_by_hash)
    matrix = np.vstack([embedding_by_hash[digest] for digest in hashes]).astype(np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "known-features.npz"
    np.savez_compressed(artifact_path, hashes=np.asarray(hashes, dtype="U64"), embeddings=matrix)
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "model_version": model_version,
        "feature_contract": FEATURE_RECIPE_ID,
        "embedding_dimension": int(matrix.shape[1]),
        "entry_count": len(hashes),
        "source_sample_count": len(joined),
        "contains_audio": False,
        "contains_labels": False,
        "artifact": {"file": artifact_path.name, "sha256": sha256_file(artifact_path)},
    }
    manifest_path = output_dir / "cache-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest = build_cache(
        Path(args.audio_manifest), Path(args.sequence_manifest), Path(args.output_dir), args.model_id, args.model_version
    )
    print(f"Serving feature cache: {manifest}")


if __name__ == "__main__":
    main()
