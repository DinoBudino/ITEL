from __future__ import annotations

import hashlib
import json

import numpy as np

from packages.model_runtime.feature_cache import FeatureCache


def test_known_cache_is_content_addressed_and_validated(tmp_path):
    payload = b"known audio"
    digest = hashlib.sha256(payload).hexdigest()
    embedding = np.arange(12, dtype=np.float64)
    artifact = tmp_path / "known-features.npz"
    np.savez_compressed(artifact, hashes=np.asarray([digest]), embeddings=embedding.reshape(1, -1))
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "cache-manifest.json").write_text(json.dumps({
        "model_id": "multilingual-cz-en",
        "model_version": "1.0.0",
        "feature_contract": "stats-v1",
        "embedding_dimension": 12,
        "contains_audio": False,
        "contains_labels": False,
        "artifact": {"file": artifact.name, "sha256": checksum},
    }), encoding="utf-8")

    cache = FeatureCache(tmp_path)
    hit = cache.get("multilingual-cz-en", "1.0.0", "stats-v1", payload)

    assert cache.validation_errors == []
    assert cache.known_entry_count == 1
    assert hit is not None and hit.source == "precomputed_cache"
    np.testing.assert_array_equal(hit.value, embedding)
    assert cache.get("other-model", "1.0.0", "stats-v1", payload) is None
    assert cache.get("multilingual-cz-en", "2.0.0", "stats-v1", payload) is None


def test_memory_cache_is_bounded_and_does_not_write_files(tmp_path):
    cache = FeatureCache(max_memory_entries=1)
    cache.put("model", "1", "contract", b"first", np.array([1.0]))
    cache.put("model", "1", "contract", b"second", np.array([2.0]))

    assert cache.memory_entry_count == 1
    assert cache.get("model", "1", "contract", b"first") is None
    hit = cache.get("model", "1", "contract", b"second")
    assert hit is not None and hit.source == "memory_cache"
    assert list(tmp_path.iterdir()) == []


def test_invalid_known_cache_is_not_used(tmp_path):
    artifact = tmp_path / "known-features.npz"
    np.savez_compressed(artifact, hashes=np.asarray(["abc"]), embeddings=np.zeros((1, 2)))
    (tmp_path / "cache-manifest.json").write_text(json.dumps({
        "model_id": "model",
        "model_version": "1",
        "feature_contract": "contract",
        "embedding_dimension": 2,
        "contains_audio": False,
        "contains_labels": False,
        "artifact": {"file": artifact.name, "sha256": "invalid"},
    }), encoding="utf-8")

    cache = FeatureCache(tmp_path)

    assert cache.validation_errors
    assert cache.known_entry_count == 0
