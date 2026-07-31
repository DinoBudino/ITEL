from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class CacheHit:
    value: np.ndarray
    source: str


class FeatureCache:
    def __init__(self, root_dir: Path | None = None, max_memory_entries: int = 512):
        self.root_dir = root_dir
        self.max_memory_entries = max(0, max_memory_entries)
        self._known: dict[tuple[str, str, str, str], np.ndarray] = {}
        self._memory: OrderedDict[tuple[str, str, str, str], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.validation_errors: list[str] = []
        if root_dir is not None:
            self._load_known_caches()

    @staticmethod
    def audio_digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def get(self, model_id: str, model_version: str, contract_id: str, payload: bytes) -> CacheHit | None:
        key = (model_id, model_version, contract_id, self.audio_digest(payload))
        with self._lock:
            if key in self._known:
                return CacheHit(self._known[key].copy(), "precomputed_cache")
            value = self._memory.get(key)
            if value is None:
                return None
            self._memory.move_to_end(key)
            return CacheHit(value.copy(), "memory_cache")

    def put(self, model_id: str, model_version: str, contract_id: str, payload: bytes, value: np.ndarray) -> None:
        if self.max_memory_entries == 0:
            return
        key = (model_id, model_version, contract_id, self.audio_digest(payload))
        stored = np.asarray(value).copy()
        with self._lock:
            self._memory[key] = stored
            self._memory.move_to_end(key)
            while len(self._memory) > self.max_memory_entries:
                self._memory.popitem(last=False)

    @property
    def known_entry_count(self) -> int:
        return len(self._known)

    @property
    def memory_entry_count(self) -> int:
        with self._lock:
            return len(self._memory)

    def _load_known_caches(self) -> None:
        if not self.root_dir or not self.root_dir.exists():
            return
        for manifest_path in sorted(self.root_dir.glob("**/cache-manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("contains_audio") is not False or manifest.get("contains_labels") is not False:
                    raise ValueError("cache manifest must explicitly exclude audio and labels")
                artifact_path = (manifest_path.parent / manifest["artifact"]["file"]).resolve()
                artifact_path.relative_to(manifest_path.parent.resolve())
                expected = manifest["artifact"]["sha256"]
                if _sha256_file(artifact_path) != expected:
                    raise ValueError("artifact checksum mismatch")
                with np.load(artifact_path) as payload:
                    hashes = payload["hashes"].astype(str)
                    embeddings = payload["embeddings"]
                    if embeddings.ndim != 2 or len(hashes) != embeddings.shape[0]:
                        raise ValueError("invalid cache array shapes")
                    if embeddings.shape[1] != int(manifest["embedding_dimension"]):
                        raise ValueError("embedding dimension differs from cache manifest")
                    for digest, embedding in zip(hashes, embeddings, strict=True):
                        key = (
                            manifest["model_id"], manifest["model_version"],
                            manifest["feature_contract"], str(digest),
                        )
                        self._known[key] = np.asarray(embedding).copy()
            except Exception as exc:
                self.validation_errors.append(f"{manifest_path}: {exc}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
