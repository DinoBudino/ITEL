from __future__ import annotations

from pathlib import Path

from packages.audio_pipeline import HubertFeatureExtractor

from .czech import CzechLayerMixRuntime
from .manifest import ModelManifest, load_manifest
from .multilingual import MultilingualRuntime
from .feature_cache import FeatureCache


RUNTIME_TYPES = {
    "multilingual_logreg_pca": MultilingualRuntime,
    "czech_layermix": CzechLayerMixRuntime,
}


class ModelRegistry:
    def __init__(self, registry_dir: Path, feature_cache_dir: Path | None = None, memory_cache_entries: int = 512):
        self.registry_dir = registry_dir
        self.manifests: dict[str, ModelManifest] = {}
        self._validation_errors: dict[str, list[str]] = {}
        self._runtimes = {}
        self.extractor = HubertFeatureExtractor("HUBERT_BASE")
        self.feature_cache = FeatureCache(feature_cache_dir, max_memory_entries=memory_cache_entries)
        self._warmup_errors: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        manifests: dict[str, ModelManifest] = {}
        if self.registry_dir.exists():
            for path in sorted(self.registry_dir.glob("**/manifest.json")):
                manifest = load_manifest(path)
                if manifest.model_id in manifests:
                    raise ValueError(f"Duplicate model_id: {manifest.model_id}")
                manifests[manifest.model_id] = manifest
        self.manifests = manifests
        self._validation_errors = {model_id: manifest.validate_artifacts() for model_id, manifest in manifests.items()}
        self._runtimes.clear()

    def catalog(self) -> list[dict]:
        return [
            {
                "id": manifest.model_id,
                "name": manifest.display_name,
                "version": manifest.version,
                "languages": manifest.languages,
                "prediction_unit": manifest.payload.get("prediction_unit"),
                "available": not self._validation_errors[manifest.model_id],
                "validation_errors": self._validation_errors[manifest.model_id],
            }
            for manifest in self.manifests.values()
        ]

    def readiness_errors(self) -> dict[str, list[str]]:
        errors = {
            model_id: errors
            for model_id, manifest in self.manifests.items()
            if (errors := self._validation_errors[model_id])
        }
        if self.feature_cache.validation_errors:
            errors["feature_cache"] = self.feature_cache.validation_errors
        if self._warmup_errors:
            errors["warmup"] = self._warmup_errors
        return errors

    def warmup(self) -> None:
        self._warmup_errors.clear()
        if self.readiness_errors():
            return
        try:
            self.extractor.warmup()
            for model_id in self.manifests:
                runtime = self.get_runtime(model_id)
                if hasattr(runtime, "warmup"):
                    runtime.warmup()
        except Exception as exc:
            self._warmup_errors.append(str(exc))

    def get_runtime(self, model_id: str):
        manifest = self.manifests.get(model_id)
        if manifest is None:
            raise KeyError(f"Unknown model: {model_id}")
        errors = self._validation_errors[model_id]
        if errors:
            raise RuntimeError(f"Model {model_id} is not ready: {'; '.join(errors)}")
        if model_id not in self._runtimes:
            runtime_type = RUNTIME_TYPES.get(manifest.family)
            if runtime_type is None:
                raise RuntimeError(f"Unsupported model family: {manifest.family}")
            self._runtimes[model_id] = runtime_type(manifest, extractor=self.extractor, feature_cache=self.feature_cache)
        return self._runtimes[model_id]
