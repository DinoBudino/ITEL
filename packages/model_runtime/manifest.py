from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelManifest:
    path: Path
    payload: dict

    @property
    def model_id(self) -> str:
        return str(self.payload["model_id"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def family(self) -> str:
        return str(self.payload["family"])

    @property
    def languages(self) -> list[str]:
        return [str(value) for value in self.payload["languages"]]

    @property
    def display_name(self) -> str:
        return str(self.payload.get("display_name", self.model_id))

    @property
    def artifact_dir(self) -> Path:
        return self.path.parent

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def validate_contract(self) -> list[str]:
        errors: list[str] = []
        required = {"schema_version", "model_id", "version", "family", "languages", "prediction_unit", "classes", "artifacts"}
        missing = sorted(required - self.payload.keys())
        if missing:
            errors.append(f"Missing manifest fields: {', '.join(missing)}")
        if self.payload.get("prediction_unit") != "subject":
            errors.append("Only subject-level manifests are supported.")
        if self.payload.get("classes") != ["Healthy", "Patients"]:
            errors.append("classes must be ['Healthy', 'Patients'].")
        return errors

    def validate_artifacts(self) -> list[str]:
        errors = self.validate_contract()
        for artifact in self.payload.get("artifacts", []):
            file_path = (self.artifact_dir / str(artifact["file"])).resolve()
            try:
                file_path.relative_to(self.artifact_dir.resolve())
            except ValueError:
                errors.append(f"Artifact escapes bundle directory: {artifact['file']}")
                continue
            if not file_path.is_file():
                errors.append(f"Missing artifact: {artifact['file']}")
                continue
            expected = artifact.get("sha256")
            if self.payload.get("status") == "promoted" and not expected:
                errors.append(f"Promoted artifact has no checksum: {artifact['file']}")
            if expected and _sha256(file_path) != expected:
                errors.append(f"Checksum mismatch: {artifact['file']}")
        if not errors:
            errors.extend(self._validate_family_artifacts())
        return errors

    def _validate_family_artifacts(self) -> list[str]:
        if self.family == "multilingual_logreg_pca":
            return self._validate_multilingual_heads()
        if self.family == "czech_layermix":
            return self._validate_czech_checkpoint()
        return [f"Unsupported model family: {self.family}"]

    def _validate_multilingual_heads(self) -> list[str]:
        import numpy as np

        errors: list[str] = []
        feature_dim = int(self.payload.get("feature_recipe", {}).get("dimension", 0))
        required = {"source_healthy_centroid", "target_healthy_centroid", "scaler_mean", "scaler_scale", "pca_mean", "pca_components", "classifier_coef", "classifier_intercept", "threshold"}
        for language in self.languages:
            file_name = self.payload.get("language_heads", {}).get(language)
            if not file_name:
                errors.append(f"Missing head mapping for language: {language}")
                continue
            try:
                with np.load(self.artifact_dir / file_name) as head:
                    missing = sorted(required - set(head.files))
                    if missing:
                        errors.append(f"{file_name} is missing arrays: {', '.join(missing)}")
                        continue
                    if head["scaler_mean"].shape != (feature_dim,) or head["scaler_scale"].shape != (feature_dim,):
                        errors.append(f"{file_name} scaler shape does not match feature dimension {feature_dim}")
                    if head["pca_mean"].shape != (feature_dim,) or head["pca_components"].ndim != 2 or head["pca_components"].shape[1] != feature_dim:
                        errors.append(f"{file_name} PCA shape does not match feature dimension {feature_dim}")
                    if head["classifier_coef"].size != head["pca_components"].shape[0]:
                        errors.append(f"{file_name} classifier input does not match PCA output")
                    if np.any(head["scaler_scale"] == 0) or not all(np.isfinite(head[key]).all() for key in required):
                        errors.append(f"{file_name} contains invalid numeric parameters")
                    threshold = float(head["threshold"].reshape(-1)[0])
                    if not 0 <= threshold <= 1:
                        errors.append(f"{file_name} has an invalid threshold")
            except Exception as exc:
                errors.append(f"Could not validate {file_name}: {exc}")
        return errors

    def _validate_czech_checkpoint(self) -> list[str]:
        import torch

        checkpoint_file = self.payload.get("checkpoint_file")
        try:
            checkpoint = torch.load(self.artifact_dir / checkpoint_file, map_location="cpu", weights_only=True)
        except Exception as exc:
            return [f"Could not validate Czech checkpoint: {exc}"]
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            return ["Czech checkpoint has no state_dict."]
        if checkpoint.get("model_config") and checkpoint["model_config"] != self.payload.get("model_config"):
            return ["Czech checkpoint model_config differs from its manifest."]
        threshold = float(self.payload.get("threshold", -1))
        return [] if 0 <= threshold <= 1 else ["Czech manifest has an invalid threshold."]


def load_manifest(path: Path) -> ModelManifest:
    return ModelManifest(path=path, payload=json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
