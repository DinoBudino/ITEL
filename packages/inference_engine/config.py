from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    model_dir_name: str = field(default_factory=lambda: os.getenv("ITEL_MODEL_DIR", "assets/models"))
    target_sr: int = field(default_factory=lambda: int(os.getenv("ITEL_TARGET_SR", "16000")))
    hp_cutoff: int = field(default_factory=lambda: int(os.getenv("ITEL_HP_CUTOFF", "100")))
    target_duration: float = field(default_factory=lambda: float(os.getenv("ITEL_TARGET_DURATION", "2.0")))
    max_upload_size_mb: int = field(default_factory=lambda: int(os.getenv("ITEL_MAX_UPLOAD_SIZE_MB", "10")))
    model_version: str = field(default_factory=lambda: os.getenv("ITEL_MODEL_VERSION", "itelv5-final-onnx"))
    hubert_bundle: str = field(default_factory=lambda: os.getenv("ITEL_HUBERT_BUNDLE", "HUBERT_BASE"))
    log_level: str = field(default_factory=lambda: os.getenv("ITEL_LOG_LEVEL", "INFO"))
    cors_origins_raw: str = field(
        default_factory=lambda: os.getenv(
            "ITEL_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        )
    )

    @property
    def model_dir(self) -> Path:
        return self.base_dir / self.model_dir_name

    @property
    def final_model_path(self) -> Path:
        return self.model_dir / "final_mlp_model.onnx"

    @property
    def final_scaler_path(self) -> Path:
        return self.model_dir / "final_mlp_scaler.joblib"

    @property
    def hparams_path(self) -> Path:
        return self.model_dir / "mlp_hparams_v5.json"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
