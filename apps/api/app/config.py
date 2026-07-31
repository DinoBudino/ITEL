from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _as_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ApiSettings:
    base_dir: Path = field(default_factory=lambda: Path(os.getenv("ITEL_BASE_DIR", Path.cwd())).resolve())
    model_registry_dir: str = field(default_factory=lambda: os.getenv("ITEL_MODEL_REGISTRY_DIR", "models/registry"))
    sample_audio_dir: str = field(default_factory=lambda: os.getenv("ITEL_SAMPLE_AUDIO_DIR", "data"))
    feature_cache_dir: str = field(default_factory=lambda: os.getenv("ITEL_FEATURE_CACHE_DIR", "artifacts/serving_cache"))
    memory_cache_entries: int = field(default_factory=lambda: int(os.getenv("ITEL_MEMORY_CACHE_ENTRIES", "512")))
    enable_sample_library: bool = field(default_factory=lambda: _as_bool("ITEL_ENABLE_SAMPLE_LIBRARY"))
    enable_demo_library: bool = field(default_factory=lambda: _as_bool("ITEL_ENABLE_DEMO_LIBRARY"))
    demo_catalog_file: str = field(default_factory=lambda: os.getenv("ITEL_DEMO_CATALOG_FILE", "artifacts/demo_catalog/.sample_audio_catalog.json"))
    demo_czech_dir: str = field(default_factory=lambda: os.getenv("ITEL_DEMO_CZECH_DIR", "research/LANNA_Czech_Data"))
    demo_english_dir: str = field(default_factory=lambda: os.getenv("ITEL_DEMO_ENGLISH_DIR", "research/English_Data"))
    max_files_per_subject: int = field(default_factory=lambda: int(os.getenv("ITEL_MAX_FILES_PER_SUBJECT", "50")))
    max_upload_size_mb: int = field(default_factory=lambda: int(os.getenv("ITEL_MAX_UPLOAD_SIZE_MB", "30")))
    max_total_upload_size_mb: int = field(default_factory=lambda: int(os.getenv("ITEL_MAX_TOTAL_UPLOAD_SIZE_MB", "50")))
    cors_origins_raw: str = field(
        default_factory=lambda: os.getenv(
            "ITEL_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        )
    )

    @property
    def registry_path(self) -> Path:
        return self.base_dir / self.model_registry_dir

    @property
    def sample_audio_path(self) -> Path:
        return self.base_dir / self.sample_audio_dir

    @property
    def feature_cache_path(self) -> Path:
        return self.base_dir / self.feature_cache_dir

    @property
    def demo_catalog_path(self) -> Path:
        return self.base_dir / self.demo_catalog_file

    @property
    def demo_czech_path(self) -> Path:
        return self.base_dir / self.demo_czech_dir

    @property
    def demo_english_path(self) -> Path:
        return self.base_dir / self.demo_english_dir

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_total_upload_size_bytes(self) -> int:
        return self.max_total_upload_size_mb * 1024 * 1024

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]
