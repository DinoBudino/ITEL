from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

from .schemas import SampleAudioItem


@lru_cache(maxsize=4)
def load_catalog(base_dir: Path) -> tuple[list[SampleAudioItem], dict[str, Path]]:
    manifest = base_dir / ".sample_audio_catalog.json"
    if not manifest.is_file():
        return [], {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    items = [SampleAudioItem.model_validate(item) for item in payload]
    paths: dict[str, Path] = {}
    safe_items: list[SampleAudioItem] = []
    for item in items:
        path = (base_dir / item.relative_path).resolve()
        try:
            path.relative_to(base_dir.resolve())
        except ValueError:
            continue
        if path.is_file():
            safe_items.append(item)
            paths[item.id] = path
    return safe_items, paths


def opaque_sample_id(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")
