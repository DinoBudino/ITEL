from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from packages.audio_pipeline.pooling import statistical_sequence_embedding as _shared_statistical_sequence_embedding


STATISTICAL_BLOCK_NAMES = (
    "Media temporal",
    "Desviación estándar",
    "Media primer cuarto",
    "Media último cuarto",
    "Diferencia primero-último",
    "Pendiente temporal",
)


def statistical_sequence_embedding(sequence: torch.Tensor) -> np.ndarray:
    return _shared_statistical_sequence_embedding(sequence)


def load_statistical_sequence_embedding(sequence_path: str | Path, layer_policy: str) -> np.ndarray:
    payload = torch.load(sequence_path, map_location="cpu", weights_only=True)
    sequence = payload["sequence"].float()

    if sequence.dim() == 3:
        if layer_policy == "last":
            sequence = sequence[-1]
        elif layer_policy == "mean_layers":
            sequence = sequence.mean(dim=0)
        else:
            raise ValueError(f"Unsupported layer_policy for 3D sequence: {layer_policy}")

    return statistical_sequence_embedding(sequence)
