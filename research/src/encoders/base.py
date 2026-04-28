from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EmbeddingOutput:
    embedding: np.ndarray
    hidden_size: int
    embedding_dim: int
    model_id: str
    pooling_recipe: str


class BaseSpeechEncoder(Protocol):
    model_id: str
    embedding_dim: int

    def encode(self, samples: np.ndarray, sr: int) -> EmbeddingOutput:
        ...
