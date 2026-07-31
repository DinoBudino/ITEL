from __future__ import annotations

import numpy as np


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one embedding is required.")
    matrix = np.vstack([np.asarray(embedding, dtype=np.float64) for embedding in embeddings])
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values.")
    return matrix.mean(axis=0)
