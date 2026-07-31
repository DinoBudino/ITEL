from __future__ import annotations

import numpy as np


FEATURE_RECIPE_ID = "hubert-base-last-temporal-stats-v1"
STATISTICAL_BLOCK_NAMES = (
    "temporal_mean",
    "temporal_std",
    "first_quarter_mean",
    "last_quarter_mean",
    "first_last_difference",
    "temporal_correlation",
)


def statistical_sequence_embedding(sequence) -> np.ndarray:
    values = sequence.detach().cpu().numpy() if hasattr(sequence, "detach") else np.asarray(sequence)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"Expected a non-empty 2D sequence, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Sequence contains non-finite values.")

    time_steps = values.shape[0]
    quarter_length = max(1, time_steps // 4)
    mean_embedding = values.mean(axis=0)
    std_embedding = values.std(axis=0, ddof=1) if time_steps > 1 else np.full(values.shape[1], np.nan)
    first_quarter = values[:quarter_length].mean(axis=0)
    last_quarter = values[-quarter_length:].mean(axis=0)

    if time_steps < 2:
        slope = np.zeros(values.shape[1], dtype=np.float64)
    else:
        centered_time = np.arange(time_steps, dtype=np.float64)
        centered_time -= centered_time.mean()
        centered_values = values - mean_embedding
        denominator = np.linalg.norm(centered_time) * np.linalg.norm(centered_values, axis=0)
        numerator = centered_time @ centered_values
        slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    return np.concatenate(
        [mean_embedding, std_embedding, first_quarter, last_quarter, first_quarter - last_quarter, slope]
    )
