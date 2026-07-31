from __future__ import annotations

import numpy as np
import pytest

from research.scripts.benchmark_chunked_serving import chunk_boundaries, embedding_distances, score_frozen_head


def test_chunk_boundaries_cover_short_input_once() -> None:
    assert chunk_boundaries(7, 10, 2) == [(0, 7)]


def test_chunk_boundaries_apply_overlap_and_cover_tail() -> None:
    assert chunk_boundaries(25, 10, 2) == [(0, 10), (8, 18), (16, 25)]


@pytest.mark.parametrize("overlap", [-1, 10, 11])
def test_chunk_boundaries_reject_invalid_overlap(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap_samples"):
        chunk_boundaries(20, 10, overlap)


def test_embedding_distances_report_identical_and_orthogonal_vectors() -> None:
    assert embedding_distances([1.0, 2.0], [1.0, 2.0]) == pytest.approx((1.0, 0.0))
    assert embedding_distances([1.0, 0.0], [0.0, 1.0]) == pytest.approx((0.0, np.sqrt(2.0)))


def test_embedding_distances_handle_zero_vectors() -> None:
    assert embedding_distances([0.0, 0.0], [0.0, 0.0]) == (1.0, 0.0)
    assert embedding_distances([0.0, 0.0], [1.0, 0.0]) == (0.0, 1.0)


def test_score_frozen_head_uses_threshold() -> None:
    head = {
        "scaler_mean": np.array([0.0, 0.0]),
        "scaler_scale": np.array([1.0, 1.0]),
        "pca_mean": np.array([0.0, 0.0]),
        "pca_components": np.eye(2),
        "classifier_coef": np.array([[1.0, 0.0]]),
        "classifier_intercept": np.array([0.0]),
        "threshold": np.array([0.6]),
    }

    probability, label = score_frozen_head([1.0, 0.0], head)

    assert probability == pytest.approx(0.7310585786)
    assert label == "Patients"
