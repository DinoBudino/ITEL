from __future__ import annotations

import numpy as np
import torch

from research.src.multilingual_features import statistical_sequence_embedding
from research.src.sequence_model import collate_sequences


def test_statistical_sequence_embedding_preserves_temporal_blocks() -> None:
    sequence = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    embedding = statistical_sequence_embedding(sequence)

    np.testing.assert_allclose(embedding[0:2], [4.0, 5.0])
    np.testing.assert_allclose(embedding[4:6], [1.0, 2.0])
    np.testing.assert_allclose(embedding[6:8], [7.0, 8.0])
    np.testing.assert_allclose(embedding[8:10], [-6.0, -6.0])
    np.testing.assert_allclose(embedding[10:12], [1.0, 1.0])


def test_collate_sequences_uses_temporal_length_for_3d_layer_mix() -> None:
    batch = [
        {
            "sequence": torch.ones(2, 3, 5),
            "mask": torch.tensor([True, True, False]),
            "label": torch.tensor(0.0),
            "sample_id": "short",
            "subject_id": "H1",
        },
        {
            "sequence": torch.full((2, 7, 5), 2.0),
            "mask": torch.ones(7, dtype=torch.bool),
            "label": torch.tensor(1.0),
            "sample_id": "long",
            "subject_id": "P1",
        },
    ]

    collated = collate_sequences(batch)

    assert collated["sequence"].shape == (2, 2, 7, 5)
    assert collated["mask"].shape == (2, 7)
    torch.testing.assert_close(collated["sequence"][0, :, :3], batch[0]["sequence"])
    assert torch.count_nonzero(collated["sequence"][0, :, 3:]) == 0
    torch.testing.assert_close(collated["mask"][0], torch.tensor([True, True, False, False, False, False, False]))
