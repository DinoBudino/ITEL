from __future__ import annotations

import numpy as np
import pandas as pd

from research.src import multilingual_evaluation as evaluation


def _subject_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [row[0] for row in rows],
            "label": [row[1] for row in rows],
            "embedding": [np.array([row[2]], dtype=float) for row in rows],
        }
    )


def test_aggregate_subject_embeddings_groups_and_averages_samples() -> None:
    samples = pd.DataFrame(
        {
            "subject_id": ["P1", "H1", "P1"],
            "label": ["Patients", "Healthy", "Patients"],
            "embedding": [np.array([2.0, 4.0]), np.array([1.0, 3.0]), np.array([4.0, 8.0])],
        }
    )

    subjects = evaluation.aggregate_subject_embeddings(samples).set_index("subject_id")

    assert subjects.loc["P1", "label"] == "Patients"
    assert subjects.loc["P1", "n_samples"] == 2
    np.testing.assert_allclose(subjects.loc["P1", "embedding"], [3.0, 6.0])
    np.testing.assert_allclose(subjects.loc["H1", "embedding"], [1.0, 3.0])


def test_centroid_shift_aligns_healthy_centroids_without_mutating_source() -> None:
    source = _subject_frame([("H0", "Healthy", 1.0), ("P0", "Patients", 5.0)])
    target_train = _subject_frame([("H1", "Healthy", 4.0), ("P1", "Patients", 8.0)])

    shifted = evaluation.shift_source_toward_target(source, target_train)

    np.testing.assert_allclose(evaluation.healthy_centroid(shifted), [4.0])
    np.testing.assert_allclose(shifted.iloc[1]["embedding"], [8.0])
    np.testing.assert_allclose(source.iloc[0]["embedding"], [1.0])


def test_loso_centroid_uses_only_target_training_subjects(monkeypatch) -> None:
    source = _subject_frame([("S-H", "Healthy", 0.0), ("S-P", "Patients", 10.0)])
    target = _subject_frame(
        [
            ("H1", "Healthy", 2.0),
            ("H2", "Healthy", 4.0),
            ("P1", "Patients", 8.0),
            ("P2", "Patients", 9.0),
        ]
    )
    fitted_matrices: list[np.ndarray] = []

    class FakeModel:
        classes_ = np.array(["Healthy", "Patients"])

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            patient_score = np.clip(X[:, 0] / 10.0, 0.01, 0.99)
            return np.column_stack([1.0 - patient_score, patient_score])

    def fake_fit_model(model_type: str, X: np.ndarray, y: pd.Series) -> FakeModel:
        fitted_matrices.append(X.copy())
        return FakeModel()

    monkeypatch.setattr(evaluation, "fit_model", fake_fit_model)

    evaluation.run_target_loso(source, target, "logreg", "centroid_shift")

    # Fold order is H1, H2, P1, P2. The first training row is the shifted
    # source healthy embedding, whose destination excludes the held-out row.
    assert [matrix[0, 0] for matrix in fitted_matrices] == [4.0, 2.0, 3.0, 3.0]


def test_best_threshold_maximizes_balanced_accuracy() -> None:
    threshold, metrics = evaluation.best_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.2, 0.4, 0.6, 0.8]),
    )

    assert threshold == 0.6
    assert metrics["balanced_accuracy"] == 1.0


def test_fit_model_returns_probability_classifier_with_patient_class() -> None:
    X = np.array([[0.0], [0.2], [0.4], [1.0], [1.2], [1.4]])
    labels = pd.Series(["Healthy"] * 3 + ["Patients"] * 3)

    model = evaluation.fit_model("logreg", X, labels)

    assert model.predict_proba(X).shape == (6, 2)
    assert evaluation.positive_class_index(model) == list(model.classes_).index("Patients")
