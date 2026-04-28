from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_embeddings(embedding_manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    vectors = [np.load(Path(path)) for path in embedding_manifest["embedding_path"].tolist()]
    matrix = np.vstack(vectors)
    return matrix, embedding_manifest


def fit_logreg_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_mlp_classifier(
    X: np.ndarray,
    labels: pd.Series,
    *,
    hidden_layer_sizes: tuple[int, ...] = (256, 64),
    alpha: float = 1e-4,
    random_state: int = 42,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size="auto",
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=False,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline
