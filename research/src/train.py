from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


RECOMMENDED_MODEL_TYPE = "mlp_small"


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: pd.Series) -> "LabelEncodedClassifier":
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(pd.Series(y).astype(str))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(indices)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)


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


def fit_mlp_small_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LabelEncodedClassifier(
                    MLPClassifier(
                        hidden_layer_sizes=(128, 32),
                        activation="relu",
                        solver="adam",
                        alpha=5e-4,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=600,
                        early_stopping=False,
                        batch_size="auto",
                        n_iter_no_change=20,
                        random_state=42,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


MODEL_TRAINERS = {
    "mlp_small": fit_mlp_small_classifier,
    "logreg": fit_logreg_classifier,
}
