from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class XGBStringLabelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: pd.Series) -> "XGBStringLabelClassifier":
        self.classes_ = np.array(sorted(pd.Series(y).astype(str).unique().tolist()))
        class_to_index = {label: idx for idx, label in enumerate(self.classes_.tolist())}
        y_encoded = np.array([class_to_index[str(label)] for label in y], dtype=np.int64)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]


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


def fit_logreg_tuned_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_linear_svm_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42)),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_xgboost_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", XGBStringLabelClassifier()),
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
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=False,
                    batch_size="auto",
                    n_iter_no_change=25,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


MODEL_TRAINERS = {
    "logreg": fit_logreg_classifier,
    "logreg_tuned": fit_logreg_tuned_classifier,
    "linear_svm": fit_linear_svm_classifier,
    "xgboost": fit_xgboost_classifier,
    "mlp_small": fit_mlp_small_classifier,
}
