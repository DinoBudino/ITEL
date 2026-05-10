from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - optional dependency for analysis-only runs
    XGBClassifier = None


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
        if XGBClassifier is None:
            raise ModuleNotFoundError("xgboost is required for the xgboost model family.")
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


def fit_logreg_pca_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=256, random_state=42)),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=4000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_linear_svm_pca_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=256, random_state=42)),
            ("clf", SVC(kernel="linear", C=0.5, class_weight="balanced", probability=True, random_state=42)),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_mlp_medium_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LabelEncodedClassifier(
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate="adaptive",
                        learning_rate_init=7.5e-4,
                        max_iter=800,
                        early_stopping=False,
                        batch_size="auto",
                        n_iter_no_change=30,
                        random_state=42,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(X, labels)
    return pipeline


def fit_mlp_pca_classifier(X: np.ndarray, labels: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=256, random_state=42)),
            (
                "clf",
                LabelEncodedClassifier(
                    MLPClassifier(
                        hidden_layer_sizes=(128, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-2,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=800,
                        early_stopping=False,
                        batch_size="auto",
                        n_iter_no_change=30,
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
    "mlp_medium": fit_mlp_medium_classifier,
    "mlp_pca": fit_mlp_pca_classifier,
    "logreg": fit_logreg_classifier,
    "logreg_tuned": fit_logreg_tuned_classifier,
    "linear_svm": fit_linear_svm_classifier,
    "xgboost": fit_xgboost_classifier,
    "logreg_pca": fit_logreg_pca_classifier,
    "linear_svm_pca": fit_linear_svm_pca_classifier,
}
