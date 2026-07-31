from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research.src.train import LabelEncodedClassifier


def aggregate_subject_embeddings(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject_id, label), group in frame.groupby(["subject_id", "label"], dropna=False):
        embedding = np.vstack(group["embedding"].to_list()).mean(axis=0)
        rows.append(
            {
                "subject_id": str(subject_id),
                "label": str(label),
                "n_samples": int(len(group)),
                "embedding": embedding,
            }
        )
    return pd.DataFrame(rows)


def healthy_centroid(frame: pd.DataFrame) -> np.ndarray:
    healthy = frame.loc[frame["label"] == "Healthy", "embedding"]
    if healthy.empty:
        raise ValueError("No Healthy rows available for centroid computation.")
    return np.vstack(healthy.to_list()).mean(axis=0)


def shift_source_toward_target(source_frame: pd.DataFrame, target_train_frame: pd.DataFrame) -> pd.DataFrame:
    source_healthy_centroid = healthy_centroid(source_frame)
    target_healthy_centroid = healthy_centroid(target_train_frame)
    shifted = source_frame.copy()
    shifted["embedding"] = shifted["embedding"].apply(
        lambda embedding: embedding - source_healthy_centroid + target_healthy_centroid
    )
    return shifted


def fit_model(model_type: str, X: np.ndarray, y: pd.Series):
    if model_type == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs")),
        ])
    elif model_type == "linear_svm":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42)),
        ])
    elif model_type == "logreg_pca":
        n_components = int(min(64, X.shape[0] - 1, X.shape[1]))
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, solver="lbfgs")),
        ])
    elif model_type == "mlp_small":
        model = Pipeline([
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
        ])
    elif model_type == "mlp_medium":
        model = Pipeline([
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
        ])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def positive_class_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps") and "clf" in model.named_steps:
        classes = list(getattr(model.named_steps["clf"], "classes_", []))
    if "Patients" in classes:
        return classes.index("Patients")
    if 1 in classes:
        return classes.index(1)
    if len(classes) == 2:
        return 1
    raise ValueError(f"Could not resolve positive class in {classes}")


def scores_to_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | int | None] = {
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    return metrics


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float | int | None]]:
    best_threshold_value = 0.5
    best_metrics = None
    for threshold in np.unique(np.round(y_score, 4)):
        metrics = scores_to_metrics(y_true, y_score, threshold=float(threshold))
        if best_metrics is None or float(metrics["balanced_accuracy"]) > float(best_metrics["balanced_accuracy"]):
            best_metrics = metrics
            best_threshold_value = float(threshold)
    return best_threshold_value, best_metrics or scores_to_metrics(y_true, y_score, threshold=0.5)


def run_target_loso(
    source_subject: pd.DataFrame,
    target_subject: pd.DataFrame,
    model_type: str,
    alignment_method: str,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    target_ids = sorted(set(target_subject["subject_id"].astype(str).tolist()))

    for test_subject in target_ids:
        target_train = target_subject.loc[target_subject["subject_id"].astype(str) != test_subject].copy()
        target_test = target_subject.loc[target_subject["subject_id"].astype(str) == test_subject].copy()

        source_train = source_subject.copy()
        if alignment_method == "centroid_shift":
            source_train = shift_source_toward_target(source_train, target_train)

        train_frame = pd.concat([source_train, target_train], ignore_index=True)
        X_train = np.vstack(train_frame["embedding"].to_list())
        y_train = train_frame["label"].reset_index(drop=True)
        model = fit_model(model_type, X_train, y_train)
        positive_index = positive_class_index(model)

        train_scores = model.predict_proba(np.vstack(target_train["embedding"].to_list()))[:, positive_index]
        train_y = (target_train["label"] == "Patients").astype(int).to_numpy()
        threshold, _ = best_threshold(train_y, train_scores)

        test_score = float(model.predict_proba(np.vstack(target_test["embedding"].to_list()))[:, positive_index][0])
        rows.append(
            {
                "subject_id": str(target_test.iloc[0]["subject_id"]),
                "label": str(target_test.iloc[0]["label"]),
                "score_patients": test_score,
                "pred_label": "Patients" if test_score >= threshold else "Healthy",
                "threshold_fold": threshold,
                "model_type": model_type,
                "alignment_method": alignment_method,
            }
        )

    predictions = pd.DataFrame(rows)
    y_true = (predictions["label"] == "Patients").astype(int).to_numpy()
    y_score = predictions["score_patients"].to_numpy()
    y_pred = (predictions["pred_label"] == "Patients").astype(int).to_numpy()
    metrics = {
        "n_rows": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
    }
    global_threshold, global_metrics = best_threshold(y_true, y_score)
    metrics["best_threshold_global"] = global_threshold
    metrics["best_balanced_accuracy_global"] = global_metrics["balanced_accuracy"]
    metrics["best_recall_global"] = global_metrics["recall"]
    metrics["best_precision_global"] = global_metrics["precision"]
    return metrics, predictions
