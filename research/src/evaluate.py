from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, roc_auc_score


def binary_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n_rows": 0,
            "n_unique_labels": 0,
            "balanced_accuracy": None,
            "auc": None,
            "f1": None,
            "classification_report": "",
        }

    metrics: dict[str, Any] = {
        "n_rows": int(len(y_true)),
        "n_unique_labels": int(pd.Series(y_true).nunique()),
    }

    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except ValueError:
        metrics["balanced_accuracy"] = None

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auc"] = None

    metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted"))

    metrics["classification_report"] = classification_report(y_true, y_pred)
    return metrics
