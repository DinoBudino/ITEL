from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:  # pragma: no cover - optional dependency for analysis-only runs
    mlflow = None

from research.src.config import CONFIG


DEFAULT_EXPERIMENT_NAME = "itel-research"
DEFAULT_TRACKING_URI = (CONFIG.paths.reports_dir / "mlruns").resolve().as_uri()


def configure_tracking(tracking_uri: str | None, experiment_name: str = DEFAULT_EXPERIMENT_NAME) -> str:
    resolved_tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
    if tracking_uri is None:
        (CONFIG.paths.reports_dir / "mlruns").mkdir(parents=True, exist_ok=True)
    if mlflow is None:
        raise ModuleNotFoundError("mlflow is required for tracking-enabled workflows.")
    mlflow.set_tracking_uri(resolved_tracking_uri)
    mlflow.set_experiment(experiment_name)
    return resolved_tracking_uri


def enable_training_autolog() -> None:
    if mlflow is None:
        raise ModuleNotFoundError("mlflow is required for tracking-enabled workflows.")
    mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=True, silent=True)


def log_params(params: Mapping[str, Any]) -> None:
    if mlflow is None:
        return
    sanitized = {key: _stringify_value(value) for key, value in params.items() if value is not None}
    if sanitized:
        mlflow.log_params(sanitized)


def log_metrics(metrics: Mapping[str, Any], prefix: str = "") -> None:
    if mlflow is None:
        return
    flat_metrics = _flatten_mapping(metrics, prefix=prefix)
    numeric_metrics = {
        key: float(value)
        for key, value in flat_metrics.items()
        if isinstance(value, Real) and not isinstance(value, bool)
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def log_dict(name: str, payload: Mapping[str, Any]) -> None:
    if mlflow is None:
        return
    mlflow.log_dict(dict(payload), name)


def log_text(name: str, text: str) -> None:
    if mlflow is None:
        return
    mlflow.log_text(text, name)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def describe_model(model: Any) -> dict[str, Any]:
    description: dict[str, Any] = {
        "model_class": type(model).__name__,
    }

    if hasattr(model, "named_steps"):
        description["pipeline_steps"] = list(model.named_steps.keys())
        clf = model.named_steps.get("clf")
        if clf is not None:
            description["classifier_class"] = type(clf).__name__
            if hasattr(clf, "classes_"):
                description["classes"] = [str(label) for label in clf.classes_]

    if hasattr(model, "classes_"):
        description["classes"] = [str(label) for label in model.classes_]

    return description


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=f"{full_key}."))
            continue
        flattened[full_key] = value
    return flattened


def _stringify_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)
