from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .exceptions import ModelLoadError


class ONNXClassifier:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._session = None

    def load(self):
        import onnxruntime as ort

        if self._session is not None:
            return self._session
        if not self.model_path.exists():
            raise ModelLoadError(f"No se encontro el modelo ONNX en {self.model_path}")

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    def predict_logits(self, embedding_scaled: np.ndarray) -> np.ndarray:
        session = self.load()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: embedding_scaled.astype(np.float32)})
        return outputs[0]


def load_scaler(scaler_path: Path):
    if not scaler_path.exists():
        raise ModelLoadError(f"No se encontro el scaler en {scaler_path}")
    return joblib.load(scaler_path)


def load_hparams(hparams_path: Path) -> dict:
    if not hparams_path.exists():
        return {}
    with hparams_path.open("r", encoding="utf-8") as file:
        return json.load(file)
