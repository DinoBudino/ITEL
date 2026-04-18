from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .exceptions import ModelLoadError
from .preprocessing import ProcessedAudio


LOGGER = logging.getLogger(__name__)


class HubertEmbeddingExtractor:
    def __init__(self, bundle_name: str = "HUBERT_BASE"):
        self.bundle_name = bundle_name
        self.device = None
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
            import torchaudio

            bundle = getattr(torchaudio.pipelines, self.bundle_name)
            model = bundle.get_model()
        except Exception as exc:
            raise ModelLoadError(f"No se pudo cargar HuBERT con bundle {self.bundle_name}.") from exc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(self.device)
        self._model = model
        LOGGER.info("HuBERT cargado en %s", self.device)
        return self._model

    def extract_embedding(self, processed_audio: ProcessedAudio) -> np.ndarray:
        import torch

        model = self._load_model()
        waveform = torch.tensor(processed_audio.samples, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features, _ = model(waveform)

        features = features.squeeze(0).cpu()
        time_steps = features.shape[0]
        if time_steps < 1:
            raise ModelLoadError("HuBERT no devolvio features para el audio.")

        mean_emb = features.mean(dim=0)
        std_emb = features.std(dim=0)
        first_25 = features[: max(1, time_steps // 4)].mean(dim=0)
        last_25 = features[-max(1, time_steps // 4) :].mean(dim=0)
        delta = first_25 - last_25
        slope_emb = self._compute_temporal_correlation(features)

        embedding = torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope_emb]).numpy()
        return embedding.astype(np.float32)

    @staticmethod
    def _compute_temporal_correlation(features: Any):
        import torch

        time_steps, dims = features.shape
        if time_steps < 2:
            return torch.zeros(dims, dtype=torch.float32)

        x = torch.arange(time_steps, dtype=torch.float32)
        x_centered = x - x.mean()
        x_norm = torch.sqrt((x_centered**2).sum())
        slopes = torch.zeros(dims, dtype=torch.float32)

        for dim in range(dims):
            y = features[:, dim].float()
            y_centered = y - y.mean()
            y_norm = torch.sqrt((y_centered**2).sum())
            if x_norm == 0 or y_norm == 0:
                slopes[dim] = 0.0
            else:
                slopes[dim] = torch.dot(x_centered, y_centered) / (x_norm * y_norm)

        return slopes
