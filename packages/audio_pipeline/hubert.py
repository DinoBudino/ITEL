from __future__ import annotations

import logging
import threading

import numpy as np

from .preprocessing import ProcessedAudio


LOGGER = logging.getLogger(__name__)


class HubertFeatureExtractor:
    def __init__(self, bundle_name: str = "HUBERT_BASE", device: str | None = None):
        self.bundle_name = bundle_name
        self.requested_device = device
        self.device = None
        self._model = None
        self._lock = threading.RLock()

    def preload(self) -> None:
        self._load_model()

    def warmup(self, sample_rate: int = 16000) -> None:
        self.preload()
        audio = ProcessedAudio(
            samples=np.zeros(sample_rate // 2, dtype=np.float32),
            sample_rate=sample_rate,
            source_duration_seconds=0.5,
        )
        self.extract_last_layer(audio)

    def extract_last_layer(self, audio: ProcessedAudio) -> np.ndarray:
        layers = self._extract_layers(audio)
        return layers[-1]

    def extract_all_layers(self, audio: ProcessedAudio) -> np.ndarray:
        return np.stack(self._extract_layers(audio), axis=0)

    def _extract_layers(self, audio: ProcessedAudio) -> list[np.ndarray]:
        import torch

        with self._lock:
            model = self._load_model()
            waveform = torch.from_numpy(audio.samples).float().unsqueeze(0).to(self.device)
            with torch.inference_mode():
                feature_list, _ = model.extract_features(waveform)
        layers = [features.squeeze(0).cpu().numpy().astype(np.float32) for features in feature_list[-12:]]
        if not layers or layers[-1].shape[0] < 1:
            raise RuntimeError("HuBERT returned no features.")
        return layers

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            import torch
            import torchaudio

            bundle = getattr(torchaudio.pipelines, self.bundle_name)
            self.device = torch.device(self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._model = bundle.get_model().eval().to(self.device)
            LOGGER.info("Loaded %s on %s", self.bundle_name, self.device)
            return self._model
