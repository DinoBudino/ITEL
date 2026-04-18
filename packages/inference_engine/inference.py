from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .embedding import HubertEmbeddingExtractor
from .exceptions import ValidationError
from .model_loader import ONNXClassifier, load_hparams, load_scaler
from .preprocessing import AudioPreprocessor


LOGGER = logging.getLogger(__name__)
LABELS = {0: "Healthy", 1: "Patients"}


@dataclass(slots=True)
class InferenceResult:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    timings_ms: dict[str, float]
    metadata: dict[str, Any]


class ITELInferenceService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.preprocessor = AudioPreprocessor(
            target_sr=self.settings.target_sr,
            hp_cutoff=self.settings.hp_cutoff,
            target_duration=self.settings.target_duration,
        )
        self.embedding_extractor = HubertEmbeddingExtractor(bundle_name=self.settings.hubert_bundle)
        self.classifier = ONNXClassifier(self.settings.final_model_path)
        self.scaler = None
        self.hparams = {}

    def warmup(self) -> None:
        self.scaler = load_scaler(self.settings.final_scaler_path)
        self.classifier.load()
        self.hparams = load_hparams(self.settings.hparams_path)
        LOGGER.info("Serving cargado con modelo %s", self.settings.model_version)

    def predict_from_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> InferenceResult:
        if len(audio_bytes) > self.settings.max_upload_size_bytes:
            raise ValidationError(
                f"El archivo excede el maximo permitido de {self.settings.max_upload_size_mb} MB."
            )
        if self.scaler is None:
            self.warmup()

        started = time.perf_counter()

        pre_start = time.perf_counter()
        processed = self.preprocessor.process_bytes(audio_bytes)
        pre_ms = (time.perf_counter() - pre_start) * 1000

        emb_start = time.perf_counter()
        embedding = self.embedding_extractor.extract_embedding(processed)
        emb_ms = (time.perf_counter() - emb_start) * 1000

        clf_start = time.perf_counter()
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
        logits = self.classifier.predict_logits(embedding_scaled)
        probabilities_arr = self._softmax(logits[0])
        pred_idx = int(np.argmax(probabilities_arr))
        clf_ms = (time.perf_counter() - clf_start) * 1000

        total_ms = (time.perf_counter() - started) * 1000
        probabilities = {LABELS[idx]: float(probabilities_arr[idx]) for idx in range(len(probabilities_arr))}

        return InferenceResult(
            predicted_label=LABELS[pred_idx],
            confidence=float(probabilities_arr[pred_idx]),
            probabilities=probabilities,
            model_version=self.settings.model_version,
            timings_ms={
                "preprocessing": round(pre_ms, 2),
                "embedding": round(emb_ms, 2),
                "classification": round(clf_ms, 2),
                "total": round(total_ms, 2),
            },
            metadata={
                "filename": filename,
                "sample_rate": processed.sample_rate,
                "duration_seconds": round(processed.duration_seconds, 3),
                "embedding_dim": int(embedding.shape[0]),
                "hparams": self.hparams,
            },
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)
