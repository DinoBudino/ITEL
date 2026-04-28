from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from ..config import CONFIG
from .base import EmbeddingOutput


class XLSREncoder:
    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or CONFIG.encoder.model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
        self.model = Wav2Vec2Model.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.embedding_dim = self.hidden_size * 6

    def _pool_hidden_states(self, hidden_states: np.ndarray) -> np.ndarray:
        mean = hidden_states.mean(axis=0)
        std = hidden_states.std(axis=0)
        q1 = hidden_states[: max(1, len(hidden_states) // 4)].mean(axis=0)
        q4 = hidden_states[-max(1, len(hidden_states) // 4) :].mean(axis=0)
        delta = hidden_states[-1] - hidden_states[0]
        slope = np.linspace(-1.0, 1.0, num=len(hidden_states), dtype=np.float32) @ hidden_states
        slope = slope / max(len(hidden_states), 1)
        return np.concatenate([mean, std, q1, q4, delta, slope], axis=0).astype(np.float32)

    @torch.inference_mode()
    def encode(self, samples: np.ndarray, sr: int) -> EmbeddingOutput:
        inputs = self.feature_extractor(samples, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[0].detach().cpu().numpy()
        embedding = self._pool_hidden_states(hidden_states)
        return EmbeddingOutput(
            embedding=embedding,
            hidden_size=self.hidden_size,
            embedding_dim=int(embedding.shape[0]),
            model_id=self.model_id,
            pooling_recipe=CONFIG.encoder.pooling_recipe,
        )
