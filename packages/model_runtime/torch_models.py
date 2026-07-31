from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMix(nn.Module):
    def __init__(self, num_layers: int = 12):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.weights[: layers.size(1)], dim=0).view(1, -1, 1, 1)
        return (layers * weights).sum(dim=1)


class AttentionBiLSTMClassifierV3(nn.Module):
    def __init__(self, input_dim=768, lstm_hidden=256, lstm_layers=2, dropout=0.3, fc_hidden=128, num_heads=1, use_layer_mix=True, num_hubert_layers=12, **_):
        super().__init__()
        if num_heads != 1:
            raise ValueError("The promoted Czech runtime currently supports one attention head.")
        self.layer_mix = ScalarMix(num_hubert_layers) if use_layer_mix else None
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.attn = nn.Sequential(nn.Linear(lstm_hidden * 2, lstm_hidden), nn.Tanh(), nn.Linear(lstm_hidden, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(lstm_hidden * 2), nn.Dropout(dropout), nn.Linear(lstm_hidden * 2, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.layer_mix is not None:
            x = self.layer_mix(x)
        output, _ = self.lstm(self.input_norm(x))
        scores = self.attn(output).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        return self.classifier(context).squeeze(-1)


@dataclass(slots=True)
class LoadedTorchModel:
    module: nn.Module
    device: torch.device
