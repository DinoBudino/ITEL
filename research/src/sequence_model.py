from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HubertSequenceEncoder:
    def __init__(self, device: str | None = None):
        import torchaudio

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        bundle = getattr(torchaudio.pipelines, "HUBERT_BASE")
        self.model = bundle.get_model().to(self.device)
        self.model.eval()
        self.sample_rate = bundle.sample_rate
        self.hidden_size = 768
        self.num_layers = 12

    @torch.inference_mode()
    def extract_sequence(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        features_list, _ = self.model.extract_features(waveform.to(self.device))
        last_hidden = features_list[-1].squeeze(0).cpu()
        mask = torch.ones(last_hidden.size(0), dtype=torch.bool)
        return last_hidden, mask

    @torch.inference_mode()
    def extract_all_layers(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        features_list, _ = self.model.extract_features(waveform.to(self.device))
        num_total = len(features_list)
        transformer_layers = features_list[-12:] if num_total >= 12 else features_list
        if len(transformer_layers) != 12:
            transformer_layers = features_list[1:] if num_total == 13 else features_list
        all_layers = torch.stack([lyr.squeeze(0).cpu() for lyr in transformer_layers], dim=0)
        mask = torch.ones(all_layers.size(1), dtype=torch.bool)
        return all_layers, mask


class ScalarMix(nn.Module):
    def __init__(self, num_layers: int = 12):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:
        actual_layers = layers.size(1) if layers.dim() == 4 else layers.size(0)
        if actual_layers == self.num_layers:
            w = F.softmax(self.weights, dim=0)
        else:
            delta = self.num_layers - actual_layers
            if delta > 0:
                padded = F.pad(self.weights, (0, abs(delta)), value=float("-inf"))
                w = F.softmax(padded, dim=0)
                w = w[:actual_layers]
            else:
                w = F.softmax(self.weights[:actual_layers], dim=0)
        if layers.dim() == 4:
            w = w.view(1, -1, 1, 1)
        else:
            w = w.view(-1, 1, 1)
        return (layers * w).sum(dim=-3 if layers.dim() == 4 else 0)


class AttentionBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        num_heads: int = 1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.lstm_dim = lstm_hidden * 2

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        if num_heads == 1:
            self.attn = nn.Sequential(
                nn.Linear(self.lstm_dim, lstm_hidden),
                nn.Tanh(),
                nn.Linear(lstm_hidden, 1),
            )
            attn_out_dim = self.lstm_dim
        else:
            self.attn_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.lstm_dim, lstm_hidden),
                    nn.Tanh(),
                    nn.Linear(lstm_hidden, 1),
                )
                for _ in range(num_heads)
            ])
            attn_out_dim = self.lstm_dim * num_heads

        self.classifier = nn.Sequential(
            nn.LayerNorm(attn_out_dim),
            nn.Dropout(dropout),
            nn.Linear(attn_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attention: bool = False, return_context: bool = False):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)

        if self.num_heads == 1:
            attn_scores = self.attn(lstm_out).squeeze(-1)
            attn_weights = self._masked_softmax(attn_scores, mask)
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        else:
            all_weights = []
            all_contexts = []
            for h in range(self.num_heads):
                scores = self.attn_heads[h](lstm_out).squeeze(-1)
                w = self._masked_softmax(scores, mask)
                all_weights.append(w)
                c = torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1)
                all_contexts.append(c)
            attn_weights = torch.stack(all_weights, dim=1)
            context = torch.cat(all_contexts, dim=1)

        logits = self.classifier(context).squeeze(-1)
        if return_context:
            if return_attention:
                return logits, context, attn_weights
            return logits, context
        if return_attention:
            return logits, attn_weights
        return logits

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights


LABEL_TO_IDX = {"Healthy": 0, "Patients": 1}


class AttentionBiLSTMClassifierV3(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 128,
        num_heads: int = 1,
        use_layer_mix: bool = False,
        num_hubert_layers: int = 12,
        embedding_noise_std: float = 0.0,
        attention_entropy_lambda: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.lstm_dim = lstm_hidden * 2
        self.use_layer_mix = use_layer_mix
        self.embedding_noise_std = embedding_noise_std
        self.attention_entropy_lambda = attention_entropy_lambda

        if use_layer_mix:
            self.layer_mix = ScalarMix(num_layers=num_hubert_layers)

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        if num_heads == 1:
            self.attn = nn.Sequential(
                nn.Linear(self.lstm_dim, lstm_hidden),
                nn.Tanh(),
                nn.Linear(lstm_hidden, 1),
            )
            attn_out_dim = self.lstm_dim
        else:
            self.attn_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.lstm_dim, lstm_hidden),
                    nn.Tanh(),
                    nn.Linear(lstm_hidden, 1),
                )
                for _ in range(num_heads)
            ])
            attn_out_dim = self.lstm_dim * num_heads

        self.classifier = nn.Sequential(
            nn.LayerNorm(attn_out_dim),
            nn.Dropout(dropout),
            nn.Linear(attn_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def _apply_layer_mix(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_layer_mix or x.dim() != 4:
            return x
        return self.layer_mix(x)

    def _apply_embedding_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_noise_std <= 0.0 or not self.training:
            return x
        noise = torch.randn_like(x) * self.embedding_noise_std
        return x + noise

    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if attn_weights.dim() == 2:
            ent = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=1).mean()
        elif attn_weights.dim() == 3:
            ent = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=2).mean()
        else:
            ent = torch.tensor(0.0, device=attn_weights.device)
        return ent

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False, return_context: bool = False):
        x = self._apply_layer_mix(x)
        x = self._apply_embedding_noise(x)
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)

        entropy_loss = torch.tensor(0.0, device=x.device)

        if self.num_heads == 1:
            attn_scores = self.attn(lstm_out).squeeze(-1)
            attn_weights = self._masked_softmax(attn_scores, mask)
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
            if self.attention_entropy_lambda > 0.0 and self.training:
                entropy_loss = self._compute_attention_entropy(attn_weights)
        else:
            all_weights = []
            all_contexts = []
            for h in range(self.num_heads):
                scores = self.attn_heads[h](lstm_out).squeeze(-1)
                w = self._masked_softmax(scores, mask)
                all_weights.append(w)
                c = torch.bmm(w.unsqueeze(1), lstm_out).squeeze(1)
                all_contexts.append(c)
            attn_weights = torch.stack(all_weights, dim=1)
            context = torch.cat(all_contexts, dim=1)
            if self.attention_entropy_lambda > 0.0 and self.training:
                entropy_loss = self._compute_attention_entropy(
                    attn_weights.mean(dim=1)
                )

        logits = self.classifier(context).squeeze(-1)

        if self.attention_entropy_lambda > 0.0 and self.training:
            if return_context:
                if return_attention:
                    return logits, context, attn_weights, entropy_loss
                return logits, context, entropy_loss
            if return_attention:
                return logits, attn_weights, entropy_loss
            return logits, entropy_loss

        if return_context:
            if return_attention:
                return logits, context, attn_weights
            return logits, context
        if return_attention:
            return logits, attn_weights
        return logits

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return weights


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df):
        self.frame = manifest_df.reset_index(drop=True)
        self._labels = [LABEL_TO_IDX[str(lbl)] for lbl in self.frame["label"]]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        payload = torch.load(row["sequence_path"], map_location="cpu", weights_only=True)
        seq = payload["sequence"].float()
        mask = payload["mask"].bool()
        return {
            "sequence": seq,
            "mask": mask,
            "label": torch.tensor(self._labels[idx], dtype=torch.float32),
            "sample_id": str(row.get("sample_id", idx)),
            "subject_id": str(row.get("subject_id", idx)),
        }


class SequenceV3Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df):
        self.frame = manifest_df.reset_index(drop=True)
        self._labels = [LABEL_TO_IDX[str(lbl)] for lbl in self.frame["label"]]
        self._has_layers: bool | None = None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        payload = torch.load(row["sequence_path"], map_location="cpu", weights_only=True)
        seq = payload["sequence"].float()
        mask = payload["mask"].bool()
        if self._has_layers is None:
            self._has_layers = seq.dim() == 3
        return {
            "sequence": seq,
            "mask": mask,
            "has_layers": seq.dim() == 3,
            "label": torch.tensor(self._labels[idx], dtype=torch.float32),
            "sample_id": str(row.get("sample_id", idx)),
            "subject_id": str(row.get("subject_id", idx)),
        }


def collate_sequences(batch: list[dict]) -> dict:
    max_len = max(item["sequence"].size(-1) if item["sequence"].dim() == 3
                  else item["sequence"].size(0) for item in batch)
    has_layers = any(item["sequence"].dim() == 3 for item in batch)
    feat_dim = batch[0]["sequence"].size(-1)
    B = len(batch)

    if has_layers:
        num_layers = next(item["sequence"].size(0) for item in batch if item["sequence"].dim() == 3)
        sequences = torch.zeros(B, num_layers, max_len, feat_dim, dtype=torch.float32)
    else:
        sequences = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)

    masks = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.float32)
    sample_ids: list[str] = []
    subject_ids: list[str] = []

    for i, item in enumerate(batch):
        s = item["sequence"]
        if s.dim() == 3:
            L = s.size(1)
            sequences[i, :, :L] = s
        else:
            L = s.size(0)
            sequences[i, :L] = s
        masks[i, :L] = item["mask"]
        labels[i] = item["label"]
        sample_ids.append(item["sample_id"])
        subject_ids.append(item["subject_id"])

    return {
        "sequence": sequences,
        "mask": masks,
        "label": labels,
        "sample_id": sample_ids,
        "subject_id": subject_ids,
    }
