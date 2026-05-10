from __future__ import annotations

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

    @torch.inference_mode()
    def extract_sequence(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        features_list, _ = self.model.extract_features(waveform.to(self.device))
        last_hidden = features_list[-1].squeeze(0).cpu()
        mask = torch.ones(last_hidden.size(0), dtype=torch.bool)
        return last_hidden, mask


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


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df):
        self.frame = manifest_df.reset_index(drop=True)
        self._labels = [LABEL_TO_IDX[str(lbl)] for lbl in self.frame["label"]]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        payload = torch.load(row["sequence_path"], map_location="cpu", weights_only=True)
        return {
            "sequence": payload["sequence"].float(),
            "mask": payload["mask"].bool(),
            "label": torch.tensor(self._labels[idx], dtype=torch.float32),
            "sample_id": str(row.get("sample_id", idx)),
            "subject_id": str(row.get("subject_id", idx)),
        }


def collate_sequences(batch: list[dict]) -> dict:
    max_len = max(item["sequence"].size(0) for item in batch)
    feat_dim = batch[0]["sequence"].size(1)
    B = len(batch)

    sequences = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    masks = torch.zeros(B, max_len, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.float32)
    sample_ids: list[str] = []
    subject_ids: list[str] = []

    for i, item in enumerate(batch):
        L = item["sequence"].size(0)
        sequences[i, :L] = item["sequence"]
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
