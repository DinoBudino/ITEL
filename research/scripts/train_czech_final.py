from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.scripts.train_optimized_v3 import supervised_contrastive_loss
from research.src.config import CONFIG
from research.src.multilingual_evaluation import best_threshold
from research.src.sequence_model import AttentionBiLSTMClassifierV3, SequenceV3Dataset, collate_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one final Czech LayerMix model after cross-validation selection.")
    parser.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_all_layers.csv"))
    parser.add_argument("--oof-predictions", default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_v3_layermix.csv"))
    parser.add_argument("--output", default=str(ROOT_DIR / "artifacts" / "czech-layermix-final.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.4)
    parser.add_argument("--lambda-contrastive", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _subject_oof_threshold(path: str | Path) -> float:
    predictions = pd.read_csv(path)
    required = {"subject_id", "label", "score_patients"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    subjects = predictions.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    labels = subjects["label"]
    y_true = (labels.astype(str).isin({"1", "1.0", "Patients"})).astype(int).to_numpy()
    threshold, _ = best_threshold(y_true, subjects["score_patients"].to_numpy())
    return float(threshold)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest = pd.read_csv(args.sequence_manifest)
    if manifest.empty:
        raise ValueError("The final training manifest is empty.")
    threshold = _subject_oof_threshold(args.oof_predictions)
    generator = torch.Generator().manual_seed(args.random_state)
    loader = DataLoader(
        SequenceV3Dataset(manifest),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
        generator=generator,
    )
    model = AttentionBiLSTMClassifierV3(use_layer_mix=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            labels = batch["label"].to(device)
            logits, context = model(
                batch["sequence"].to(device),
                batch["mask"].to(device),
                return_context=True,
            )
            bce = criterion(logits, labels)
            contrastive = supervised_contrastive_loss(context, labels, temperature=args.temperature)
            loss = bce + args.lambda_contrastive * contrastive
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={running_loss / max(len(loader), 1):.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "model_config": model.model_config,
            "best_threshold": threshold,
            "preprocessing": {
                "target_sr": CONFIG.audio.target_sr,
                "highpass_hz": CONFIG.audio.highpass_hz,
                "target_rms": CONFIG.audio.target_rms,
                "max_peak_after_rms": CONFIG.audio.max_peak_after_rms,
                "default_profile": CONFIG.audio.default_profile,
            },
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "pos_weight": args.pos_weight,
                "lambda_contrastive": args.lambda_contrastive,
                "temperature": args.temperature,
                "random_state": args.random_state,
            },
            "source_manifest": str(args.sequence_manifest),
            "threshold_source": str(args.oof_predictions),
        },
        output,
    )
    print(f"Final Czech checkpoint: {output}")
    print(f"Frozen subject-level threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
