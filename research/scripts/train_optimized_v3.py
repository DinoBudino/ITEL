from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from research.src.config import CONFIG
from research.src.sequence_model import (AttentionBiLSTMClassifierV3,
                                          SequenceV3Dataset as SequenceDataset,
                                          collate_sequences)


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
    B = embeddings.size(0)
    sim = (embeddings @ embeddings.T) / temperature

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0.0)

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
    num = (exp_sim * pos_mask).sum(dim=1)

    loss_per_sample = -torch.log(num[valid] / (denom.squeeze(1)[valid] + 1e-8) + 1e-8)
    return loss_per_sample.mean()


def _fold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true_arr = np.array(y_true)
    m = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }
    m["auc"] = float(roc_auc_score(y_true_arr, y_prob)) if len(np.unique(y_true_arr)) > 1 else None
    return m


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    all_contexts = []
    all_sample_ids, all_subject_ids = [], []
    for batch in loader:
        has_entropy = model.attention_entropy_lambda > 0.0
        out = model(batch["sequence"].to(device), batch["mask"].to(device), return_context=True)
        if has_entropy and len(out) == 3:
            logits, ctx, _ent = out
        else:
            logits, ctx = out
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_contexts.append(ctx.cpu().numpy())
        all_sample_ids.extend(batch["sample_id"])
        all_subject_ids.extend(batch["subject_id"])
    emb = np.concatenate(all_contexts, axis=0)
    return pd.DataFrame({
        "sample_id": all_sample_ids, "subject_id": all_subject_ids,
        "label": all_labels, "score_patients": all_probs,
    }), emb


@torch.no_grad()
def _find_best_threshold(model, loader, device):
    pred_frame, _ = _predict(model, loader, device)
    y_true = pred_frame["label"].values
    y_prob = pred_frame["score_patients"].values
    best_thr, best_recall = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.05):
        thr = float(round(thr, 2))
        m = _fold_metrics(y_true, y_prob, threshold=thr)
        if m["recall"] > best_recall:
            best_recall = m["recall"]
            best_thr = thr
    return best_thr


def _train_fold(model, train_loader, val_loader, device, epochs, lr, weight_decay,
                pos_weight, lambda_contrastive, temperature, checkpoint_path=None):
    model.to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_recall = -1.0
    best_threshold = 0.5
    has_entropy = model.attention_entropy_lambda > 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_bce = 0.0
        running_ctr = 0.0
        running_ent = 0.0
        for batch in train_loader:
            x = batch["sequence"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device)

            out = model(x, m, return_context=True)
            if has_entropy and len(out) == 3:
                logits, ctx, entropy_loss = out
            else:
                logits, ctx = out
                entropy_loss = torch.tensor(0.0, device=device)

            loss_bce = criterion_bce(logits, y)
            loss_ctr = supervised_contrastive_loss(ctx, y, temperature=temperature)
            loss = loss_bce + lambda_contrastive * loss_ctr
            if has_entropy:
                loss = loss + model.attention_entropy_lambda * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_bce += loss_bce.item()
            running_ctr += loss_ctr.item()
            running_ent += entropy_loss.item() if has_entropy else 0.0

        threshold = _find_best_threshold(model, val_loader, device)
        val_pred, _ = _predict(model, val_loader, device)
        val_metrics = _fold_metrics(
            val_pred["label"].values, val_pred["score_patients"].values, threshold=threshold,
        )
        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        n = max(len(train_loader), 1)
        ent_str = f" ENT={running_ent/n:.4f}" if has_entropy else ""
        print(
            f"  Epoch {epoch:02d} | BCE={running_bce/n:.4f} CTR={running_ctr/n:.4f}{ent_str} | "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']}"
        )

    model.load_state_dict(best_state)
    if checkpoint_path and best_state is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "model_config": {
                "input_dim": 768, "lstm_hidden": 256, "lstm_layers": 2,
                "dropout": 0.3, "fc_hidden": 128, "num_heads": 1,
                "use_layer_mix": model.use_layer_mix,
                "num_hubert_layers": 12,
                "embedding_noise_std": model.embedding_noise_std,
                "attention_entropy_lambda": model.attention_entropy_lambda,
            },
            "best_threshold": float(best_threshold),
            "best_recall": float(best_recall),
            "training_config": {
                "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                "pos_weight": pos_weight, "lambda_contrastive": lambda_contrastive,
                "temperature": temperature,
            },
            "preprocessing": {
                "target_sr": CONFIG.audio.target_sr,
                "highpass_hz": CONFIG.audio.highpass_hz,
                "target_rms": CONFIG.audio.target_rms,
                "max_peak_after_rms": CONFIG.audio.max_peak_after_rms,
                "default_profile": CONFIG.audio.default_profile,
            },
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    return model, best_threshold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V3: HuBERT + BiLSTM + Attention with layer-mix, noise, sparse-attn.")

    g_data = p.add_argument_group("Data")
    g_data.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_cleaned.csv"))
    g_data.add_argument("--limit", type=int, default=None, help="Limit to first N samples (quick runs).")

    g_cv = p.add_argument_group("Cross-Validation")
    g_cv.add_argument("--n-splits", type=int, default=10)
    g_cv.add_argument("--random-state", type=int, default=42)

    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=30)
    g_train.add_argument("--lr", type=float, default=1e-3)
    g_train.add_argument("--weight-decay", type=float, default=1e-4)
    g_train.add_argument("--pos-weight", type=float, default=1.4)
    g_train.add_argument("--batch-size", type=int, default=16)
    g_train.add_argument("--lambda-contrastive", type=float, default=0.1)
    g_train.add_argument("--temperature", type=float, default=0.07)

    g_model = p.add_argument_group("Model Architecture")
    g_model.add_argument("--lstm-hidden", type=int, default=256)
    g_model.add_argument("--lstm-layers", type=int, default=2)
    g_model.add_argument("--dropout", type=float, default=0.3)
    g_model.add_argument("--fc-hidden", type=int, default=128)
    g_model.add_argument("--num-heads", type=int, default=1)

    g_feat = p.add_argument_group("Feature Fusion (Mejora A)")
    g_feat.add_argument("--use-layer-mix", action="store_true", default=False,
                        help="Use ScalarMix weighted sum of all 12 HuBERT layers.")

    g_noise = p.add_argument_group("Embedding Noise (Mejora B)")
    g_noise.add_argument("--embedding-noise-std", type=float, default=0.0,
                         help="Std of Gaussian noise injected into embeddings during training.")

    g_attn = p.add_argument_group("Sparse Attention (Mejora C)")
    g_attn.add_argument("--attention-entropy-lambda", type=float, default=0.0,
                        help="Weight for attention entropy penalty. 0 = disabled.")

    g_out = p.add_argument_group("Output")
    g_out.add_argument("--device", default=None)
    g_out.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV"))
    g_out.add_argument("--save-checkpoint-dir", default=str(CONFIG.paths.reports_dir / "checkpoints" / "v3"))

    return p.parse_args()


def _build_tag(args) -> str:
    parts = ["v3"]
    if args.use_layer_mix:
        parts.append("layermix")
    if args.embedding_noise_std > 0:
        parts.append(f"noise{args.embedding_noise_std}")
    if args.attention_entropy_lambda > 0:
        parts.append(f"sparse{args.attention_entropy_lambda}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = _build_tag(args)
    print(f"=== V3 Training: {tag} ===")
    print(f"Device: {device}")
    print(f"Layer mix: {args.use_layer_mix}  Noise std: {args.embedding_noise_std}  "
          f"Attn entropy lambda: {args.attention_entropy_lambda}")
    print(f"CV folds: {args.n_splits}  Epochs: {args.epochs}  LR: {args.lr}")

    manifest = pd.read_csv(args.sequence_manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    print(f"Loaded {len(manifest)} samples  |  Label dist: {manifest.label.value_counts().to_dict()}")

    y = manifest["label"].values
    groups = manifest["subject_id"].values
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    checkpoint_dir = Path(args.save_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    all_embs = []
    fold_metrics_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(manifest)), y, groups=groups), start=1):
        print(f"\n===== Fold {fold_idx}/{args.n_splits} =====")
        train_df = manifest.iloc[train_idx].copy()
        val_df = manifest.iloc[val_idx].copy()
        print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

        train_loader = DataLoader(SequenceDataset(train_df), batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_sequences, num_workers=0)
        val_loader = DataLoader(SequenceDataset(val_df), batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_sequences, num_workers=0)

        model = AttentionBiLSTMClassifierV3(
            input_dim=768,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            fc_hidden=args.fc_hidden,
            num_heads=args.num_heads,
            use_layer_mix=args.use_layer_mix,
            num_hubert_layers=12,
            embedding_noise_std=args.embedding_noise_std,
            attention_entropy_lambda=args.attention_entropy_lambda,
        )

        ckpt_path = str(checkpoint_dir / f"{tag}_fold{fold_idx}_of_{args.n_splits}.pt")
        model, best_thr = _train_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            pos_weight=args.pos_weight,
            lambda_contrastive=args.lambda_contrastive,
            temperature=args.temperature,
            checkpoint_path=ckpt_path,
        )
        print(f"  Best threshold: {best_thr:.2f}")

        val_pred, val_emb = _predict(model, val_loader, device)
        val_pred["fold"] = fold_idx
        val_pred["pred_label"] = val_pred["score_patients"].apply(
            lambda s: "Patients" if s >= best_thr else "Healthy")

        val_metrics = _fold_metrics(val_pred["label"].values, val_pred["score_patients"].values, threshold=best_thr)
        val_metrics["fold"] = fold_idx
        val_metrics["best_threshold"] = best_thr
        val_metrics["n_train"] = len(train_df)
        val_metrics["n_val"] = len(val_df)
        fold_metrics_list.append(val_metrics)
        all_preds.append(val_pred)

        emb_df = pd.DataFrame(val_emb, index=val_pred.index)
        emb_df["sample_id"] = val_pred["sample_id"].values
        emb_df["subject_id"] = val_pred["subject_id"].values
        emb_df["label"] = val_pred["label"].values
        emb_df["fold"] = fold_idx
        all_embs.append(emb_df)

        print(f"  Fold metrics: RECALL={val_metrics['recall']:.4f} (PRIORITY) "
              f"bal_acc={val_metrics['balanced_accuracy']:.4f} f1={val_metrics['f1']:.4f} "
              f"auc={val_metrics['auc']}")

    all_pred = pd.concat(all_preds, ignore_index=True)
    all_emb = pd.concat(all_embs, ignore_index=True)

    sample_m = _fold_metrics(all_pred["label"].values, all_pred["score_patients"].values, threshold=0.5)

    subj = all_pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj["pred_label"] = subj["score_patients"].apply(lambda s: "Patients" if s >= 0.5 else "Healthy")
    subject_m = _fold_metrics(subj["label"].values, subj["score_patients"].values, threshold=0.5)

    fold_summary = {}
    for m_name in ["balanced_accuracy", "auc", "f1", "recall"]:
        vals = [float(r[m_name]) for r in fold_metrics_list if r.get(m_name) is not None]
        if vals:
            fold_summary[m_name] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            }

    result = {
        "config": {
            "arch": "V3_HuBERT_BASE_AttentionBiLSTM",
            "tag": tag,
            "use_layer_mix": args.use_layer_mix,
            "embedding_noise_std": args.embedding_noise_std,
            "attention_entropy_lambda": args.attention_entropy_lambda,
            "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
            "dropout": args.dropout, "fc_hidden": args.fc_hidden, "num_heads": args.num_heads,
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "pos_weight": args.pos_weight, "lambda_contrastive": args.lambda_contrastive,
            "temperature": args.temperature,
            "n_splits": args.n_splits, "random_state": args.random_state,
        },
        "dataset": {
            "n_samples": int(len(all_pred)),
            "n_subjects": int(all_pred["subject_id"].nunique()),
            "label_distribution": manifest["label"].value_counts().to_dict(),
        },
        "sample_level": {"metrics": sample_m, "folds": fold_metrics_list, "fold_summary": fold_summary},
        "subject_level": {"metrics": subject_m},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_out = out_dir / f"predictions_{tag}.csv"
    emb_out = out_dir / f"embeddings_{tag}.parquet"
    metrics_out = out_dir / f"metrics_{tag}.json"

    all_pred.to_csv(pred_out, index=False)
    all_emb.to_parquet(emb_out)
    metrics_out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    metadata = {
        "arch": "V3_HuBERT_BASE -> AttentionBiLSTM + SupCon",
        "tag": tag,
        "improvements": {
            "layer_mix": args.use_layer_mix,
            "embedding_noise_std": args.embedding_noise_std,
            "attention_entropy_lambda": args.attention_entropy_lambda,
        },
        "preprocessing": {
            "target_sr": CONFIG.audio.target_sr,
            "highpass_hz": CONFIG.audio.highpass_hz,
            "target_rms": CONFIG.audio.target_rms,
            "max_peak_after_rms": CONFIG.audio.max_peak_after_rms,
            "default_profile": CONFIG.audio.default_profile,
        },
        "model_config": {"input_dim": 768, "lstm_hidden": 256, "lstm_layers": 2, "dropout": 0.3, "fc_hidden": 128, "num_heads": 1},
        "training_config": {"lambda_contrastive": args.lambda_contrastive, "temperature": args.temperature, "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay, "pos_weight": args.pos_weight},
        "source_manifest": args.sequence_manifest,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"\n===== V3 RESULTS ({tag}) =====")
    print(f"Sample - RECALL={sample_m['recall']:.4f} (PRIORITY) bal_acc={sample_m['balanced_accuracy']:.4f} "
          f"f1={sample_m['f1']:.4f} auc={sample_m['auc']}")
    print(f"Subject - RECALL={subject_m['recall']:.4f} (PRIORITY) bal_acc={subject_m['balanced_accuracy']:.4f} "
          f"f1={subject_m['f1']:.4f} auc={subject_m['auc']}")

    for m_name, v in fold_summary.items():
        print(f"  Fold {m_name}: {v['mean']:.4f} +/- {v['std']:.4f}")

    fp_n = int((all_pred["label"] == 0.0).sum() - (all_pred["pred_label"] == "Healthy").sum())
    healthy_n = int((all_pred["label"] == 0.0).sum())
    print(f"FPs: {fp_n}/{healthy_n} = {fp_n/max(healthy_n,1)*100:.1f}%")

    print(f"\nPredictions: {pred_out}")
    print(f"Embeddings:  {emb_out}")
    print(f"Metrics:     {metrics_out}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
