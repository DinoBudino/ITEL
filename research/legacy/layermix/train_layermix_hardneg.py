from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from research.src.config import CONFIG
from research.src.sequence_model import (AttentionBiLSTMClassifierV3,
                                          SequenceV3Dataset as SequenceDataset,
                                          collate_sequences)

HARD_NEGATIVES = {"H26", "H27", "H28", "H29"}


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
    B = embeddings.size(0)
    sim = (embeddings @ embeddings.T) / temperature
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0.0)
    n_pos = pos_mask.sum(dim=1); valid = n_pos > 0
    if valid.sum() == 0: return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)
    num = (exp_sim * pos_mask).sum(dim=1)
    return -(torch.log(num[valid] / (denom.squeeze(1)[valid] + 1e-8) + 1e-8)).mean()


def _fold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true_arr = np.array(y_true)
    m = {"accuracy": float(accuracy_score(y_true_arr, y_pred)),
         "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
         "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
         "f1": float(f1_score(y_true_arr, y_pred, zero_division=0))}
    m["auc"] = float(roc_auc_score(y_true_arr, y_prob)) if len(np.unique(y_true_arr)) > 1 else None
    return m


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_probs, all_labels, all_contexts, all_sids, all_subjs = [], [], [], [], []
    for batch in loader:
        out = model(batch["sequence"].to(device), batch["mask"].to(device), return_context=True)
        logits, ctx = out if len(out) == 2 else out[:2]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist()); all_labels.extend(batch["label"].numpy().tolist())
        all_contexts.append(ctx.cpu().numpy())
        all_sids.extend(batch["sample_id"]); all_subjs.extend(batch["subject_id"])
    emb = np.concatenate(all_contexts, axis=0)
    return pd.DataFrame({"sample_id": all_sids, "subject_id": all_subjs, "label": all_labels, "score_patients": all_probs}), emb


@torch.no_grad()
def _find_best_threshold(model, loader, device):
    pred_frame, _ = _predict(model, loader, device)
    yt, yp = pred_frame["label"].values, pred_frame["score_patients"].values
    best_thr, best_recall = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.05):
        m = _fold_metrics(yt, yp, threshold=round(thr, 2))
        if m["recall"] > best_recall: best_recall = m["recall"]; best_thr = round(thr, 2)
    return best_thr


class HardNegDataset(SequenceDataset):
    def __init__(self, manifest_df, oversample_factor=2, hard_negatives=None):
        super().__init__(manifest_df)
        self.hard_negatives = hard_negatives or set()
        if self.hard_negatives and oversample_factor > 1:
            hn_indices = [i for i, r in self.frame.iterrows() if str(r["subject_id"]) in self.hard_negatives]
            extra = [i for i in hn_indices for _ in range(oversample_factor - 1)]
            self._indices = list(range(len(self.frame))) + extra
        else:
            self._indices = list(range(len(self.frame)))

    def __len__(self): return len(self._indices)

    def __getitem__(self, idx):
        return super().__getitem__(self._indices[idx])


class HardNegWeightedLoss(nn.Module):
    def __init__(self, pos_weight=1.4, hn_weight=2.0, hard_negatives=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.hn_weight = hn_weight
        self.hard_negatives = hard_negatives or set()

    def forward(self, logits, targets, subject_ids):
        base_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor([self.pos_weight], device=logits.device), reduction="none")
        weights = torch.ones_like(base_loss)
        for i, sid in enumerate(subject_ids):
            if str(sid) in self.hard_negatives and targets[i].item() == 0.0:
                weights[i] = self.hn_weight
        return (base_loss * weights).mean()


def _compute_fold_metrics_at_thresholds(pred_frame):
    yt = pred_frame["label"].values
    ys = pred_frame["score_patients"].values
    best_ba, best_thr = -1.0, 0.5
    best_m = None
    for thr in np.arange(0.01, 1.0, 0.01):
        m = _fold_metrics(yt, ys, threshold=round(thr, 4))
        if m["recall"] >= 0.95 and m["balanced_accuracy"] > best_ba:
            best_ba = m["balanced_accuracy"]; best_thr = round(thr, 4); best_m = m
    return best_m if best_m else _fold_metrics(yt, ys, threshold=0.5), best_thr


def _train_fold(model, train_loader, val_loader, device, epochs, lr, weight_decay,
                pos_weight, lambda_contrastive, temperature, hn_weight, use_hn,
                checkpoint_path=None, hard_negatives=None):
    model.to(device)
    hn_loss_fn = HardNegWeightedLoss(pos_weight=pos_weight, hn_weight=hn_weight, hard_negatives=hard_negatives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_score = -1.0
    best_threshold = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, m, y = batch["sequence"].to(device), batch["mask"].to(device), batch["label"].to(device)
            out = model(x, m, return_context=True)
            logits, ctx = out if len(out) == 2 else out[:2]

            if use_hn:
                loss_bce = hn_loss_fn(logits, y, batch["subject_id"])
            else:
                loss_bce = criterion(logits, y)
            loss_ctr = supervised_contrastive_loss(ctx, y, temperature=temperature)
            loss = loss_bce + lambda_contrastive * loss_ctr

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        val_pred, _ = _predict(model, val_loader, device)
        best_val_m, val_thr = _compute_fold_metrics_at_thresholds(val_pred)
        score = best_val_m["balanced_accuracy"]
        if score > best_score:
            best_score = score; best_threshold = val_thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:02d} | loss={running_loss/max(len(train_loader),1):.4f} | "
              f"val_ba={best_val_m['balanced_accuracy']:.4f} val_rec={best_val_m['recall']:.4f} "
              f"val_prec={best_val_m['precision']:.4f} thr={val_thr:.2f}")

    model.load_state_dict(best_state)
    if checkpoint_path and best_state:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_threshold": float(best_threshold),
                     "model_config": {"input_dim": 768, "lstm_hidden": 256, "lstm_layers": 2,
                                      "dropout": 0.3, "fc_hidden": 128, "num_heads": 1,
                                      "use_layer_mix": True, "num_hubert_layers": 12,
                                      "embedding_noise_std": 0.0, "attention_entropy_lambda": 0.0}},
                    checkpoint_path)
    return model, best_threshold


def parse_args():
    p = argparse.ArgumentParser(description="V3 LayerMix with hard-negative focused training.")
    p.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_all_layers.csv"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=1.4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lambda-contrastive", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--device", default=None)
    p.add_argument("--oversample-factor", type=int, default=0, help="0=disabled, 2=2x oversample H26-H29")
    p.add_argument("--hn-loss-weight", type=float, default=0.0, help="0=disabled, 2.0=2x weight on HN")
    p.add_argument("--save-checkpoint-dir", default=str(CONFIG.paths.reports_dir / "checkpoints" / "v3_layermix_hardneg"))
    p.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV"))
    p.add_argument("--tag", default="v3_layermix_hardneg")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_hn = args.oversample_factor > 0 or args.hn_loss_weight > 0
    print(f"Device: {device}  Oversample: {args.oversample_factor}  HN-weight: {args.hn_loss_weight}")
    print(f"Pos-weight: {args.pos_weight}  Epochs: {args.epochs}  Folds: {args.n_splits}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"Samples: {len(manifest)}  Labels: {manifest['label'].value_counts().to_dict()}")
    hn_count = int(manifest["subject_id"].isin(HARD_NEGATIVES).sum())
    print(f"Hard negatives: {hn_count} samples from {HARD_NEGATIVES}")

    y = manifest["label"].values; groups = manifest["subject_id"].values
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    all_preds, fold_metrics = [], []
    ckpt_dir = Path(args.save_checkpoint_dir)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(manifest)), y, groups=groups), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        tr_df, vl_df = manifest.iloc[train_idx].copy(), manifest.iloc[val_idx].copy()
        print(f"  Train: {len(tr_df)}  Val: {len(vl_df)}")
        hn_train = int(tr_df["subject_id"].isin(HARD_NEGATIVES).sum())
        print(f"  HN in train: {hn_train}")

        if args.oversample_factor > 0:
            train_ds = HardNegDataset(tr_df, oversample_factor=args.oversample_factor, hard_negatives=HARD_NEGATIVES)
            print(f"  After oversample: {len(train_ds)}")
        else:
            train_ds = SequenceDataset(tr_df)

        tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_sequences, num_workers=0)
        vl_loader = DataLoader(SequenceDataset(vl_df), batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_sequences, num_workers=0)

        model = AttentionBiLSTMClassifierV3(use_layer_mix=True, num_hubert_layers=12, input_dim=768,
                                             lstm_hidden=256, lstm_layers=2, dropout=0.3, fc_hidden=128, num_heads=1)
        model, best_thr = _train_fold(model, tr_loader, vl_loader, device,
                                       epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                       pos_weight=args.pos_weight, lambda_contrastive=args.lambda_contrastive,
                                       temperature=args.temperature, hn_weight=args.hn_loss_weight, use_hn=use_hn,
                                       checkpoint_path=str(ckpt_dir / f"{args.tag}_fold{fold}_of_{args.n_splits}.pt"),
                                       hard_negatives=HARD_NEGATIVES)

        val_pred, val_emb = _predict(model, vl_loader, device)
        val_pred["fold"] = fold
        val_pred["pred_label"] = val_pred["score_patients"].apply(lambda s: "Patients" if s >= best_thr else "Healthy")
        vm = _fold_metrics(val_pred["label"], val_pred["score_patients"], threshold=best_thr)
        vm["fold"] = fold; vm["best_threshold"] = best_thr; vm["n_train"] = len(tr_df); vm["n_val"] = len(vl_df)
        fold_metrics.append(vm); all_preds.append(val_pred)
        print(f"  Fold: bal_acc={vm['balanced_accuracy']:.4f} rec={vm['recall']:.4f} prec={vm['precision']:.4f}")

    all_pred = pd.concat(all_preds, ignore_index=True)
    subj = all_pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj["pred_label"] = subj["score_patients"].apply(lambda s: "Patients" if s >= 0.5 else "Healthy")

    sample_m = _fold_metrics(all_pred["label"], all_pred["score_patients"], threshold=0.5)
    subject_m = _fold_metrics(subj["label"], subj["score_patients"], threshold=0.5)

    fold_summary = {}
    for mn in ["balanced_accuracy", "auc", "f1", "recall"]:
        vals = [float(r[mn]) for r in fold_metrics if r.get(mn) is not None]
        if vals: fold_summary[mn] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                      "min": float(np.min(vals)), "max": float(np.max(vals))}

    result = {"config": {"tag": args.tag, "oversample_factor": args.oversample_factor,
               "hn_loss_weight": args.hn_loss_weight, "pos_weight": args.pos_weight,
               "epochs": args.epochs, "n_splits": args.n_splits},
              "dataset": {"n_samples": int(len(all_pred)), "n_subjects": int(all_pred["subject_id"].nunique())},
              "sample_level": {"metrics": sample_m, "folds": fold_metrics, "fold_summary": fold_summary},
              "subject_level": {"metrics": subject_m}}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_pred.to_csv(out_dir / f"predictions_{args.tag}.csv", index=False)
    (out_dir / f"metrics_{args.tag}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    hn_subj = subj[subj["subject_id"].isin(HARD_NEGATIVES)]
    print(f"\n===== {args.tag} =====")
    print(f"Sample: bal_acc={sample_m['balanced_accuracy']:.4f} rec={sample_m['recall']:.4f} prec={sample_m['precision']:.4f}")
    print(f"Subject: bal_acc={subject_m['balanced_accuracy']:.4f} rec={subject_m['recall']:.4f} prec={subject_m['precision']:.4f}")
    for _, r in hn_subj.iterrows():
        print(f"  {r['subject_id']}: score={r['score_patients']:.4f} pred={r['pred_label']}")
    print(f"\nOutputs: {out_dir}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
