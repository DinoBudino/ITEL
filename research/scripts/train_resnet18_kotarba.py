from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from research.src.config import CONFIG


class SpecDataset(Dataset):
    def __init__(self, manifest_df):
        self.frame = manifest_df.reset_index(drop=True)
        self._labels = [1.0 if str(lbl) == "Patients" else 0.0 for lbl in self.frame["label"]]

    def __len__(self): return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        spec = np.load(row["feature_path"]).astype(np.float32)
        spec = torch.tensor(spec).unsqueeze(0)
        return {"spec": spec, "label": torch.tensor(self._labels[idx], dtype=torch.float32),
                "sample_id": str(row.get("sample_id", idx)), "subject_id": str(row.get("subject_id", idx))}


def build_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def collate_specs(batch):
    specs = torch.stack([b["spec"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    sids = [b["sample_id"] for b in batch]
    subjs = [b["subject_id"] for b in batch]
    return {"spec": specs, "label": labels, "sample_id": sids, "subject_id": subjs}


def _metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true_arr = np.array(y_true)
    m = {"accuracy": float(accuracy_score(y_true_arr, y_pred)),
         "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
         "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
         "f1": float(f1_score(y_true_arr, y_pred, zero_division=0))}
    try: m["auc"] = float(roc_auc_score(y_true_arr, y_prob))
    except ValueError: m["auc"] = None
    return m


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    probs, labels, sids, subjs = [], [], [], []
    for b in loader:
        logits = model(b["spec"].to(device)).squeeze(-1)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(b["label"].numpy().tolist())
        sids.extend(b["sample_id"]); subjs.extend(b["subject_id"])
    return pd.DataFrame({"sample_id": sids, "subject_id": subjs, "label": labels, "score_patients": probs})


@torch.no_grad()
def _find_best_threshold(model, loader, device):
    pred = _predict(model, loader, device)
    best_thr, best_recall = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.05):
        m = _metrics(pred["label"], pred["score_patients"], threshold=round(thr, 2))
        if m["recall"] > best_recall: best_recall = m["recall"]; best_thr = round(thr, 2)
    return best_thr


def _train_fold(model, train_loader, val_loader, device, epochs, lr, pos_weight, checkpoint_path=None):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    best_state, best_recall, best_thr = None, -1.0, 0.5
    patience_counter, max_patience = 0, 10

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for b in train_loader:
            x = b["spec"].to(device)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])
            y = b["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        thr = _find_best_threshold(model, val_loader, device)
        val_pred = _predict(model, val_loader, device)
        vm = _metrics(val_pred["label"], val_pred["score_patients"], threshold=thr)
        scheduler.step(vm["recall"])

        if vm["recall"] > best_recall:
            best_recall = vm["recall"]; best_thr = thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:02d} | loss={running_loss/max(len(train_loader),1):.4f} lr={current_lr:.2e} | "
              f"val_rec={vm['recall']:.4f} val_f1={vm['f1']:.4f} val_auc={vm['auc']}")

        if patience_counter >= max_patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    if checkpoint_path and best_state:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_threshold": float(best_thr),
                     "best_recall": float(best_recall)}, checkpoint_path)
    return model, best_thr


def parse_args():
    p = argparse.ArgumentParser(description="ResNet-18 Kotarba-style benchmark on Czech spectrograms.")
    p.add_argument("--spec-manifest", default=str(CONFIG.paths.processed_manifest_dir / "cz_main_spectrograms_kotarba.csv"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pos-weight", type=float, default=1.4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--save-checkpoint-dir", default=str(CONFIG.paths.reports_dir / "checkpoints" / "resnet18_kotarba"))
    p.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    manifest = pd.read_csv(args.spec_manifest)
    print(f"Samples: {len(manifest)}  Labels: {manifest['label'].value_counts().to_dict()}")

    y = manifest["label"].values
    groups = manifest["subject_id"].values
    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    all_preds, fold_metrics = [], []
    ckpt_dir = Path(args.save_checkpoint_dir)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(manifest)), y, groups=groups), start=1):
        print(f"\n===== Fold {fold}/{args.n_splits} =====")
        tr_df, vl_df = manifest.iloc[train_idx].copy(), manifest.iloc[val_idx].copy()
        print(f"  Train: {len(tr_df)}  Val: {len(vl_df)}")
        tr_loader = DataLoader(SpecDataset(tr_df), batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_specs, num_workers=0)
        vl_loader = DataLoader(SpecDataset(vl_df), batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_specs, num_workers=0)
        model = build_resnet18()
        model, best_thr = _train_fold(model, tr_loader, vl_loader, device,
                                       epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight,
                                       checkpoint_path=str(ckpt_dir / f"resnet18_fold{fold}_of_{args.n_splits}.pt"))
        val_pred = _predict(model, vl_loader, device)
        val_pred["fold"] = fold
        val_pred["pred_label"] = val_pred["score_patients"].apply(lambda s: "Patients" if s >= best_thr else "Healthy")
        vm = _metrics(val_pred["label"], val_pred["score_patients"], threshold=best_thr)
        vm["fold"] = fold; vm["best_threshold"] = best_thr; vm["n_train"] = len(tr_df); vm["n_val"] = len(vl_df)
        fold_metrics.append(vm); all_preds.append(val_pred)
        print(f"  Fold: bal_acc={vm['balanced_accuracy']:.4f} rec={vm['recall']:.4f} f1={vm['f1']:.4f} auc={vm['auc']}")

    all_pred = pd.concat(all_preds, ignore_index=True)
    sample_m = _metrics(all_pred["label"], all_pred["score_patients"], threshold=0.5)
    subj = all_pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj["pred_label"] = subj["score_patients"].apply(lambda s: "Patients" if s >= 0.5 else "Healthy")
    subject_m = _metrics(subj["label"], subj["score_patients"], threshold=0.5)

    fold_summary = {}
    for mn in ["balanced_accuracy", "auc", "f1", "recall"]:
        vals = [float(r[mn]) for r in fold_metrics if r.get(mn) is not None]
        if vals: fold_summary[mn] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                      "min": float(np.min(vals)), "max": float(np.max(vals))}

    result = {"config": {"arch": "ResNet18_1ch_LogPowerSpectrogram", "n_splits": args.n_splits,
               "epochs": args.epochs, "lr": args.lr, "pos_weight": args.pos_weight},
              "dataset": {"n_samples": int(len(all_pred)), "n_subjects": int(all_pred["subject_id"].nunique())},
              "sample_level": {"metrics": sample_m, "folds": fold_metrics, "fold_summary": fold_summary},
              "subject_level": {"metrics": subject_m}}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_pred.to_csv(out_dir / "predictions_resnet18_kotarba.csv", index=False)
    (out_dir / "metrics_resnet18_kotarba.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"\n===== ResNet-18 Results =====")
    print(f"Sample: acc={sample_m['accuracy']:.4f} bal_acc={sample_m['balanced_accuracy']:.4f} rec={sample_m['recall']:.4f} f1={sample_m['f1']:.4f} auc={sample_m['auc']}")
    print(f"Subject: acc={subject_m['accuracy']:.4f} bal_acc={subject_m['balanced_accuracy']:.4f} rec={subject_m['recall']:.4f} f1={subject_m['f1']:.4f} auc={subject_m['auc']}")
    for m, v in fold_summary.items(): print(f"  Fold {m}: {v['mean']:.4f} +/- {v['std']:.4f}")


if __name__ == "__main__":
    main()
