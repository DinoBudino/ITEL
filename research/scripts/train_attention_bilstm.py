from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from research.src.config import CONFIG
from research.src.sequence_model import (AttentionBiLSTMClassifier,
                                          SequenceDataset,
                                          collate_sequences)
from research.src.tracking import (configure_tracking, log_artifact, log_dict,
                                    log_metrics, log_params)


def _fold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    return metrics


def _aggregate_subjects(frame):
    subject_frame = (
        frame.groupby(["fold", "subject_id", "label"], as_index=False)["score_patients"]
        .mean()
        .rename(columns={"score_patients": "score_patients_mean"})
    )
    subject_frame["pred_label"] = subject_frame["score_patients_mean"].apply(
        lambda s: "Patients" if s >= 0.5 else "Healthy"
    )
    return subject_frame


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_sample_ids = []
    all_subject_ids = []
    for batch in loader:
        logits = model(batch["sequence"].to(device), batch["mask"].to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_sample_ids.extend(batch["sample_id"])
        all_subject_ids.extend(batch["subject_id"])
    return pd.DataFrame({
        "sample_id": all_sample_ids,
        "subject_id": all_subject_ids,
        "label": all_labels,
        "score_patients": all_probs,
    })


@torch.no_grad()
def _find_best_threshold(model, loader, device):
    pred_frame = _predict(model, loader, device)
    y_true = pred_frame["label"].values
    y_prob = pred_frame["score_patients"].values
    best_thr = 0.5
    best_recall = -1.0
    best_metrics = None
    for thr in np.arange(0.20, 0.81, 0.05):
        thr = float(round(thr, 2))
        m = _fold_metrics(y_true, y_prob, threshold=thr)
        if m["recall"] > best_recall:
            best_recall = m["recall"]
            best_thr = thr
            best_metrics = m
    return best_thr, best_metrics


def _train_fold(model, train_loader, val_loader, device, epochs, lr, weight_decay, pos_weight):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_recall = -1.0
    best_threshold = 0.5

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["sequence"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, m), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        threshold, val_metrics = _find_best_threshold(model, val_loader, device)
        if val_metrics["recall"] > best_recall:
            best_recall = val_metrics["recall"]
            best_threshold = threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch:02d} | loss={running_loss / max(len(train_loader), 1):.4f} | "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']}"
        )

    model.load_state_dict(best_state)
    return model, best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Attention Bi-LSTM on HuBERT sequences with Recall-optimized threshold."
    )
    parser.add_argument(
        "--duration-mode",
        choices=["fixed_2s", "natural"],
        default="fixed_2s",
        help="Which sequence manifest to use. auto-derives --sequence-manifest path.",
    )
    parser.add_argument(
        "--sequence-manifest",
        default=None,
        help="Path to sequence manifest CSV (overrides --duration-mode auto-path).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc-hidden", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads (1=single, 4=multi).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--predictions-output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "metrics_attention_bilstm.json"),
    )
    parser.add_argument("--mlflow-experiment", default="itel-attention-bilstm")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sequence_manifest is None:
        if args.duration_mode == "natural":
            args.sequence_manifest = str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural.csv")
        else:
            args.sequence_manifest = str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_train_default.csv")

    tag = args.duration_mode
    if args.num_heads > 1:
        tag = f"{tag}_mha{args.num_heads}"
    if args.predictions_output == str(CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm.csv"):
        args.predictions_output = str(CONFIG.paths.reports_dir / "CSV" / f"predictions_attention_bilstm_{tag}.csv")
    if args.metrics_output == str(CONFIG.paths.reports_dir / "CSV" / "metrics_attention_bilstm.json"):
        args.metrics_output = str(CONFIG.paths.reports_dir / "CSV" / f"metrics_attention_bilstm_{tag}.json")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Duration mode: {tag}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"Loaded {len(manifest)} sequence samples")
    print(f"Label distribution: {manifest['label'].value_counts().to_dict()}")

    lengths = manifest["sequence_length"].values
    print(f"Sequence lengths: min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")

    y = manifest["label"].values
    groups = manifest["subject_id"].values

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    if mlflow is not None:
        tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    else:
        tracking_uri = None

    with (mlflow.start_run(run_name=args.mlflow_run_name) if mlflow is not None else _NullContext()):
        all_predictions = []
        fold_metrics_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(manifest)), y, groups=groups), start=1
        ):
            print(f"\n===== Fold {fold_idx}/{args.n_splits} =====")

            train_df = manifest.iloc[train_idx].copy()
            val_df = manifest.iloc[val_idx].copy()
            print(f"  Train: {len(train_df)}  Val: {len(val_df)}")

            train_loader = DataLoader(
                SequenceDataset(train_df), batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_sequences, num_workers=0,
            )
            val_loader = DataLoader(
                SequenceDataset(val_df), batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_sequences, num_workers=0,
            )

            model = AttentionBiLSTMClassifier(
                input_dim=768,
                lstm_hidden=args.lstm_hidden,
                lstm_layers=args.lstm_layers,
                dropout=args.dropout,
                fc_hidden=args.fc_hidden,
                num_heads=args.num_heads,
            )

            model, best_thr = _train_fold(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay, pos_weight=args.pos_weight,
            )
            print(f"  Best threshold: {best_thr:.2f}")

            val_pred = _predict(model, val_loader, device)
            val_pred["fold"] = fold_idx
            val_pred["pred_label"] = val_pred["score_patients"].apply(
                lambda s: "Patients" if s >= best_thr else "Healthy"
            )
            val_metrics = _fold_metrics(
                val_pred["label"].values, val_pred["score_patients"].values, threshold=best_thr,
            )
            val_metrics["fold"] = fold_idx
            val_metrics["best_threshold"] = best_thr
            val_metrics["n_train"] = len(train_df)
            val_metrics["n_val"] = len(val_df)
            fold_metrics_list.append(val_metrics)
            all_predictions.append(val_pred)

            print(
                f"  Fold metrics: bal_acc={val_metrics['balanced_accuracy']:.4f} "
                f"recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} "
                f"auc={val_metrics['auc']}"
            )

        all_pred = pd.concat(all_predictions, ignore_index=True)
        subject_pred = _aggregate_subjects(all_pred)

        sample_metrics = _fold_metrics(
            all_pred["label"].values, all_pred["score_patients"].values, threshold=0.5,
        )
        subject_metrics = _fold_metrics(
            subject_pred["label"].values, subject_pred["score_patients_mean"].values, threshold=0.5,
        )

        fold_summary = {}
        for metric_name in ["balanced_accuracy", "auc", "f1", "recall"]:
            values = [float(row[metric_name]) for row in fold_metrics_list if row.get(metric_name) is not None]
            if values:
                fold_summary[metric_name] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values)),
                    "min": float(np.min(values)), "max": float(np.max(values)),
                }

        result = {
            "config": {
                "arch": "HuBERT_BASE -> AttentionBiLSTM",
                "num_heads": args.num_heads,
                "duration_mode": tag,
                "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
                "dropout": args.dropout, "fc_hidden": args.fc_hidden,
                "epochs": args.epochs, "lr": args.lr, "pos_weight": args.pos_weight,
                "n_splits": args.n_splits, "random_state": args.random_state,
            },
            "dataset": {
                "n_samples": len(all_pred), "n_subjects": int(all_pred["subject_id"].nunique()),
                "label_distribution": manifest["label"].value_counts().to_dict(),
            },
            "sample_level": {
                "metrics": sample_metrics, "folds": fold_metrics_list, "fold_summary": fold_summary,
            },
            "subject_level": {"metrics": subject_metrics},
        }

        predictions_output = Path(args.predictions_output)
        metrics_output = Path(args.metrics_output)
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.parent.mkdir(parents=True, exist_ok=True)

        all_pred.to_csv(predictions_output, index=False)
        subject_csv = predictions_output.with_stem(predictions_output.stem + "_subjects")
        subject_pred.to_csv(subject_csv, index=False)
        metrics_output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

        print("\n===== Cross-validation summary =====")
        print(
            f"Sample-level: bal_acc={sample_metrics['balanced_accuracy']:.4f} "
            f"recall={sample_metrics['recall']:.4f} f1={sample_metrics['f1']:.4f} "
            f"auc={sample_metrics['auc']}"
        )
        print(
            f"Subject-level: bal_acc={subject_metrics['balanced_accuracy']:.4f} "
            f"recall={subject_metrics['recall']:.4f} f1={subject_metrics['f1']:.4f} "
            f"auc={subject_metrics['auc']}"
        )
        for m, v in fold_summary.items():
            print(f"  {m}: {v['mean']:.4f} +/- {v['std']:.4f}")

        print(f"\nPredictions: {predictions_output}")
        print(f"Subjects:     {subject_csv}")
        print(f"Metrics:      {metrics_output}")

        if mlflow is not None:
            mlflow.set_tags({"stage": "research", "script": "train_attention_bilstm.py", "arch": "HuBERT+AttentionBiLSTM", "duration_mode": tag, "num_heads": str(args.num_heads)})
            log_params({
                "duration_mode": tag,
                "num_heads": args.num_heads,
                "lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
                "dropout": args.dropout, "fc_hidden": args.fc_hidden,
                "epochs": args.epochs, "lr": args.lr, "pos_weight": args.pos_weight,
                "n_splits": args.n_splits, "n_samples": len(all_pred),
                "tracking_uri": tracking_uri or "local",
            })
            log_dict("metadata/training_config.json", result["config"])
            log_dict("metadata/label_distribution.json", result["dataset"])
            log_metrics({"sample_" + k: v for k, v in sample_metrics.items() if v is not None})
            log_metrics({"subject_" + k: v for k, v in subject_metrics.items() if v is not None})
            log_artifact(predictions_output, artifact_path="predictions")
            log_artifact(metrics_output, artifact_path="metrics")


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
