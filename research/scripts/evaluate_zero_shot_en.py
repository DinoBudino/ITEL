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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score, roc_curve)

from research.src.config import CONFIG
from research.src.sequence_model import (AttentionBiLSTMClassifier,
                                          SequenceDataset,
                                          collate_sequences)


def _metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true_arr = np.array(y_true, dtype=np.float64)
    m = {
        "n_samples": int(len(y_true_arr)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }
    try:
        m["auc"] = float(roc_auc_score(y_true_arr, y_prob))
    except ValueError:
        m["auc"] = None
    return m


def _threshold_sweep(y_true, y_prob):
    rows = []
    best = None
    for thr in np.arange(0.05, 1.0, 0.05):
        thr = float(round(thr, 2))
        row = _metrics(y_true, y_prob, threshold=thr)
        rows.append(row)
        if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
            best = row
    return rows, best


def _subject_aggregation(frame):
    subj = frame.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj.rename(columns={"score_patients": "score_patients_subject"}, inplace=True)
    subj["pred_label"] = subj["score_patients_subject"].apply(
        lambda s: 1.0 if s >= 0.5 else 0.0
    )
    return subj


def load_checkpoint(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = payload["model_config"]
    model = AttentionBiLSTMClassifier(
        input_dim=model_config["input_dim"],
        lstm_hidden=model_config["lstm_hidden"],
        lstm_layers=model_config["lstm_layers"],
        dropout=model_config["dropout"],
        fc_hidden=model_config["fc_hidden"],
        num_heads=model_config["num_heads"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ckpt_threshold = payload.get("best_threshold", 0.5)
    return model, model_config, payload.get("preprocessing", {}), ckpt_threshold


@torch.no_grad()
def predict_sequences(model, manifest_df, device, batch_size=1):
    model.to(device)
    model.eval()
    all_probs = []
    all_contexts = []
    all_labels = []
    all_sample_ids = []
    all_subject_ids = []

    for row in manifest_df.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)

        logits, ctx = model(seq, mask, return_context=True)
        prob = float(torch.sigmoid(logits).cpu().item())
        all_probs.append(prob)
        all_contexts.append(ctx.cpu().numpy().squeeze(0))
        all_labels.append(float(row.label == "Patients"))
        all_sample_ids.append(row.sample_id)
        all_subject_ids.append(row.subject_id)

    emb = np.stack(all_contexts, axis=0)
    pred_frame = pd.DataFrame({
        "sample_id": all_sample_ids,
        "subject_id": all_subject_ids,
        "label": all_labels,
        "score_patients": all_probs,
    })
    return pred_frame, emb


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot: evaluate Czech HuBERT+BiLSTM checkpoint on English test set.")
    parser.add_argument("--checkpoint", default=str(Path("research/checkpoints/best_model_checo_fase1.pt")))
    parser.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test.csv"))
    parser.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-name", default="zero_shot_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model, model_config, preproc, ckpt_threshold = load_checkpoint(checkpoint_path)
    model.to(device)
    print(f"Model config: {model_config}")
    print(f"Checkpoint threshold: {ckpt_threshold:.3f}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"\nEnglish test samples: {len(manifest)}")
    print(f"Label distribution: {manifest['label'].value_counts().to_dict()}")

    pred_frame, embeddings = predict_sequences(model, manifest, device)
    pred_frame["pred_label"] = pred_frame["score_patients"].apply(
        lambda s: 1.0 if s >= ckpt_threshold else 0.0
    )

    sample_metrics = _metrics(pred_frame["label"], pred_frame["score_patients"], threshold=ckpt_threshold)
    sweep_rows, best_sweep = _threshold_sweep(pred_frame["label"], pred_frame["score_patients"])

    subj = _subject_aggregation(pred_frame)
    subject_metrics = _metrics(subj["label"], subj["score_patients_subject"], threshold=0.5)
    subj_sweep_rows, subj_best_sweep = _threshold_sweep(subj["label"], subj["score_patients_subject"])

    report = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "model_config": model_config,
            "ckpt_threshold": ckpt_threshold,
            "preprocessing": preproc,
            "source_manifest": args.sequence_manifest,
        },
        "dataset": {
            "n_samples": int(len(manifest)),
            "n_subjects": int(manifest["subject_id"].nunique()),
            "label_distribution": manifest["label"].value_counts().to_dict(),
        },
        "sample_level": {
            "metrics": sample_metrics,
            "threshold_sweep": {"grid": sweep_rows, "best": best_sweep},
        },
        "subject_level": {
            "metrics": subject_metrics,
            "threshold_sweep": {"grid": subj_sweep_rows, "best": subj_best_sweep},
        },
        "embeddings_shape": list(embeddings.shape),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_name = args.report_name

    pred_out = out_dir / f"predictions_{report_name}.csv"
    emb_out = out_dir / f"embeddings_{report_name}.parquet"
    metrics_out = out_dir / f"metrics_{report_name}.json"

    pred_frame.to_csv(pred_out, index=False)
    emb_df = pd.DataFrame(embeddings)
    emb_df["sample_id"] = pred_frame["sample_id"].values
    emb_df["subject_id"] = pred_frame["subject_id"].values
    emb_df["label"] = pred_frame["label"].values
    emb_df.to_parquet(emb_out)
    metrics_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n===== ZERO-SHOT RESULTS =====")
    print(f"Sample-level:")
    for k, v in sample_metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSubject-level:")
    for k, v in subject_metrics.items():
        print(f"  {k}: {v}")

    print(f"\nThreshold sweep best (sample): thr={best_sweep['threshold']}, bal_acc={best_sweep['balanced_accuracy']:.4f}")
    print(f"Threshold sweep best (subject): thr={subj_best_sweep['threshold']}, bal_acc={subj_best_sweep['balanced_accuracy']:.4f}")

    print(f"\nPredictions: {pred_out}")
    print(f"Embeddings:  {emb_out}")
    print(f"Metrics:     {metrics_out}")

    print(f"\nPer-subject scores:")
    for _, row in subj.iterrows():
        sid = str(row["subject_id"])
        lab = int(float(row["label"]))
        sc = float(row["score_patients_subject"])
        pl = int(float(row["pred_label"]))
        status = "FP" if lab == 0 and pl == 1 else \
                 "FN" if lab == 1 and pl == 0 else "OK"
        print(f"  {sid:>6s}  label={lab:1d}  score={sc:.4f}  {status}")


if __name__ == "__main__":
    main()
