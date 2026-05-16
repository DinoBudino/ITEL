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
                              roc_auc_score)

from research.src.config import CONFIG
from research.src.sequence_model import AttentionBiLSTMClassifierV3


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
    subj["pred_label"] = subj["score_patients_subject"].apply(lambda s: 1.0 if s >= 0.5 else 0.0)
    return subj


def load_checkpoint_v3(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    model = AttentionBiLSTMClassifierV3(
        input_dim=mc["input_dim"],
        lstm_hidden=mc["lstm_hidden"],
        lstm_layers=mc["lstm_layers"],
        dropout=mc["dropout"],
        fc_hidden=mc["fc_hidden"],
        num_heads=mc["num_heads"],
        use_layer_mix=mc.get("use_layer_mix", False),
        num_hubert_layers=mc.get("num_hubert_layers", 12),
        embedding_noise_std=mc.get("embedding_noise_std", 0.0),
        attention_entropy_lambda=mc.get("attention_entropy_lambda", 0.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ckpt_threshold = payload.get("best_threshold", 0.5)
    return model, mc, payload.get("preprocessing", {}), ckpt_threshold


@torch.no_grad()
def predict_sequences_v3(model, manifest_df, device):
    model.to(device)
    model.eval()
    all_probs, all_contexts, all_labels = [], [], []
    all_sample_ids, all_subject_ids = [], []

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
    return pd.DataFrame({
        "sample_id": all_sample_ids, "subject_id": all_subject_ids,
        "label": all_labels, "score_patients": all_probs,
    }), emb


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot V3: evaluate V3 LayerMix checkpoint on English.")
    p.add_argument("--checkpoint", default=str(Path("research/checkpoints/v3_final/layermix/v3_layermix_fold8_of_10.pt")))
    p.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_all_layers.csv"))
    p.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en"))
    p.add_argument("--device", default=None)
    p.add_argument("--report-name", default="zero_shot_v3_layermix")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  CUDA: {torch.cuda.is_available()}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading V3 checkpoint: {checkpoint_path}")
    model, model_config, preproc, ckpt_threshold = load_checkpoint_v3(checkpoint_path)
    model.to(device)
    print(f"Model: use_layer_mix={model_config.get('use_layer_mix')}  noise_std={model_config.get('embedding_noise_std')}  "
          f"entropy_lambda={model_config.get('attention_entropy_lambda')}")
    print(f"Checkpoint threshold: {ckpt_threshold:.3f}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"\nEnglish test samples: {len(manifest)}  Labels: {manifest['label'].value_counts().to_dict()}")

    pred_frame, embeddings = predict_sequences_v3(model, manifest, device)
    pred_frame["pred_label_raw"] = pred_frame["score_patients"].apply(lambda s: 1.0 if s >= ckpt_threshold else 0.0)

    sample_raw = _metrics(pred_frame["label"], pred_frame["score_patients"], threshold=ckpt_threshold)
    sweep_rows, best_sweep = _threshold_sweep(pred_frame["label"], pred_frame["score_patients"])
    pred_frame["pred_label"] = pred_frame["score_patients"].apply(lambda s: 1.0 if s >= best_sweep["threshold"] else 0.0)

    subj = _subject_aggregation(pred_frame)
    subject_raw = _metrics(subj["label"], subj["score_patients_subject"], threshold=ckpt_threshold)
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
        "zero_shot_raw": {
            "threshold": ckpt_threshold,
            "sample_metrics": sample_raw,
            "subject_metrics": subject_raw,
        },
        "zero_shot_recalibrated": {
            "best_threshold": best_sweep["threshold"],
            "best_balanced_accuracy": best_sweep["balanced_accuracy"],
            "sample_metrics": best_sweep,
            "sample_sweep": sweep_rows,
            "subject_best": subj_best_sweep,
            "subject_sweep": subj_sweep_rows,
        },
        "embeddings_shape": list(embeddings.shape),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rn = args.report_name

    pred_out = out_dir / f"predictions_{rn}.csv"
    emb_out = out_dir / f"embeddings_{rn}.parquet"
    metrics_out = out_dir / f"metrics_{rn}.json"

    pred_frame.to_csv(pred_out, index=False)
    emb_df = pd.DataFrame(embeddings)
    emb_df["sample_id"] = pred_frame["sample_id"].values
    emb_df["subject_id"] = pred_frame["subject_id"].values
    emb_df["label"] = pred_frame["label"].values
    emb_df.to_parquet(emb_out)
    metrics_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n===== ZERO-SHOT V3 LAYERMIX =====")
    print(f"Raw (thr={ckpt_threshold}):  bal_acc={sample_raw['balanced_accuracy']:.4f}  "
          f"auc={sample_raw['auc']}  recall={sample_raw['recall']:.4f}")
    print(f"Recalib (thr={best_sweep['threshold']}): bal_acc={best_sweep['balanced_accuracy']:.4f}  "
          f"auc={sweep_rows[0]['auc']}  recall={best_sweep['recall']:.4f}")

    print(f"\nPer-subject scores:")
    for _, row in subj.iterrows():
        sid = str(row["subject_id"])
        lab = int(float(row["label"]))
        sc = float(row["score_patients_subject"])
        pl = int(float(row["pred_label"]))
        status = "FP" if lab == 0 and pl == 1 else "FN" if lab == 1 and pl == 0 else "OK"
        print(f"  {sid:>6s}  label={lab:1d}  score={sc:.4f}  {status}")

    print(f"\nPredictions: {pred_out}")
    print(f"Embeddings:  {emb_out}")
    print(f"Metrics:     {metrics_out}")


if __name__ == "__main__":
    main()
