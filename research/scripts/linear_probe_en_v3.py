from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from research.src.config import CONFIG
from research.src.sequence_model import AttentionBiLSTMClassifierV3

OUTPUT_DIR = CONFIG.paths.reports_dir / "transfer_en"
PLOT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "transfer_en"


def load_checkpoint_v3(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    model = AttentionBiLSTMClassifierV3(
        input_dim=mc["input_dim"], lstm_hidden=mc["lstm_hidden"],
        lstm_layers=mc["lstm_layers"], dropout=mc["dropout"],
        fc_hidden=mc["fc_hidden"], num_heads=mc["num_heads"],
        use_layer_mix=mc.get("use_layer_mix", False),
        num_hubert_layers=mc.get("num_hubert_layers", 12),
        embedding_noise_std=mc.get("embedding_noise_std", 0.0),
        attention_entropy_lambda=mc.get("attention_entropy_lambda", 0.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, mc, payload.get("best_threshold", 0.5)


@torch.no_grad()
def extract_context_vectors_v3(model, seq_manifest, device):
    model.to(device)
    model.eval()
    contexts, labels, subject_ids = [], [], []
    for row in seq_manifest.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        _, ctx = model(seq, mask, return_context=True)
        contexts.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        subject_ids.append(str(row.subject_id))
    return np.stack(contexts, axis=0), np.array(labels), subject_ids


def loso_logistic_regression(X, y, subjects, C_values=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)):
    unique_subjects = sorted(set(subjects))
    scaler = StandardScaler()
    all_scores, all_preds, all_test_subjects, all_test_labels, fold_results = [], [], [], [], []

    for test_subj in unique_subjects:
        train_mask = np.array([s != test_subj for s in subjects])
        test_mask = np.array([s == test_subj for s in subjects])
        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        train_subjs_arr = np.array(subjects)[train_mask]
        best_C, best_inner_ba = C_values[0], -1.0
        for C_val in C_values:
            inner_ba_folds = []
            inner_unique = sorted(set(train_subjs_arr.tolist()))
            if len(inner_unique) <= 1:
                continue
            for inner_test in inner_unique:
                inner_train_mask = np.array([s != inner_test for s in train_subjs_arr])
                inner_test_mask = np.array([s == inner_test for s in train_subjs_arr])
                if inner_train_mask.sum() == 0 or inner_test_mask.sum() == 0:
                    continue
                X_inner_train, y_inner_train = X_train[inner_train_mask], y_train[inner_train_mask]
                X_inner_test, y_inner_test = X_train[inner_test_mask], y_train[inner_test_mask]
                clf = LogisticRegression(C=C_val, max_iter=5000, class_weight="balanced", random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_inner_train, y_inner_train)
                inner_ba_folds.append(float(balanced_accuracy_score(y_inner_test, clf.predict(X_inner_test))))
            if inner_ba_folds and np.mean(inner_ba_folds) > best_inner_ba:
                best_inner_ba = float(np.mean(inner_ba_folds))
                best_C = C_val
        clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        patient_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        score = float(proba[0, patient_idx])
        pred = int(score >= 0.5)
        all_scores.append(score); all_preds.append(pred)
        all_test_subjects.append(test_subj); all_test_labels.append(int(y_test[0]))
        fold_results.append({"test_subject": test_subj, "label": int(y_test[0]), "score": score, "pred": pred, "best_C": best_C, "n_train": int(train_mask.sum())})

    y_true = np.array(all_test_labels)
    y_scores, y_pred = np.array(all_scores), np.array(all_preds)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else None,
    }, fold_results, y_scores, y_true, all_test_subjects


def run_umap_v3(cz_X, cz_labels, en_X, en_labels, en_subjects, en_scores, output_path):
    X_all = np.vstack([cz_X, en_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    if HAS_UMAP:
        from umap import UMAP
        X_2d = UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        X_temp = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
        X_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1)).fit_transform(X_temp)
    n_cz = len(cz_X)
    cz_2d, en_2d = X_2d[:n_cz], X_2d[n_cz:]
    h_cz, p_cz = cz_2d[cz_labels == 0.0].mean(axis=0), cz_2d[cz_labels == 1.0].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax in axes:
        ax.scatter(cz_2d[cz_labels == 0.0, 0], cz_2d[cz_labels == 0.0, 1], c="steelblue", alpha=0.06, s=5, zorder=1)
        ax.scatter(cz_2d[cz_labels == 1.0, 0], cz_2d[cz_labels == 1.0, 1], c="crimson", alpha=0.06, s=5, zorder=1)
        ax.scatter(h_cz[0], h_cz[1], marker="X", s=300, edgecolors="darkblue", facecolors="skyblue", linewidth=3, zorder=10)
        ax.scatter(p_cz[0], p_cz[1], marker="X", s=300, edgecolors="darkred", facecolors="salmon", linewidth=3, zorder=10)
    axes[0].scatter(en_2d[en_labels == 0.0, 0], en_2d[en_labels == 0.0, 1], c="cyan", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="o", label="EN Healthy", zorder=5)
    axes[0].scatter(en_2d[en_labels == 1.0, 0], en_2d[en_labels == 1.0, 1], c="magenta", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="s", label="EN Patients", zorder=5)
    axes[0].set_title("V3 LayerMix: True Labels", fontsize=12, fontweight="bold")
    axes[0].legend(markerscale=1.2, fontsize=7, loc="upper right")
    sc = axes[1].scatter(en_2d[:, 0], en_2d[:, 1], c=en_scores, cmap="RdYlGn_r", s=160, edgecolors="black", linewidth=1.2, vmin=0.0, vmax=1.0, zorder=5)
    for i, (sid, x, y, sc_val) in enumerate(zip(en_subjects, en_2d[:, 0], en_2d[:, 1], en_scores)):
        axes[1].annotate(f"{sid}\n{sc_val:.3f}", (x, y), fontsize=5.5, fontweight="bold", ha="center", va="bottom")
    cbar = plt.colorbar(sc, ax=axes[1], shrink=0.75)
    cbar.set_label("Probe score", fontsize=9)
    axes[1].set_title("V3 LayerMix: Probe Scores", fontsize=12, fontweight="bold")
    fig.suptitle("Linear Probing (LOSO) -- V3 LayerMix on English", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Linear probing V3 (LOSO) on English context vectors.")
    p.add_argument("--checkpoint", default=str(Path("research/checkpoints/v3_final/layermix/v3_layermix_fold8_of_10.pt")))
    p.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_all_layers.csv"))
    p.add_argument("--cz-embeddings", default=str(CONFIG.paths.reports_dir / "CSV" / "embeddings_v3_layermix.parquet"))
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--plot-dir", default=str(PLOT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  CUDA: {torch.cuda.is_available()}")
    output_dir = Path(args.output_dir); plot_dir = Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading V3 checkpoint: {args.checkpoint}")
    model, mc, ckpt_thr = load_checkpoint_v3(args.checkpoint)
    print(f"  use_layer_mix={mc.get('use_layer_mix')}  threshold={ckpt_thr:.3f}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"Extracting context vectors for {len(manifest)} English samples ...")
    X_en, y_en, subjects_en = extract_context_vectors_v3(model, manifest, device)
    print(f"  Shape: {X_en.shape}")

    print("\n=== LOSO Logistic Regression (V3) ===")
    metrics_lr, fold_results, scores_lr, y_true, test_subjects = loso_logistic_regression(X_en, y_en, subjects_en)

    print(f"\nLinear Probe V3 Results (LOSO LR):")
    for k, v in metrics_lr.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    cz_df = pd.read_parquet(args.cz_embeddings)
    cz_embed_cols = [c for c in cz_df.columns if str(c).isdigit()]
    cz_X = cz_df[cz_embed_cols].astype(np.float64).values
    cz_labels = cz_df["label"].astype(float).values
    print(f"\nGenerating UMAP (CZ={len(cz_X)} + EN={len(X_en)}) ...")
    run_umap_v3(cz_X, cz_labels, X_en, y_en, test_subjects, scores_lr, plot_dir / "umap_linear_probe_v3_layermix.png")

    subject_details = []
    for i, sid in enumerate(test_subjects):
        subject_details.append({
            "subject_id": sid, "label": int(y_true[i]), "lr_probe_score": float(scores_lr[i]),
            "lr_pred": int(scores_lr[i] >= 0.5), "lr_correct": int(scores_lr[i] >= 0.5) == int(y_true[i]),
        })

    report = {
        "environment": {"cuda_available": torch.cuda.is_available(), "device": "cuda" if torch.cuda.is_available() else "cpu"},
        "checkpoint": str(args.checkpoint),
        "model_config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v for k, v in mc.items()},
        "linear_probe_loso": {**{k: round(v, 6) if v is not None else None for k, v in metrics_lr.items()}, "folds": fold_results},
        "per_subject": subject_details,
    }
    (output_dir / "linear_probe_v3_layermix_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(subject_details).to_csv(output_dir / "linear_probe_v3_layermix_per_subject.csv", index=False)

    print(f"\nPer-subject:")
    for d in subject_details:
        print(f"  {d['subject_id']:>6s}  label={d['label']}  score={d['lr_probe_score']:.4f}  {'OK' if d['lr_correct'] else 'ERR'}")
    print(f"\nReport:  {output_dir / 'linear_probe_v3_layermix_report.json'}")
    print(f"Subjects: {output_dir / 'linear_probe_v3_layermix_per_subject.csv'}")
    print(f"UMAP:     {plot_dir / 'umap_linear_probe_v3_layermix.png'}")


if __name__ == "__main__":
    main()
