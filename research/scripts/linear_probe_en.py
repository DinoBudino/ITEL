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
import torch.nn as nn
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
from research.src.sequence_model import AttentionBiLSTMClassifier

OUTPUT_DIR = CONFIG.paths.reports_dir / "transfer_en"
PLOT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "transfer_en"


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = AttentionBiLSTMClassifier(
        input_dim=payload["model_config"]["input_dim"],
        lstm_hidden=payload["model_config"]["lstm_hidden"],
        lstm_layers=payload["model_config"]["lstm_layers"],
        dropout=payload["model_config"]["dropout"],
        fc_hidden=payload["model_config"]["fc_hidden"],
        num_heads=payload["model_config"]["num_heads"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def extract_context_vectors(model, seq_manifest, device):
    model.to(device)
    model.eval()
    contexts = []
    labels = []
    subject_ids = []
    for row in seq_manifest.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        with torch.no_grad():
            _, ctx = model(seq, mask, return_context=True)
        contexts.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        subject_ids.append(str(row.subject_id))
    return np.stack(contexts, axis=0), np.array(labels), subject_ids


def loso_logistic_regression(X, y, subjects, C_values=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)):
    unique_subjects = sorted(set(subjects))
    scaler = StandardScaler()

    all_preds = []
    all_scores = []
    all_test_subjects = []
    all_test_labels = []
    fold_results = []

    for test_subj in unique_subjects:
        train_mask = np.array([s != test_subj for s in subjects])
        test_mask = np.array([s == test_subj for s in subjects])

        X_train_raw = X[train_mask]
        y_train = y[train_mask]
        X_test_raw = X[test_mask]
        y_test = y[test_mask]

        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # inner LOSO for C selection (on 11 training samples)
        train_subjs_arr = np.array(subjects)[train_mask]
        best_C = C_values[0]
        best_inner_ba = -1.0

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
                X_inner_train = X_train[inner_train_mask]
                y_inner_train = y_train[inner_train_mask]
                X_inner_test = X_train[inner_test_mask]
                y_inner_test = y_train[inner_test_mask]

                clf = LogisticRegression(C=C_val, max_iter=5000, class_weight="balanced", random_state=42)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(X_inner_train, y_inner_train)
                preds = clf.predict(X_inner_test)
                inner_ba_folds.append(float(balanced_accuracy_score(y_inner_test, preds)))

            if inner_ba_folds:
                mean_ba = float(np.mean(inner_ba_folds))
                if mean_ba > best_inner_ba:
                    best_inner_ba = mean_ba
                    best_C = C_val

        clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        patient_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        score = float(proba[0, patient_idx])
        pred = int(score >= 0.5)

        all_scores.append(score)
        all_preds.append(pred)
        all_test_subjects.append(test_subj)
        all_test_labels.append(int(y_test[0]))
        fold_results.append({
            "test_subject": test_subj,
            "label": int(y_test[0]),
            "score": score,
            "pred": pred,
            "best_C": best_C,
            "n_train": int(train_mask.sum()),
        })

    y_true = np.array(all_test_labels)
    y_scores = np.array(all_scores)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else None,
    }

    return metrics, fold_results, y_scores, y_true, all_test_subjects


def train_full_probe(X, y, subjects, best_C_counts):
    best_C = max(set(best_C_counts), key=best_C_counts.count) if best_C_counts else 1.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_scaled, y)
    return clf, scaler, best_C


def run_umap_post_probe(cz_X, cz_labels, en_X, en_labels, en_subjects, en_scores, output_path):
    X_all = np.vstack([cz_X, en_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    if HAS_UMAP:
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        reducer = PCA(n_components=50, random_state=42)
        X_temp = reducer.fit_transform(X_scaled)
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1))
        X_2d = reducer.fit_transform(X_temp)

    n_cz = len(cz_X)
    cz_2d = X_2d[:n_cz]
    en_2d = X_2d[n_cz:]

    h_cz = cz_2d[cz_labels == 0.0].mean(axis=0)
    p_cz = cz_2d[cz_labels == 1.0].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Left: by label
    ax = axes[0]
    ax.scatter(cz_2d[cz_labels == 0.0, 0], cz_2d[cz_labels == 0.0, 1],
               c="steelblue", alpha=0.06, s=5, label="CZ Healthy", zorder=1)
    ax.scatter(cz_2d[cz_labels == 1.0, 0], cz_2d[cz_labels == 1.0, 1],
               c="crimson", alpha=0.06, s=5, label="CZ Patients", zorder=1)
    ax.scatter(en_2d[en_labels == 0.0, 0], en_2d[en_labels == 0.0, 1],
               c="cyan", alpha=0.95, s=140, edgecolors="black", linewidth=1.5,
               marker="o", label="EN Healthy", zorder=5)
    ax.scatter(en_2d[en_labels == 1.0, 0], en_2d[en_labels == 1.0, 1],
               c="magenta", alpha=0.95, s=140, edgecolors="black", linewidth=1.5,
               marker="s", label="EN Patients", zorder=5)
    ax.scatter(h_cz[0], h_cz[1], marker="X", s=300, edgecolors="darkblue",
               facecolors="skyblue", linewidth=3, zorder=10, label="CZ Healthy centroid")
    ax.scatter(p_cz[0], p_cz[1], marker="X", s=300, edgecolors="darkred",
               facecolors="salmon", linewidth=3, zorder=10, label="CZ Patient centroid")
    for i, (sid, x, y) in enumerate(zip(en_subjects, en_2d[:, 0], en_2d[:, 1])):
        ax.annotate(sid, (x, y), fontsize=6, fontweight="bold", ha="center", va="bottom")
    ax.set_title("Embedding Space: True Labels", fontsize=12, fontweight="bold")
    ax.legend(markerscale=1.2, fontsize=7, loc="upper right")

    # Right: by probe score
    ax2 = axes[1]
    ax2.scatter(cz_2d[cz_labels == 0.0, 0], cz_2d[cz_labels == 0.0, 1],
                c="steelblue", alpha=0.04, s=4, zorder=1)
    ax2.scatter(cz_2d[cz_labels == 1.0, 0], cz_2d[cz_labels == 1.0, 1],
                c="crimson", alpha=0.04, s=4, zorder=1)
    sc = ax2.scatter(en_2d[:, 0], en_2d[:, 1], c=en_scores, cmap="RdYlGn_r",
                     s=160, edgecolors="black", linewidth=1.2, vmin=0.0, vmax=1.0, zorder=5)
    ax2.scatter(h_cz[0], h_cz[1], marker="X", s=300, edgecolors="darkblue",
                facecolors="skyblue", linewidth=3, zorder=10)
    ax2.scatter(p_cz[0], p_cz[1], marker="X", s=300, edgecolors="darkred",
                facecolors="salmon", linewidth=3, zorder=10)
    for i, (sid, x, y, sc_val) in enumerate(zip(en_subjects, en_2d[:, 0], en_2d[:, 1], en_scores)):
        ax2.annotate(f"{sid}\n{sc_val:.3f}", (x, y), fontsize=5.5, fontweight="bold",
                     ha="center", va="bottom")
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.75)
    cbar.set_label("Probe score (0=Healthy, 1=Patients)", fontsize=9)
    ax2.set_title("Embedding Space: Probe Scores\n(green=healthy, red=patient)", fontsize=12, fontweight="bold")

    fig.suptitle("Linear Probing (LOSO Logistic Regression) on English Context Vectors",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probing (LOSO) on English HuBERT context vectors.")
    parser.add_argument("--checkpoint", default=str(Path("research/checkpoints/best_model_checo_fase1.pt")))
    parser.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test.csv"))
    parser.add_argument("--cz-embeddings", default=str(CONFIG.paths.reports_dir / "CSV" / "embeddings_contrastive_l0.1.parquet"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--plot-dir", default=str(PLOT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  CUDA: {torch.cuda.is_available()}")

    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, ckpt = load_checkpoint(args.checkpoint)
    print(f"Model config: {ckpt['model_config']}")

    seq_manifest = pd.read_csv(args.sequence_manifest)
    print(f"\nExtracting context vectors for {len(seq_manifest)} English samples ...")
    X_en, y_en, subjects_en = extract_context_vectors(model, seq_manifest, device)
    print(f"  Shape: {X_en.shape}")

    print("\n=== LOSO Logistic Regression Probing ===")
    metrics_lr, fold_results, scores_lr, y_true, test_subjects = loso_logistic_regression(
        X_en, y_en, subjects_en,
    )

    print(f"\nLinear Probe Results (LOSO Logistic Regression):")
    for k, v in metrics_lr.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    # Also try training the original classifier (2-layer MLP) with LOSO
    print(f"\n=== LOSO with Original Classifier (2-layer MLP) ===")

    mlp_all_scores = []
    mlp_all_preds = []
    mlp_all_labels = []
    mlp_fold_results = []

    unique_subjects = sorted(set(subjects_en))
    for test_subj in unique_subjects:
        train_mask = np.array([s != test_subj for s in subjects_en])
        test_mask = np.array([s == test_subj for s in subjects_en])

        X_train = torch.tensor(X_en[train_mask], dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_en[train_mask], dtype=torch.float32).to(device)
        X_test = torch.tensor(X_en[test_mask], dtype=torch.float32).to(device)

        # Freeze all except classifier
        classifier_model = AttentionBiLSTMClassifier(
            input_dim=ckpt["model_config"]["input_dim"],
            lstm_hidden=ckpt["model_config"]["lstm_hidden"],
            lstm_layers=ckpt["model_config"]["lstm_layers"],
            dropout=ckpt["model_config"]["dropout"],
            fc_hidden=ckpt["model_config"]["fc_hidden"],
            num_heads=ckpt["model_config"]["num_heads"],
        )
        classifier_model.load_state_dict(ckpt["state_dict"])
        classifier_model.to(device)

        for name, param in classifier_model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.4], device=device))
        optimizer = torch.optim.AdamW(
            [p for p in classifier_model.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=0.5,
        )

        classifier_model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            logits = classifier_model.classifier(X_train).squeeze(-1)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

        classifier_model.eval()
        with torch.no_grad():
            test_logits = classifier_model.classifier(X_test).squeeze(-1)
            test_prob = float(torch.sigmoid(test_logits).cpu().item())

        pred = int(test_prob >= 0.5)
        mlp_all_scores.append(test_prob)
        mlp_all_preds.append(pred)
        mlp_all_labels.append(int(y_en[test_mask][0]))

        correct = int(pred == int(y_en[test_mask][0]))
        mlp_fold_results.append({
            "test_subject": test_subj, "label": int(y_en[test_mask][0]),
            "score": test_prob, "pred": pred, "correct": correct,
        })

    y_true_mlp = np.array(mlp_all_labels)
    y_scores_mlp = np.array(mlp_all_scores)
    y_pred_mlp = np.array(mlp_all_preds)

    metrics_mlp = {
        "accuracy": float(accuracy_score(y_true_mlp, y_pred_mlp)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_mlp, y_pred_mlp)),
        "precision": float(precision_score(y_true_mlp, y_pred_mlp, zero_division=0)),
        "recall": float(recall_score(y_true_mlp, y_pred_mlp, zero_division=0)),
        "f1": float(f1_score(y_true_mlp, y_pred_mlp, zero_division=0)),
        "auc": float(roc_auc_score(y_true_mlp, y_scores_mlp)) if len(np.unique(y_true_mlp)) > 1 else None,
    }

    print(f"\nOriginal Classifier Probe Results (LOSO MLP):")
    for k, v in metrics_mlp.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    # Train final full probe on all data
    C_values_used = [r["best_C"] for r in fold_results]
    full_clf, full_scaler, final_C = train_full_probe(X_en, y_en, subjects_en, C_values_used)
    full_scores = full_clf.predict_proba(full_scaler.transform(X_en))
    patient_idx = list(full_clf.classes_).index(1) if 1 in full_clf.classes_ else 1
    final_scores = full_scores[:, patient_idx]

    print(f"\nFinal probe (trained on all {len(X_en)} samples, C={final_C}):")
    print(f"  Coefficients shape: {full_clf.coef_.shape}")
    print(f"  Intercept: {full_clf.intercept_[0]:.4f}")

    # UMAP
    print(f"\nGenerating UMAP ...")
    cz_df = pd.read_parquet(args.cz_embeddings)
    cz_embed_cols = [c for c in cz_df.columns if str(c).isdigit()]
    cz_X = cz_df[cz_embed_cols].astype(np.float64).values
    cz_labels_full = cz_df["label"].astype(float).values

    run_umap_post_probe(
        cz_X, cz_labels_full,
        X_en, y_en,
        subjects_en, scores_lr,
        plot_dir / "umap_linear_probe_loso.png",
    )

    # Score distribution comparison plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-probing (zero-shot)
    en_pred_df = pd.read_csv(Path("research/research_results/CSV/transfer_en/predictions_zero_shot_baseline.csv"))
    zs_scores = en_pred_df["score_patients"].values
    zs_labels = en_pred_df["label"].values

    for ax, scores, lbls, title in [
        (axes2[0], zs_scores, zs_labels, f"Zero-shot\nAUC={0.600:.3f}  bal_acc=0.500"),
        (axes2[1], scores_lr, y_true, f"Linear Probe (LOSO)\nAUC={metrics_lr['auc']:.3f}  bal_acc={metrics_lr['balanced_accuracy']:.3f}"),
    ]:
        healthy = scores[lbls == 0]
        patients = scores[lbls == 1]
        bins = np.linspace(0.0, 1.0, 26)
        ax.hist(healthy, bins=bins, alpha=0.6, color="steelblue", label=f"Healthy (n={len(healthy)})")
        ax.hist(patients, bins=bins, alpha=0.6, color="crimson", label=f"Patients (n={len(patients)})")
        ax.axvline(x=0.5, color="green", linestyle="--", linewidth=2, label="Threshold 0.5")
        ax.set_xlabel("score_patients")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(0.0, 1.0)

    fig2.suptitle("Score Distribution: Zero-shot vs Linear Probing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    score_dist_path = plot_dir / "score_distribution_comparison.png"
    fig2.savefig(score_dist_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Per-subject details
    subject_details = []
    for i, sid in enumerate(test_subjects):
        subject_details.append({
            "subject_id": sid,
            "label": int(y_true[i]),
            "zero_shot_score": float(zs_scores[i]),
            "lr_probe_score": float(scores_lr[i]),
            "mlp_probe_score": float(mlp_all_scores[i]) if i < len(mlp_all_scores) else None,
            "lr_pred": int(scores_lr[i] >= 0.5),
            "mlp_pred": int(mlp_all_preds[i]) if i < len(mlp_all_preds) else None,
            "lr_correct": int(scores_lr[i] >= 0.5) == int(y_true[i]),
            "mlp_correct": int(mlp_all_preds[i]) if i < len(mlp_all_preds) else None == int(y_true[i]) if i < len(mlp_all_preds) else None,
        })

    # Build comprehensive report
    zero_shot_report = json.loads((output_dir / "diagnostic_audit_report.json").read_text(encoding="utf-8"))

    report = {
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "probe_type": "Logistic Regression (L2, class_weight=balanced, inner LOSO C-selection)",
        "zero_shot_reference": {
            "auc": zero_shot_report["zero_shot_performance"]["auc"],
            "best_threshold_bal_acc": zero_shot_report["zero_shot_performance"]["best_balanced_accuracy"],
            "best_threshold": zero_shot_report["zero_shot_performance"]["best_english_threshold"],
        },
        "linear_probe_loso": {
            "method": "LogisticRegression LOSO with inner LOSO C-selection",
            **{k: round(v, 6) if v is not None else None for k, v in metrics_lr.items()},
            "folds": fold_results,
            "final_probe_C": final_C,
        },
        "mlp_classifier_probe_loso": {
            "method": "Original 2-layer MLP classifier, LOSO",
            "arch": ckpt["model_config"],
            **{k: round(v, 6) if v is not None else None for k, v in metrics_mlp.items()},
            "folds": mlp_fold_results,
        },
        "per_subject": subject_details,
        "improvement_over_zero_shot": {
            "linear_probe_bal_acc_gain": round(metrics_lr["balanced_accuracy"] - 0.70, 4),
            "linear_probe_vs_zero_shot_bal_acc": round(metrics_lr["balanced_accuracy"] - 0.50, 4),
        },
    }

    report_path = output_dir / "linear_probe_report.json"
    subject_path = output_dir / "linear_probe_per_subject.csv"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(subject_details).to_csv(subject_path, index=False)

    print(f"\n===== PER-SUBJECT COMPARISON =====")
    print(f"{'Subj':>6s} {'Label':>5s} {'ZS_score':>9s} {'LR_score':>9s} {'MLP_score':>9s} {'LR_OK':>5s} {'MLP_OK':>5s}")
    for d in subject_details:
        zs = f"{d['zero_shot_score']:.4f}"
        lr = f"{d['lr_probe_score']:.4f}"
        mlp = f"{d['mlp_probe_score']:.4f}" if d['mlp_probe_score'] is not None else "N/A"
        print(f"{d['subject_id']:>6s} {d['label']:>5d} {zs:>9s} {lr:>9s} {mlp:>9s} {str(d['lr_correct']):>5s} {str(d['mlp_correct']):>5s}")

    print(f"\n===== COMPARISON =====")
    print(f"Zero-shot (best threshold):   bal_acc=0.7000  AUC=0.600")
    print(f"Linear Probe (LOSO LR):       bal_acc={metrics_lr['balanced_accuracy']:.4f}  AUC={metrics_lr['auc']:.4f}")
    print(f"MLP Classifier Probe (LOSO):  bal_acc={metrics_mlp['balanced_accuracy']:.4f}  AUC={metrics_mlp['auc']:.4f}")

    print(f"\nReport:  {report_path}")
    print(f"Subjects: {subject_path}")
    print(f"Plots:   {plot_dir}")


if __name__ == "__main__":
    main()
