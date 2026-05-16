from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, falling back to PCA+t-SNE")

from research.src.config import CONFIG

OUTPUT_DIR = CONFIG.paths.reports_dir / "transfer_en"
PLOT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "transfer_en"


def load_embedding_columns(df):
    embed_cols = [c for c in df.columns if str(c).isdigit()]
    X = df[embed_cols].astype(np.float64).values
    labels = df["label"].astype(float).values
    subjects = df["subject_id"].astype(str).values
    return X, labels, subjects


def compute_centroids(X, labels):
    healthy_mask = labels == 0.0
    patients_mask = labels == 1.0
    h_center = X[healthy_mask].mean(axis=0) if healthy_mask.sum() > 0 else None
    p_center = X[patients_mask].mean(axis=0) if patients_mask.sum() > 0 else None
    return h_center, p_center


def distances_to_centroids(X, h_center, p_center):
    d_h = np.linalg.norm(X - h_center, axis=1) if h_center is not None else np.full(len(X), np.nan)
    d_p = np.linalg.norm(X - p_center, axis=1) if p_center is not None else np.full(len(X), np.nan)
    return d_h, d_p


def run_umap(czech_X, czech_labels, english_X, english_labels, output_path):
    X_all = np.vstack([czech_X, english_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    if HAS_UMAP:
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42)
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        reducer = PCA(n_components=50, random_state=42)
        X_temp = reducer.fit_transform(X_scaled)
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1))

    X_2d = reducer.fit_transform(X_scaled)

    n_cz = len(czech_X)
    cz_2d = X_2d[:n_cz]
    en_2d = X_2d[n_cz:]

    h_cz = cz_2d[czech_labels == 0.0].mean(axis=0)
    p_cz = cz_2d[czech_labels == 1.0].mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(cz_2d[czech_labels == 0.0, 0], cz_2d[czech_labels == 0.0, 1],
               c="steelblue", alpha=0.08, s=6, label="Czech Healthy", zorder=1)
    ax.scatter(cz_2d[czech_labels == 1.0, 0], cz_2d[czech_labels == 1.0, 1],
               c="crimson", alpha=0.08, s=6, label="Czech Patients", zorder=1)

    ax.scatter(en_2d[english_labels == 0.0, 0], en_2d[english_labels == 0.0, 1],
               c="cyan", alpha=0.9, s=120, edgecolors="black", linewidth=1.5,
               marker="o", label="EN Healthy", zorder=5)
    ax.scatter(en_2d[english_labels == 1.0, 0], en_2d[english_labels == 1.0, 1],
               c="magenta", alpha=0.9, s=120, edgecolors="black", linewidth=1.5,
               marker="s", label="EN Patients", zorder=5)

    ax.scatter(h_cz[0], h_cz[1], marker="X", s=350, edgecolors="darkblue",
               facecolors="skyblue", linewidth=3, zorder=10, label="CZ Healthy centroid")
    ax.scatter(p_cz[0], p_cz[1], marker="X", s=350, edgecolors="darkred",
               facecolors="salmon", linewidth=3, zorder=10, label="CZ Patient centroid")

    cz_to_cz = np.linalg.norm(h_cz - p_cz)

    for i, (sid, lab, x, y) in enumerate(zip(
        [str(s) for s in english_labels], english_labels, en_2d[:, 0], en_2d[:, 1]
    )):
        d_h = np.linalg.norm(np.array([x, y]) - h_cz)
        d_p = np.linalg.norm(np.array([x, y]) - p_cz)
        color = "cyan" if lab == 0.0 else "magenta"
        ax.annotate(str(i + 1), (x, y), fontsize=7, fontweight="bold",
                    ha="center", va="bottom", color=color)

    ax.set_title(f"UMAP: Czech (n={n_cz}) + English Zero-shot (n={len(english_X)})\n"
                 f"CZ centroid distance = {cz_to_cz:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=1.0, fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "cz_healthy_centroid_2d": h_cz.tolist(),
        "cz_patients_centroid_2d": p_cz.tolist(),
        "cz_centroid_distance_2d": float(cz_to_cz),
    }


def optimal_threshold_english(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)

    best_thr = 0.5
    best_ba = -1.0
    best_metrics = None
    sweep = []

    unique_scores = sorted(set(scores))
    candidates = list(np.arange(0.05, 1.0, 0.01)) + list(unique_scores)

    for thr in sorted(set(round(t, 4) for t in candidates)):
        preds = (scores >= thr).astype(int)
        ba = float(balanced_accuracy_score(labels, preds))
        row = {
            "threshold": thr,
            "balanced_accuracy": ba,
            "accuracy": float(accuracy_score(labels, preds)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
        sweep.append(row)
        if ba > best_ba:
            best_ba = ba
            best_thr = thr
            best_metrics = row

    return best_thr, best_metrics, sweep


def build_report(czech_emb_path, en_emb_path, en_pred_path, output_dir, plot_dir):
    cz_df = pd.read_parquet(czech_emb_path)
    en_df = pd.read_parquet(en_emb_path)
    en_pred_df = pd.read_csv(en_pred_path)

    cz_X, cz_labels, cz_subjects = load_embedding_columns(cz_df)
    en_X, en_labels, en_subjects = load_embedding_columns(en_df)
    en_scores = en_pred_df["score_patients"].values

    h_center_full, p_center_full = compute_centroids(cz_X, cz_labels)
    cz_centroid_dist = float(np.linalg.norm(h_center_full - p_center_full)) if h_center_full is not None and p_center_full is not None else None

    en_d_h, en_d_p = distances_to_centroids(en_X, h_center_full, p_center_full)

    best_thr, best_metrics, sweep = optimal_threshold_english(en_scores, en_labels)

    preds_at_best = (en_scores >= best_thr).astype(int)
    auc_val = float(roc_auc_score(en_labels, en_scores)) if len(np.unique(en_labels)) > 1 else None

    umap_info = run_umap(cz_X, cz_labels, en_X, en_labels,
                         plot_dir / "umap_zero_shot_cz_en.png")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    healthy_scores = en_scores[en_labels == 0.0]
    patients_scores = en_scores[en_labels == 1.0]

    bins = np.linspace(0.0, 1.0, 51)
    ax2.hist(healthy_scores, bins=bins, alpha=0.6, color="steelblue", label=f"Healthy (n={len(healthy_scores)})")
    ax2.hist(patients_scores, bins=bins, alpha=0.6, color="crimson", label=f"Patients (n={len(patients_scores)})")
    ax2.axvline(x=best_thr, color="green", linestyle="--", linewidth=2.5,
                label=f"Best EN threshold = {best_thr:.4f}")
    ax2.axvline(x=0.2, color="orange", linestyle=":", linewidth=2,
                label="CZ checkpoint threshold = 0.20")

    ax2.set_xlabel("score_patients", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Zero-shot Score Distribution on English\n"
                  f"AUC={auc_val:.4f}  best_bal_acc={best_metrics['balanced_accuracy']:.4f}  best_thr={best_thr:.4f}",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.0, 1.0)
    fig2.tight_layout()
    hist_path = plot_dir / "score_histogram_en.png"
    fig2.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    subject_rows = []
    for i in range(len(en_X)):
        sid = en_subjects[i]
        lab = int(en_labels[i])
        sc = float(en_scores[i])
        dh = float(en_d_h[i])
        dp = float(en_d_p[i])
        closer_to = "healthy" if dh < dp else "patients"
        correct_geo = (lab == 0 and closer_to == "healthy") or (lab == 1 and closer_to == "patients")
        pred_at_best = int(preds_at_best[i])
        subject_rows.append({
            "subject_id": sid,
            "label": lab,
            "score_patients": sc,
            "d_to_cz_healthy_centroid": round(dh, 4),
            "d_to_cz_patients_centroid": round(dp, 4),
            "closer_to_cz_centroid": closer_to,
            "geometry_matches_label": correct_geo,
            "pred_at_best_threshold": pred_at_best,
        })

    subject_df = pd.DataFrame(subject_rows)
    correct_geo_n = int(subject_df["geometry_matches_label"].sum())
    total_n = len(subject_df)

    report = {
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "datasets": {
            "czech_samples": int(len(cz_X)),
            "czech_healthy": int((cz_labels == 0.0).sum()),
            "czech_patients": int((cz_labels == 1.0).sum()),
            "english_samples": int(len(en_X)),
            "english_healthy": int((en_labels == 0.0).sum()),
            "english_patients": int((en_labels == 1.0).sum()),
        },
        "centroids": {
            "cz_healthy_centroid_norm": float(np.linalg.norm(h_center_full)),
            "cz_patients_centroid_norm": float(np.linalg.norm(p_center_full)),
            "cz_centroid_distance": cz_centroid_dist,
        },
        "english_distances_to_cz_centroids": {
            "healthy_en_to_cz_healthy_mean": float(np.mean(en_d_h[en_labels == 0.0])) if (en_labels == 0.0).sum() > 0 else None,
            "healthy_en_to_cz_patients_mean": float(np.mean(en_d_p[en_labels == 0.0])) if (en_labels == 0.0).sum() > 0 else None,
            "patients_en_to_cz_healthy_mean": float(np.mean(en_d_h[en_labels == 1.0])) if (en_labels == 1.0).sum() > 0 else None,
            "patients_en_to_cz_patients_mean": float(np.mean(en_d_p[en_labels == 1.0])) if (en_labels == 1.0).sum() > 0 else None,
            "geometry_accuracy": f"{correct_geo_n}/{total_n} = {correct_geo_n/total_n:.4f}",
        },
        "zero_shot_performance": {
            "auc": auc_val,
            "best_english_threshold": float(best_thr),
            "best_balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "best_recall": float(best_metrics["recall"]),
            "best_precision": float(best_metrics["precision"]),
            "best_f1": float(best_metrics["f1"]),
            "threshold_sweep": sweep,
        },
        "umap": umap_info,
        "per_subject_detail": subject_df.to_dict(orient="records"),
    }

    return report, subject_df


def parse_args():
    parser = argparse.ArgumentParser(description="Geometric + calibration audit for zero-shot English transfer.")
    parser.add_argument("--cz-embeddings", default=str(CONFIG.paths.reports_dir / "CSV" / "embeddings_contrastive_l0.1.parquet"))
    parser.add_argument("--en-embeddings", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en" / "embeddings_zero_shot_baseline.parquet"))
    parser.add_argument("--en-predictions", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en" / "predictions_zero_shot_baseline.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--plot-dir", default=str(PLOT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    report, subject_df = build_report(
        args.cz_embeddings, args.en_embeddings, args.en_predictions,
        output_dir, plot_dir,
    )

    report_path = output_dir / "diagnostic_audit_report.json"
    subject_path = output_dir / "diagnostic_per_subject.csv"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    subject_df.to_csv(subject_path, index=False)

    print(f"\n===== DIAGNOSTIC REPORT =====")
    print(f"Datasets: CZ={report['datasets']['czech_samples']}  EN={report['datasets']['english_samples']}")
    print(f"Centroids: CZ distance={report['centroids']['cz_centroid_distance']:.4f}")
    print(f"Geometry: {report['english_distances_to_cz_centroids']['geometry_accuracy']} correct")
    print(f"\nZero-shot AUC: {report['zero_shot_performance']['auc']:.4f}")
    print(f"Best EN threshold: {report['zero_shot_performance']['best_english_threshold']:.4f}")
    print(f"Best bal_acc: {report['zero_shot_performance']['best_balanced_accuracy']:.4f}")
    print(f"Best recall: {report['zero_shot_performance']['best_recall']:.4f}")
    print(f"Best precision: {report['zero_shot_performance']['best_precision']:.4f}")
    print(f"Best F1: {report['zero_shot_performance']['best_f1']:.4f}")

    print(f"\nPer-subject centroid distances:")
    print(f"{'Subj':>6s} {'Label':>5s} {'Score':>8s} {'d_CZ_H':>10s} {'d_CZ_P':>10s} {'Closer':>10s} {'GeoOK':>5s}")
    for row in report["per_subject_detail"]:
        print(f"{row['subject_id']:>6s} {row['label']:>5d} {row['score_patients']:>8.4f} "
              f"{row['d_to_cz_healthy_centroid']:>10.4f} {row['d_to_cz_patients_centroid']:>10.4f} "
              f"{row['closer_to_cz_centroid']:>10s} {str(row['geometry_matches_label']):>5s}")

    print(f"\nReport:  {report_path}")
    print(f"Subjects: {subject_path}")
    print(f"Plots:   {plot_dir}")


if __name__ == "__main__":
    main()
