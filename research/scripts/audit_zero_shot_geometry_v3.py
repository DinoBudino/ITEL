from __future__ import annotations

import argparse, json
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
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from research.src.config import CONFIG

TRANSFER_DIR = CONFIG.paths.reports_dir / "transfer_en"
PLOT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "transfer_en"


def load_embedding_cols(df):
    cols = [c for c in df.columns if str(c).isdigit()]
    return df[cols].astype(np.float64).values, df["label"].astype(float).values, df["subject_id"].astype(str).values


def centroid_dist_report(cz_X, cz_labels, en_X, en_labels, en_subjects, en_scores):
    h_c = cz_X[cz_labels == 0.0].mean(axis=0)
    p_c = cz_X[cz_labels == 1.0].mean(axis=0)
    cz_dist = float(np.linalg.norm(h_c - p_c))
    d_h = np.linalg.norm(en_X - h_c, axis=1)
    d_p = np.linalg.norm(en_X - p_c, axis=1)

    rows = []
    for i in range(len(en_X)):
        sid, lab, sc, dh, dp = en_subjects[i], int(en_labels[i]), float(en_scores[i]), float(d_h[i]), float(d_p[i])
        closer = "healthy" if dh < dp else "patients"
        geo_ok = (lab == 0 and closer == "healthy") or (lab == 1 and closer == "patients")
        rows.append({"subject_id": sid, "label": lab, "score": sc, "d_to_cz_healthy": round(dh, 4),
                     "d_to_cz_patients": round(dp, 4), "closer_to": closer, "geometry_matches": geo_ok})

    geo_acc = sum(1 for r in rows if r["geometry_matches"]) / max(len(rows), 1)
    return {
        "cz_centroid_distance": cz_dist,
        "en_healthy_to_cz_healthy_mean": float(np.mean(d_h[en_labels == 0.0])) if (en_labels == 0.0).sum() > 0 else None,
        "en_healthy_to_cz_patients_mean": float(np.mean(d_p[en_labels == 0.0])) if (en_labels == 0.0).sum() > 0 else None,
        "en_patients_to_cz_healthy_mean": float(np.mean(d_h[en_labels == 1.0])) if (en_labels == 1.0).sum() > 0 else None,
        "en_patients_to_cz_patients_mean": float(np.mean(d_p[en_labels == 1.0])) if (en_labels == 1.0).sum() > 0 else None,
        "geometry_accuracy": round(geo_acc, 4),
    }, rows


def umap_plot(cz_X, cz_labels, en_X, en_labels, en_subjects, en_scores, out_path, tag):
    X_all = np.vstack([cz_X, en_X])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    if HAS_UMAP:
        X_2d = umap_lib.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        X_temp = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
        X_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all) - 1)).fit_transform(X_temp)
    n_cz = len(cz_X)
    cz_2d, en_2d = X_2d[:n_cz], X_2d[n_cz:]
    h_c, p_c = cz_2d[cz_labels == 0.0].mean(axis=0), cz_2d[cz_labels == 1.0].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax in axes:
        ax.scatter(cz_2d[cz_labels == 0.0, 0], cz_2d[cz_labels == 0.0, 1], c="steelblue", alpha=0.06, s=5, zorder=1)
        ax.scatter(cz_2d[cz_labels == 1.0, 0], cz_2d[cz_labels == 1.0, 1], c="crimson", alpha=0.06, s=5, zorder=1)
        ax.scatter(h_c[0], h_c[1], marker="X", s=300, edgecolors="darkblue", facecolors="skyblue", linewidth=3, zorder=10, label="CZ Healthy centroid")
        ax.scatter(p_c[0], p_c[1], marker="X", s=300, edgecolors="darkred", facecolors="salmon", linewidth=3, zorder=10, label="CZ Patient centroid")

    axes[0].scatter(en_2d[en_labels == 0.0, 0], en_2d[en_labels == 0.0, 1], c="cyan", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="o", label="EN Healthy", zorder=5)
    axes[0].scatter(en_2d[en_labels == 1.0, 0], en_2d[en_labels == 1.0, 1], c="magenta", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="s", label="EN Patients", zorder=5)
    axes[0].set_title(f"{tag}: True Labels", fontsize=12, fontweight="bold")
    axes[0].legend(markerscale=1.2, fontsize=7, loc="upper right")

    sc = axes[1].scatter(en_2d[:, 0], en_2d[:, 1], c=en_scores, cmap="RdYlGn_r", s=160, edgecolors="black", linewidth=1.2, vmin=0.0, vmax=1.0, zorder=5)
    for i, (sid, x, y, sv) in enumerate(zip(en_subjects, en_2d[:, 0], en_2d[:, 1], en_scores)):
        axes[1].annotate(f"{sid}\n{sv:.3f}", (x, y), fontsize=5.5, fontweight="bold", ha="center", va="bottom")
    plt.colorbar(sc, ax=axes[1], shrink=0.75).set_label("Score", fontsize=9)
    axes[1].set_title(f"{tag}: Scores", fontsize=12, fontweight="bold")

    fig.suptitle(tag, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def histogram_plot(scores, labels, thr, out_path, tag, auc_val):
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(scores[labels == 0.0], bins=bins, alpha=0.6, color="steelblue", label=f"Healthy (n={int((labels == 0.0).sum())})")
    ax.hist(scores[labels == 1.0], bins=bins, alpha=0.6, color="crimson", label=f"Patients (n={int((labels == 1.0).sum())})")
    ax.axvline(x=thr, color="green", linestyle="--", linewidth=2.5, label=f"Best EN threshold = {thr:.4f}")
    ax.set_xlabel("score_patients", fontsize=12); ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{tag}\nAUC={auc_val:.4f}  best_thr={thr:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0.0, 1.0)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def find_best_threshold(scores, labels):
    best_thr, best_ba = 0.5, -1.0
    best_metrics = None
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
    for thr in sorted(set(round(s, 4) for s in scores)):
        preds = (scores >= thr).astype(int)
        ba = float(balanced_accuracy_score(labels, preds))
        if ba > best_ba:
            best_ba = ba; best_thr = thr
            best_metrics = {"threshold": thr, "balanced_accuracy": ba, "accuracy": float(accuracy_score(labels, preds)),
                            "recall": float(recall_score(labels, preds, zero_division=0)),
                            "precision": float(precision_score(labels, preds, zero_division=0)),
                            "f1": float(f1_score(labels, preds, zero_division=0))}
    return best_thr, best_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Geometry audit V3: compare V3 LayerMix EN embeddings vs CZ centroids.")
    p.add_argument("--cz-embeddings", default=str(CONFIG.paths.reports_dir / "CSV" / "embeddings_v3_layermix.parquet"))
    p.add_argument("--en-embeddings", default=str(TRANSFER_DIR / "embeddings_zero_shot_v3_layermix.parquet"))
    p.add_argument("--en-predictions", default=str(TRANSFER_DIR / "predictions_zero_shot_v3_layermix.csv"))
    p.add_argument("--tag", default="V3_LayerMix")
    p.add_argument("--output-dir", default=str(TRANSFER_DIR))
    p.add_argument("--plot-dir", default=str(PLOT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    output_dir, plot_dir = Path(args.output_dir), Path(args.plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)

    cz_X, cz_lbl, cz_subj = load_embedding_cols(pd.read_parquet(args.cz_embeddings))
    en_X, en_lbl, en_subj = load_embedding_cols(pd.read_parquet(args.en_embeddings))
    en_pred = pd.read_csv(args.en_predictions)
    en_scores = en_pred["score_patients"].values

    geo, geo_rows = centroid_dist_report(cz_X, cz_lbl, en_X, en_lbl, en_subj, en_scores)
    best_thr, best_m = find_best_threshold(en_scores, en_lbl)

    from sklearn.metrics import roc_auc_score
    auc_val = float(roc_auc_score(en_lbl, en_scores)) if len(np.unique(en_lbl)) > 1 else None

    umap_tag = f"{args.tag} (n={len(en_X)})"
    umap_plot(cz_X, cz_lbl, en_X, en_lbl, en_subj, en_scores, plot_dir / f"umap_zero_shot_cz_en_{args.tag.lower()}.png", umap_tag)
    histogram_plot(en_scores, en_lbl, best_thr, plot_dir / f"score_histogram_en_{args.tag.lower()}.png",
                   f"{args.tag} Score Distribution (EN only)", auc_val)

    report = {
        "tag": args.tag,
        "datasets": {"cz_samples": int(len(cz_X)), "cz_healthy": int((cz_lbl == 0.0).sum()), "cz_patients": int((cz_lbl == 1.0).sum()),
                     "en_samples": int(len(en_X)), "en_healthy": int((en_lbl == 0.0).sum()), "en_patients": int((en_lbl == 1.0).sum())},
        "geometry": geo,
        "best_en_threshold": best_thr, "best_en_metrics": best_m, "auc": auc_val,
        "per_subject": geo_rows,
    }
    (output_dir / f"diagnostic_audit_report_{args.tag.lower()}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(geo_rows).to_csv(output_dir / f"diagnostic_per_subject_{args.tag.lower()}.csv", index=False)

    print(f"\n===== {args.tag} AUDIT =====")
    print(f"Datasets: CZ={len(cz_X)}  EN={len(en_X)}")
    print(f"Geometry accuracy: {geo['geometry_accuracy']:.4f}")
    print(f"AUC: {auc_val:.4f}  Best EN thr: {best_thr:.4f}  bal_acc: {best_m['balanced_accuracy']:.4f}")
    print(f"EN Healthy -> CZ Healthy: {geo['en_healthy_to_cz_healthy_mean']:.4f}  "
          f"EN Healthy -> CZ Patients: {geo['en_healthy_to_cz_patients_mean']:.4f}")
    print(f"EN Patients -> CZ Healthy: {geo['en_patients_to_cz_healthy_mean']:.4f}  "
          f"EN Patients -> CZ Patients: {geo['en_patients_to_cz_patients_mean']:.4f}")
    for r in geo_rows:
        print(f"  {r['subject_id']:>6s}  label={r['label']}  score={r['score']:.4f}  "
              f"d_H={r['d_to_cz_healthy']:.3f}  d_P={r['d_to_cz_patients']:.3f}  "
              f"closer={r['closer_to']}  geo_ok={r['geometry_matches']}")


if __name__ == "__main__":
    main()
