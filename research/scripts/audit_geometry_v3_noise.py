from __future__ import annotations
import json, sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
try: import umap as umap_lib; HAS_UMAP = True
except ImportError: HAS_UMAP = False

TRANSFER = Path("research/research_results/transfer_en")
TRANSFER_CSV = Path("research/research_results/CSV/transfer_en")
PLOT = Path("research/research_results/attention_viz/transfer_en")

def load_emb(df):
    cols = [c for c in df.columns if str(c).isdigit()]
    return df[cols].astype(np.float64).values, df["label"].astype(float).values, df["subject_id"].astype(str).values

# load data
cz_df = pd.read_parquet("research/research_results/CSV/embeddings_v3_noise0.03.parquet")
en_df = pd.read_parquet(TRANSFER_CSV / "embeddings_zero_shot_v3_noise.parquet")
en_pred = pd.read_csv(TRANSFER_CSV / "predictions_zero_shot_v3_noise.csv")
cz_X, cz_lbl, cz_subj = load_emb(cz_df)
en_X, en_lbl, en_subj = load_emb(en_df)
en_scores = en_pred["score_patients"].values

# centroids
h_c = cz_X[cz_lbl == 0.0].mean(axis=0)
p_c = cz_X[cz_lbl == 1.0].mean(axis=0)
d_h = np.linalg.norm(en_X - h_c, axis=1)
d_p = np.linalg.norm(en_X - p_c, axis=1)

rows = []
for i in range(len(en_X)):
    sid, lab, sc, dhv, dpv = en_subj[i], int(en_lbl[i]), float(en_scores[i]), float(d_h[i]), float(d_p[i])
    closer = "healthy" if dhv < dpv else "patients"
    geo_ok = (lab == 0 and closer == "healthy") or (lab == 1 and closer == "patients")
    rows.append({"subject_id": sid, "label": lab, "score": sc, "d_to_cz_healthy": round(dhv,4),
                 "d_to_cz_patients": round(dpv,4), "closer_to": closer, "geometry_matches": geo_ok})

# compare with prior results
try:
    f2 = json.loads((TRANSFER / "diagnostic_audit_report.json").read_text(encoding="utf-8"))
    f2_geo = f2["english_distances_to_cz_centroids"]["geometry_accuracy"]
except: f2_geo = "N/A"
try:
    lm = json.loads((TRANSFER / "diagnostic_audit_report_v3_layermix.json").read_text(encoding="utf-8"))
    lm_geo = lm["geometry"]["geometry_accuracy"]
except: lm_geo = "N/A"

geo_acc = sum(1 for r in rows if r["geometry_matches"]) / max(len(rows), 1)

TRANSFER.mkdir(parents=True, exist_ok=True)
report = {
    "tag": "V3_Noise003",
    "datasets": {"cz_samples": int(len(cz_X)), "en_samples": int(len(en_X))},
    "centroids": {"cz_centroid_distance": float(np.linalg.norm(h_c - p_c))},
    "distances": {
        "en_healthy_to_cz_healthy_mean": float(np.mean(d_h[en_lbl == 0.0])) if (en_lbl == 0.0).sum() > 0 else None,
        "en_healthy_to_cz_patients_mean": float(np.mean(d_p[en_lbl == 0.0])) if (en_lbl == 0.0).sum() > 0 else None,
        "en_patients_to_cz_healthy_mean": float(np.mean(d_h[en_lbl == 1.0])) if (en_lbl == 1.0).sum() > 0 else None,
        "en_patients_to_cz_patients_mean": float(np.mean(d_p[en_lbl == 1.0])) if (en_lbl == 1.0).sum() > 0 else None,
    },
    "geometry_accuracy": round(geo_acc, 4),
    "comparison": {"fase2_geo_accuracy": f2_geo, "layermix_geo_accuracy": lm_geo},
    "per_subject": rows,
}
(TRANSFER / "diagnostic_audit_report_v3_noise.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
pd.DataFrame(rows).to_csv(TRANSFER / "diagnostic_per_subject_v3_noise.csv", index=False)

# UMAP
X_all = np.vstack([cz_X, en_X])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
if HAS_UMAP:
    X_2d = umap_lib.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42).fit_transform(X_scaled)
else:
    from sklearn.decomposition import PCA; from sklearn.manifold import TSNE
    X_temp = PCA(n_components=50, random_state=42).fit_transform(X_scaled)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all)-1)).fit_transform(X_temp)
n_cz = len(cz_X)
cz_2d, en_2d = X_2d[:n_cz], X_2d[n_cz:]
h_c2d, p_c2d = cz_2d[cz_lbl == 0.0].mean(axis=0), cz_2d[cz_lbl == 1.0].mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
for ax in axes:
    ax.scatter(cz_2d[cz_lbl == 0.0, 0], cz_2d[cz_lbl == 0.0, 1], c="steelblue", alpha=0.06, s=5, zorder=1)
    ax.scatter(cz_2d[cz_lbl == 1.0, 0], cz_2d[cz_lbl == 1.0, 1], c="crimson", alpha=0.06, s=5, zorder=1)
    ax.scatter(h_c2d[0], h_c2d[1], marker="X", s=300, edgecolors="darkblue", facecolors="skyblue", linewidth=3, zorder=10, label="CZ Healthy")
    ax.scatter(p_c2d[0], p_c2d[1], marker="X", s=300, edgecolors="darkred", facecolors="salmon", linewidth=3, zorder=10, label="CZ Patient")
axes[0].scatter(en_2d[en_lbl == 0.0, 0], en_2d[en_lbl == 0.0, 1], c="cyan", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="o", label="EN Healthy", zorder=5)
axes[0].scatter(en_2d[en_lbl == 1.0, 0], en_2d[en_lbl == 1.0, 1], c="magenta", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="s", label="EN Patients", zorder=5)
axes[0].set_title("V3 Noise 0.03: True Labels", fontsize=12, fontweight="bold"); axes[0].legend(markerscale=1.2, fontsize=7, loc="upper right")
sc = axes[1].scatter(en_2d[:, 0], en_2d[:, 1], c=en_scores, cmap="RdYlGn_r", s=160, edgecolors="black", linewidth=1.2, vmin=0.0, vmax=1.0, zorder=5)
for i, (sid, x, y, sv) in enumerate(zip(en_subj, en_2d[:, 0], en_2d[:, 1], en_scores)):
    axes[1].annotate(f"{sid}\n{sv:.3f}", (x, y), fontsize=5.5, fontweight="bold", ha="center", va="bottom")
plt.colorbar(sc, ax=axes[1], shrink=0.75).set_label("Score", fontsize=9)
axes[1].set_title("V3 Noise 0.03: Scores", fontsize=12, fontweight="bold")
fig.suptitle("V3 Noise 0.03 — Geometry Audit", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); PLOT.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOT / "umap_zero_shot_cz_en_v3_noise.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# histogram
fig2, ax2 = plt.subplots(figsize=(12, 6))
bins = np.linspace(0.0, 1.0, 26)
ax2.hist(en_scores[en_lbl == 0.0], bins=bins, alpha=0.6, color="steelblue", label=f"Healthy (n={int((en_lbl==0.0).sum())})")
ax2.hist(en_scores[en_lbl == 1.0], bins=bins, alpha=0.6, color="crimson", label=f"Patients (n={int((en_lbl==1.0).sum())})")
from sklearn.metrics import roc_auc_score
auc_v = float(roc_auc_score(en_lbl, en_scores)) if len(np.unique(en_lbl)) > 1 else None
ax2.set_title(f"V3 Noise 0.03 Score Distribution on English\nAUC={auc_v:.4f}", fontsize=13, fontweight="bold")
ax2.set_xlabel("score_patients"); ax2.set_ylabel("Count"); ax2.legend(); ax2.set_xlim(0, 1)
fig2.tight_layout()
fig2.savefig(PLOT / "score_histogram_en_v3_noise.png", dpi=200, bbox_inches="tight"); plt.close(fig2)

print(f"Geometry accuracy: {geo_acc:.4f} (Fase2={f2_geo}, LayerMix={lm_geo})")
print(f"EN Healthy -> CZ Healthy: {report['distances']['en_healthy_to_cz_healthy_mean']:.4f}")
print(f"EN Healthy -> CZ Patients: {report['distances']['en_healthy_to_cz_patients_mean']:.4f}")
print(f"EN Patients -> CZ Healthy: {report['distances']['en_patients_to_cz_healthy_mean']:.4f}")
print(f"EN Patients -> CZ Patients: {report['distances']['en_patients_to_cz_patients_mean']:.4f}")
for r in rows:
    print(f"  {r['subject_id']:>6s} label={r['label']} score={r['score']:.4f} d_H={r['d_to_cz_healthy']:.3f} d_P={r['d_to_cz_patients']:.3f} closer={r['closer_to']} geo_ok={r['geometry_matches']}")
print("\nDone.")
