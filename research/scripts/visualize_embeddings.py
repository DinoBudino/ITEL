from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    import umap as umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

embeddings_path = "research/research_results/CSV/embeddings_contrastive_l0.1.parquet"
output_dir = Path("research/research_results/attention_viz")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(embeddings_path)
print(f"Loaded {len(df)} embeddings")

embed_cols = [c for c in df.columns if c.isdigit()]
X = df[embed_cols].astype(np.float64).values
labels = df["label"].values
subjects = df["subject_id"].values

# Reduce to 2D
print("Reducing to 2D ...")
if HAS_UMAP:
    reducer = umap_module.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
else:
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    X = X_pca

X_2d = reducer.fit_transform(X)
print("Reduction done.")

CRITICAL = ["H26", "H27", "H28", "H29"]

# Healthy centroid and Patient centroid
healthy_mask = labels == 0.0
patients_mask = labels == 1.0
healthy_center = X_2d[healthy_mask].mean(axis=0)
patients_center = X_2d[patients_mask].mean(axis=0)

# Distances for critical subjects
print("\n=== DISTANCES TO CENTROIDS ===")
for s in CRITICAL:
    sm = df["subject_id"] == s
    if sm.sum() > 0:
        points = X_2d[sm.values]
        center = points.mean(axis=0)
        d_healthy = float(np.linalg.norm(center - healthy_center))
        d_patients = float(np.linalg.norm(center - patients_center))
        ratio = d_healthy / max(d_patients, 1e-8)
        verdict = "Healthy side" if d_healthy < d_patients else "Patient side"
        print(f"  {s}: d_healthy={d_healthy:.3f}  d_patients={d_patients:.3f}  ratio={ratio:.3f}  -> {verdict}")

# Overall healthy/patient centroids for reference
print(f"\nHealthy centroid distance to Patient centroid: {np.linalg.norm(healthy_center - patients_center):.3f}")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

h_m = healthy_mask
p_m = patients_mask
ax.scatter(X_2d[h_m, 0], X_2d[h_m, 1], c="steelblue", alpha=0.15, s=8, label="Healthy")
ax.scatter(X_2d[p_m, 0], X_2d[p_m, 1], c="crimson", alpha=0.15, s=8, label="Patients")

ax.scatter(healthy_center[0], healthy_center[1], marker="X", s=200, edgecolors="darkblue", facecolors="cyan", linewidth=2, zorder=10, label="Healthy centroid")
ax.scatter(patients_center[0], patients_center[1], marker="X", s=200, edgecolors="darkred", facecolors="orange", linewidth=2, zorder=10, label="Patient centroid")

colors_crit = {"H26": "magenta", "H27": "lime", "H28": "gold", "H29": "orange"}
for s in CRITICAL:
    sm = df["subject_id"] == s
    if sm.sum() > 0:
        pts = X_2d[sm.values]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors_crit[s], alpha=0.7, s=35, edgecolors="black", linewidth=0.5, label=f"{s} ({sm.sum()})", zorder=5)
        # center of subject
        ctr = pts.mean(axis=0)
        ax.scatter(ctr[0], ctr[1], marker="D", s=80, edgecolors="black", facecolors=colors_crit[s], linewidth=1.5, zorder=6)
        ax.annotate(s, ctr, fontsize=9, fontweight="bold", ha="center", va="bottom", color=colors_crit[s])

ax.set_title("Embedding space (BCE + SupCon, lambda=0.1)\nCritical subjects highlighted", fontsize=12)
ax.legend(markerscale=1.5, fontsize=7, loc="upper right")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")

plt.tight_layout()
plot_path = output_dir / "umap_contrastive_l0.1.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved: {plot_path}")
