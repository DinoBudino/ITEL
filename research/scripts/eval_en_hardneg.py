from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings

try: import umap as umap_lib; HAS_UMAP = True
except ImportError: HAS_UMAP = False

from research.src.sequence_model import AttentionBiLSTMClassifierV3
from research.src.config import CONFIG

CKPT = "research/research_results/checkpoints/v3_layermix_hardneg/v3_layermix_hardneg_hnweight2x_fold1_of_10.pt"
EN_SEQ = "research/processed_data/manifests/hubert_sequences_en_test_all_layers.csv"
OUT = Path("research/research_results/CSV/transfer_en")
PLOT = Path("research/research_results/attention_viz/transfer_en")
OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)


def load_model(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    mc = p["model_config"]
    model = AttentionBiLSTMClassifierV3(
        input_dim=mc["input_dim"], lstm_hidden=mc["lstm_hidden"], lstm_layers=mc["lstm_layers"],
        dropout=mc["dropout"], fc_hidden=mc["fc_hidden"], num_heads=mc["num_heads"],
        use_layer_mix=mc.get("use_layer_mix", True), num_hubert_layers=mc.get("num_hubert_layers", 12),
        embedding_noise_std=mc.get("embedding_noise_std", 0.0),
        attention_entropy_lambda=mc.get("attention_entropy_lambda", 0.0))
    model.load_state_dict(p["state_dict"]); model.eval()
    return model, mc, p.get("best_threshold", 0.5)


def _metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    yt = np.array(y_true, dtype=np.float64)
    m = {"accuracy": float(accuracy_score(yt, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(yt, y_pred)),
         "precision": float(precision_score(yt, y_pred, zero_division=0)), "recall": float(recall_score(yt, y_pred, zero_division=0)),
         "f1": float(f1_score(yt, y_pred, zero_division=0))}
    try: m["auc"] = float(roc_auc_score(yt, y_prob))
    except: m["auc"] = None
    return m


def best_thr(scores, labels, min_recall=0.0):
    best, best_m = 0.5, None
    for t in np.arange(0.05, 1.0, 0.05):
        m = _metrics(labels, scores, threshold=round(t, 2))
        if m["recall"] >= min_recall and (best_m is None or m["balanced_accuracy"] > best_m["balanced_accuracy"]):
            best, best_m = t, m
    return best, best_m


@torch.no_grad()
def predict_en(model, manifest, device):
    model.to(device); model.eval()
    probs, ctxs, labels, sids, subjs = [], [], [], [], []
    for row in manifest.itertuples(index=False):
        pld = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = pld["sequence"].float().unsqueeze(0).to(device)
        mask = pld["mask"].bool().unsqueeze(0).to(device)
        logits, ctx = model(seq, mask, return_context=True)
        probs.append(float(torch.sigmoid(logits).cpu().item()))
        ctxs.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        sids.append(row.sample_id); subjs.append(row.subject_id)
    return pd.DataFrame({"sample_id": sids, "subject_id": subjs, "label": labels, "score_patients": probs}), np.stack(ctxs, axis=0)


@torch.no_grad()
def extract_contexts(model, manifest, device):
    model.to(device); model.eval()
    ctxs, labels, subjs = [], [], []
    for row in manifest.itertuples(index=False):
        pld = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = pld["sequence"].float().unsqueeze(0).to(device)
        mask = pld["mask"].bool().unsqueeze(0).to(device)
        _, ctx = model(seq, mask, return_context=True)
        ctxs.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        subjs.append(str(row.subject_id))
    return np.stack(ctxs, axis=0), np.array(labels), subjs


def loso_lr(X, y, subjects):
    unique = sorted(set(subjects))
    scaler = StandardScaler()
    scores, preds, test_sids, test_y, folds = [], [], [], [], []
    for ts in unique:
        tr_mask = np.array([s != ts for s in subjects])
        te_mask = np.array([s == ts for s in subjects])
        Xtr, ytr = X[tr_mask], y[tr_mask]; Xte, yte = X[te_mask], y[te_mask]
        Xtrs = scaler.fit_transform(Xtr); Xtes = scaler.transform(Xte)
        clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced", random_state=42)
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); clf.fit(Xtrs, ytr)
        proba = clf.predict_proba(Xtes)
        pi = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        sc = float(proba[0, pi]); pr = int(sc >= 0.5)
        scores.append(sc); preds.append(pr); test_sids.append(ts); test_y.append(int(yte[0]))
        folds.append({"test_subject": ts, "label": int(yte[0]), "score": sc, "pred": pr})
    yt = np.array(test_y); ys = np.array(scores); yp = np.array(preds)
    return {"accuracy": float(accuracy_score(yt, yp)), "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)), "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auc": float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else None}, folds, ys, yt, test_sids


def compute_geometry(en_X, en_lbl, en_subj, en_scores, cz_X, cz_lbl):
    hc = cz_X[cz_lbl == 0.0].mean(axis=0)
    pc = cz_X[cz_lbl == 1.0].mean(axis=0)
    dh = np.linalg.norm(en_X - hc, axis=1)
    dp = np.linalg.norm(en_X - pc, axis=1)
    rows = []
    for i in range(len(en_X)):
        closer = "healthy" if dh[i] < dp[i] else "patients"
        geo_ok = (int(en_lbl[i]) == 0 and closer == "healthy") or (int(en_lbl[i]) == 1 and closer == "patients")
        rows.append({"subject_id": en_subj[i], "label": int(en_lbl[i]), "score": float(en_scores[i]),
                     "d_to_cz_healthy": round(float(dh[i]), 4), "d_to_cz_patients": round(float(dp[i]), 4),
                     "closer_to": closer, "geometry_matches": geo_ok})
    geo_acc = sum(1 for r in rows if r["geometry_matches"]) / max(len(rows), 1)
    return {"cz_centroid_distance": float(np.linalg.norm(hc - pc)),
            "en_healthy_to_cz_h_mean": float(np.mean(dh[en_lbl == 0.0])) if (en_lbl == 0.0).sum() > 0 else None,
            "en_healthy_to_cz_p_mean": float(np.mean(dp[en_lbl == 0.0])) if (en_lbl == 0.0).sum() > 0 else None,
            "en_patients_to_cz_h_mean": float(np.mean(dh[en_lbl == 1.0])) if (en_lbl == 1.0).sum() > 0 else None,
            "en_patients_to_cz_p_mean": float(np.mean(dp[en_lbl == 1.0])) if (en_lbl == 1.0).sum() > 0 else None,
            "geometry_accuracy": round(geo_acc, 4)}, rows


def plot_umap(cz_X, cz_lbl, en_X, en_lbl, en_subj, en_scores, path, tag):
    Xall = np.vstack([cz_X, en_X])
    sc = StandardScaler(); Xsc = sc.fit_transform(Xall)
    if HAS_UMAP: X2d = umap_lib.UMAP(n_neighbors=15, min_dist=0.15, n_components=2, random_state=42).fit_transform(Xsc)
    else:
        from sklearn.decomposition import PCA; from sklearn.manifold import TSNE
        X2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(Xall) - 1)).fit_transform(PCA(n_components=50, random_state=42).fit_transform(Xsc))
    nc = len(cz_X); cz2, en2 = X2d[:nc], X2d[nc:]
    h2, p2 = cz2[cz_lbl == 0.0].mean(axis=0), cz2[cz_lbl == 1.0].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax in axes:
        ax.scatter(cz2[cz_lbl == 0.0, 0], cz2[cz_lbl == 0.0, 1], c="steelblue", alpha=0.06, s=5, zorder=1)
        ax.scatter(cz2[cz_lbl == 1.0, 0], cz2[cz_lbl == 1.0, 1], c="crimson", alpha=0.06, s=5, zorder=1)
        ax.scatter(h2[0], h2[1], marker="X", s=300, edgecolors="darkblue", facecolors="skyblue", linewidth=3, zorder=10, label="CZ Healthy")
        ax.scatter(p2[0], p2[1], marker="X", s=300, edgecolors="darkred", facecolors="salmon", linewidth=3, zorder=10, label="CZ Patient")
    axes[0].scatter(en2[en_lbl == 0.0, 0], en2[en_lbl == 0.0, 1], c="cyan", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="o", label="EN Healthy", zorder=5)
    axes[0].scatter(en2[en_lbl == 1.0, 0], en2[en_lbl == 1.0, 1], c="magenta", alpha=0.95, s=140, edgecolors="black", linewidth=1.5, marker="s", label="EN Patients", zorder=5)
    axes[0].set_title(f"{tag}: Labels", fontsize=12, fontweight="bold"); axes[0].legend(markerscale=1.2, fontsize=7, loc="upper right")
    sc2 = axes[1].scatter(en2[:, 0], en2[:, 1], c=en_scores, cmap="RdYlGn_r", s=160, edgecolors="black", linewidth=1.2, vmin=0.0, vmax=1.0, zorder=5)
    for i, (sid, x, y, sv) in enumerate(zip(en_subj, en2[:, 0], en2[:, 1], en_scores)):
        axes[1].annotate(f"{sid}\n{sv:.3f}", (x, y), fontsize=5.5, fontweight="bold", ha="center", va="bottom")
    plt.colorbar(sc2, ax=axes[1], shrink=0.75).set_label("Score", fontsize=9)
    axes[1].set_title(f"{tag}: Scores", fontsize=12, fontweight="bold")
    fig.suptitle(f"{tag} — Geometry Audit", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_histogram(scores, labels, thr, path, tag, auc_val):
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(scores[labels == 0.0], bins=bins, alpha=0.6, color="steelblue", label=f"Healthy (n={int((labels == 0.0).sum())})")
    ax.hist(scores[labels == 1.0], bins=bins, alpha=0.6, color="crimson", label=f"Patients (n={int((labels == 1.0).sum())})")
    ax.axvline(x=thr, color="green", linestyle="--", linewidth=2.5, label=f"Best thr={thr:.4f}")
    ax.set_xlabel("score_patients"); ax.set_ylabel("Count")
    ax.set_title(f"{tag}\nAUC={auc_val:.4f} best_thr={thr:.4f}", fontsize=13, fontweight="bold")
    ax.legend(); ax.set_xlim(0, 1)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# === MAIN ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"\nLoading: {CKPT}")
model, mc, ckpt_thr = load_model(CKPT)
model.to(device)
print(f"  use_layer_mix={mc.get('use_layer_mix')}  threshold={ckpt_thr:.3f}")

manifest = pd.read_csv(EN_SEQ)
print(f"English samples: {len(manifest)}  Labels: {manifest['label'].value_counts().to_dict()}")

# ZERO-SHOT
print("\n=== Zero-shot ===")
pred_frame, emb_en = predict_en(model, manifest, device)
best_t, best_m = best_thr(pred_frame["score_patients"].values, pred_frame["label"].values)
subj_ag = pred_frame.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
subj_ag["pred_label"] = subj_ag["score_patients"].apply(lambda s: 1.0 if s >= best_t else 0.0)
subj_m = _metrics(subj_ag["label"], subj_ag["score_patients"], threshold=best_t)

pred_frame.to_csv(OUT / "predictions_zero_shot_v3_layermix_hardneg_en.csv", index=False)
edf = pd.DataFrame(emb_en); edf["sample_id"] = pred_frame["sample_id"].values
edf["subject_id"] = pred_frame["subject_id"].values; edf["label"] = pred_frame["label"].values
edf.to_parquet(OUT / "embeddings_zero_shot_v3_layermix_hardneg_en.parquet")

zs_report = {"checkpoint": CKPT, "ckpt_threshold": ckpt_thr, "best_en_threshold": best_t,
             "sample_raw": _metrics(pred_frame["label"], pred_frame["score_patients"], threshold=ckpt_thr),
             "sample_best": best_m, "subject_best": subj_m}
(OUT / "metrics_zero_shot_v3_layermix_hardneg_en.json").write_text(json.dumps(zs_report, indent=2, default=str), encoding="utf-8")

print(f"Raw (thr={ckpt_thr}): bal_acc={zs_report['sample_raw']['balanced_accuracy']:.4f} auc={zs_report['sample_raw']['auc']}")
print(f"Best (thr={best_t}): bal_acc={best_m['balanced_accuracy']:.4f} rec={best_m['recall']:.4f} prec={best_m['precision']:.4f}")
print(f"Subject: bal_acc={subj_m['balanced_accuracy']:.4f} rec={subj_m['recall']:.4f} prec={subj_m['precision']:.4f}")

# LINEAR PROBE
print("\n=== LOSO Linear Probe ===")
X_en, y_en, subjs_en = extract_contexts(model, manifest, device)
lp_m, lp_folds, lp_scores, lp_true, lp_sids = loso_lr(X_en, y_en, subjs_en)
lp_details = [{"subject_id": sid, "label": int(lp_true[i]), "score": float(lp_scores[i]),
               "pred": int(lp_scores[i] >= 0.5), "correct": int(lp_scores[i] >= 0.5) == int(lp_true[i])}
              for i, sid in enumerate(lp_sids)]
(OUT / "linear_probe_v3_layermix_hardneg_en_report.json").write_text(
    json.dumps({"lp_metrics": lp_m, "folds": lp_folds, "per_subject": lp_details}, indent=2, default=str), encoding="utf-8")
pd.DataFrame(lp_details).to_csv(OUT / "linear_probe_v3_layermix_hardneg_en_per_subject.csv", index=False)
print(f"LP: bal_acc={lp_m['balanced_accuracy']:.4f} auc={lp_m['auc']} rec={lp_m['recall']:.4f} prec={lp_m['precision']:.4f}")

# GEOMETRY
print("\n=== Geometry ===")
cz_emb_df = pd.read_parquet("research/research_results/CSV/embeddings_v3_noise0.03.parquet")
cz_ecols = [c for c in cz_emb_df.columns if str(c).isdigit()]
cz_X = cz_emb_df[cz_ecols].astype(np.float64).values
cz_lbl = cz_emb_df["label"].astype(float).values
geo, geo_rows = compute_geometry(emb_en, pred_frame["label"].values, pred_frame["subject_id"].astype(str).values,
                                  pred_frame["score_patients"].values, cz_X, cz_lbl)

tag = "V3_LayerMix_hardneg"
(OUT / f"diagnostic_audit_report_{tag}_en.json").write_text(json.dumps(geo, indent=2, default=str), encoding="utf-8")
pd.DataFrame(geo_rows).to_csv(OUT / f"diagnostic_per_subject_{tag}_en.csv", index=False)

plot_umap(cz_X, cz_lbl, emb_en, pred_frame["label"].values, pred_frame["subject_id"].astype(str).values,
          pred_frame["score_patients"].values, PLOT / f"umap_zero_shot_cz_en_{tag}_en.png", tag)
plot_histogram(pred_frame["score_patients"].values, pred_frame["label"].values, best_t,
               PLOT / f"score_histogram_en_{tag}_en.png", tag, best_m["auc"])

print(f"Geometry accuracy: {geo['geometry_accuracy']:.4f}")
print(f"EN Healthy->CZ H: {geo['en_healthy_to_cz_h_mean']:.4f}  EN Healthy->CZ P: {geo['en_healthy_to_cz_p_mean']:.4f}")
print(f"EN Patients->CZ H: {geo['en_patients_to_cz_h_mean']:.4f}  EN Patients->CZ P: {geo['en_patients_to_cz_p_mean']:.4f}")
for r in geo_rows:
    print(f"  {r['subject_id']:>6s} lab={r['label']} score={r['score']:.4f} d_H={r['d_to_cz_healthy']:.3f} d_P={r['d_to_cz_patients']:.3f} closer={r['closer_to']} geo={r['geometry_matches']}")

print(f"\nOutputs: {OUT}")
print(f"Plots:   {PLOT}")
