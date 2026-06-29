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
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                              f1_score, precision_score, recall_score, roc_auc_score)

OUT_CSV = Path("research/research_results/CSV")
OUT_PLOT = Path("research/research_results/attention_viz/final_comparison")
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
pred = pd.read_csv("research/research_results/CSV/predictions_v3_layermix.csv")
tax = pd.read_csv("research/research_results/CSV/pronunciation_audit/pronunciation_master_table.csv")

pred = pred.merge(tax[["sample_id", "task_type", "syllable_group"]], on="sample_id", how="left")
for c in ["task_type", "syllable_group"]:
    pred[c] = pred[c].fillna("unknown")

subj = pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
subj["label_num"] = (subj["label"].astype(float) == 1.0).astype(int)
subj = subj.merge(pred[["subject_id", "task_type", "syllable_group"]].drop_duplicates(subset="subject_id"), on="subject_id", how="left")

print(f"Samples: {len(pred)}, Subjects: {len(subj)}")
print(f"Labels: {subj['label'].value_counts().to_dict()}")

# ── 1. Calibration methods ────────────────────────────────────
yu = subj["label_num"].values.astype(float)
su = subj["score_patients"].values.astype(float).reshape(-1, 1)
su_flat = su.ravel()

# Platt scaling: logistic regression on raw scores
platt_lr = LogisticRegression(max_iter=2000)
platt_lr.fit(su, yu)
scores_platt = platt_lr.predict_proba(su)[:, 1]

# Temperature scaling
def temperature_scale(scores, temp):
    logits = np.log(np.clip(scores, 0.001, 0.999) / np.clip(1 - scores, 0.001, 0.999))
    return 1 / (1 + np.exp(-logits / temp))

best_temp, best_ba = 1.0, 0.0
for t in np.arange(0.2, 3.0, 0.1):
    sp = temperature_scale(su_flat, t)
    ba = float(balanced_accuracy_score(yu, (sp >= 0.5).astype(int)))
    if ba > best_ba: best_ba = ba; best_temp = t
scores_temp = temperature_scale(su_flat, best_temp)

# Isotonic regression
iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
scores_iso = iso.fit_transform(su_flat, yu)

def cal_metrics(scores, labels, thr=0.5):
    p = (scores >= thr).astype(int)
    m = {
        "threshold": thr,
        "accuracy": round(float(accuracy_score(labels, p)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, p)), 4),
        "precision": round(float(precision_score(labels, p, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, p, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, p, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(labels, np.clip(scores, 0, 1))), 4),
    }
    try: m["auc"] = round(float(roc_auc_score(labels, scores)), 4)
    except: m["auc"] = None
    return m

def best_thr_constrained(scores, labels, min_recall=0.95):
    best = None
    for t in np.arange(0.01, 1.0, 0.01):
        m = cal_metrics(scores, labels, thr=round(t, 4))
        if m["recall"] >= min_recall:
            if best is None or m["precision"] > best["precision"]: best = m
    return best if best else cal_metrics(scores, labels, thr=0.5)

orig_best = best_thr_constrained(su_flat, yu)
platt_best = best_thr_constrained(scores_platt, yu)
temp_best = best_thr_constrained(scores_temp, yu)
iso_best = best_thr_constrained(scores_iso, yu)

# ── 2. Sub-task thresholds ────────────────────────────────────
subj_syl = pred.groupby(["subject_id", "label", "syllable_group"], as_index=False)["score_patients"].mean()
subj_scores = subj_syl.pivot_table(index=["subject_id", "label"], columns="syllable_group", values="score_patients").reset_index()
syl_cols = [c for c in subj_scores.columns if c not in ("subject_id", "label")]

# Find best threshold per syllable group
syl_thresholds = {}
for sc in syl_cols:
    gs = pred[pred["syllable_group"] == sc]
    if len(gs) < 10: continue
    yg = (gs["label"] == "Patients").astype(int).values
    sg = gs["score_patients"].values
    best = None
    for t in np.arange(0.01, 1.0, 0.01):
        m = cal_metrics(sg, yg, thr=round(t, 4))
        if m["recall"] >= 0.95:
            if best is None or m["precision"] > best["precision"]: best = m
    if best: syl_thresholds[sc] = best["threshold"]

# Apply per-syllable thresholds to subjects
subj["pred_us"] = su_flat >= orig_best["threshold"]
subj["pred_platt"] = scores_platt >= platt_best["threshold"]
subj["pred_temp"] = scores_temp >= temp_best["threshold"]
subj["pred_iso"] = scores_iso >= iso_best["threshold"]

# Per-syllable voting
preds_syl = []
for _, r in subj.iterrows():
    sid = r["subject_id"]
    ss = pred[pred["subject_id"] == sid]
    votes = []
    for sc_name, thr_s in syl_thresholds.items():
        gs = ss[ss["syllable_group"] == sc_name]
        if len(gs) > 0:
            votes.append(int(gs["score_patients"].mean() >= thr_s))
    preds_syl.append(int(np.mean(votes) >= 0.5) if votes else int(su_flat[len(preds_syl)] >= orig_best["threshold"]))
subj["pred_syl"] = preds_syl

def subj_metrics(pred_col):
    yt = subj["label_num"].values.astype(float)
    yp = subj[pred_col].values.astype(int)
    sc = subj["score_patients"].values
    m = {
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
    }
    try: m["auc"] = round(float(roc_auc_score(yt, sc)), 4)
    except: m["auc"] = None
    return m

# ── 3. Hard-negative analysis ─────────────────────────────────
hn_data = []
for sid in ["H26", "H27", "H28", "H29"]:
    ss = pred[pred["subject_id"] == sid]
    ms = float(ss["score_patients"].mean())
    for sc in syl_cols:
        gs = ss[ss["syllable_group"] == sc]
        if len(gs) == 0: continue
        fp_orig = int((gs["score_patients"] >= orig_best["threshold"]).sum())
        fp_syl = int((gs["score_patients"] >= syl_thresholds.get(sc, 0.61)).sum())
        hn_data.append({
            "subject_id": sid, "syllable_group": sc, "n_samples": len(gs),
            "mean_score": round(float(gs["score_patients"].mean()), 4),
            "fp_orig_thr": fp_orig, "fp_orig_rate": round(fp_orig / max(len(gs), 1), 4),
            "fp_syl_thr": fp_syl, "fp_syl_rate": round(fp_syl / max(len(gs), 1), 4),
        })

hn_df = pd.DataFrame(hn_data)
hn_df.to_csv(OUT_CSV / "v3_layermix_hard_negative_analysis.csv", index=False)

# ── 4. Comparison table ───────────────────────────────────────
methods = {
    "Original (thr=0.20)": cal_metrics(su_flat, yu, thr=0.20),
    "Recalib (thr=0.61)": orig_best,
    "Platt": platt_best,
    "Temperature": temp_best,
    "Isotonic": iso_best,
    "Sub-task thresholds": subj_metrics("pred_syl"),
}

comparison_rows = []
for name, m in methods.items():
    comparison_rows.append({
        "strategy": name,
        "threshold": m.get("threshold", "N/A"),
        "subject_bal_acc": m.get("balanced_accuracy"),
        "subject_precision": m.get("precision"),
        "subject_recall": m.get("recall"),
        "subject_f1": m.get("f1"),
        "subject_auc": m.get("auc"),
        "brier": m.get("brier", None),
    })
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(OUT_CSV / "v3_layermix_postcal_comparison.csv", index=False)

# Save full metrics
final_metrics = {
    "config": {"arch": "V3_LayerMix_postcal", "original_threshold": 0.20, "recalibrated_threshold": 0.61},
    "best_temp": best_temp,
    "syllable_thresholds": syl_thresholds,
    "strategies": comp_df.to_dict(orient="records"),
    "hard_negatives": hn_df.to_dict(orient="records"),
}
(OUT_CSV / "metrics_v3_layermix_postcal.json").write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")

# ── 5. CHARTS ─────────────────────────────────────────────────

# Chart 1: Reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, scores, title in [
    (axes[0], su_flat, "Raw Scores"),
    (axes[1], scores_platt, "Platt Calibrated"),
]:
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(scores, bins) - 1
    frac_pos, mean_pred = [], []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.sum() >= 3:
            frac_pos.append(yu[mask].mean())
            mean_pred.append(scores[mask].mean())
        else:
            frac_pos.append(np.nan); mean_pred.append(np.nan)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.plot(mean_pred, frac_pos, "b.-", linewidth=2, markersize=8)
    ax.set_xlabel("Mean predicted"); ax.set_ylabel("Fraction positive")
    ax.set_title(title, fontweight="bold")
fig.suptitle("V3 LayerMix — Calibration Reliability", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_PLOT / "v3_layermix_postcal_reliability.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Chart 2: Post-calibration comparison
fig2, ax2 = plt.subplots(figsize=(12, 6))
names = list(methods.keys())
precisions = [m["precision"] for m in methods.values()]
recalls = [m["recall"] for m in methods.values()]
x = np.arange(len(names))
w = 0.35
ax2.bar(x - w/2, recalls, w, label="Recall", color="#66c2a5", edgecolor="black")
ax2.bar(x + w/2, precisions, w, label="Precision", color="#fc8d62", edgecolor="black")
for i, (r, p) in enumerate(zip(recalls, precisions)):
    ax2.text(i - w/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=7)
    ax2.text(i + w/2, p + 0.01, f"{p:.3f}", ha="center", fontsize=7)
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Score"); ax2.set_title("V3 LayerMix Post-Calibration Strategies", fontweight="bold")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT_PLOT / "v3_layermix_postcal_thresholds.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

# Chart 3: Hard negatives
fig3, ax3 = plt.subplots(figsize=(10, 6))
hn_summary = hn_df.groupby(["subject_id", "syllable_group"])["mean_score"].mean().reset_index()
hn_pivot = hn_summary.pivot(index="subject_id", columns="syllable_group", values="mean_score")
hn_pivot.round(4).plot(kind="bar", ax=ax3, edgecolor="black")
ax3.axhline(y=0.61, color="red", linestyle="--", linewidth=2, label="thr=0.61")
ax3.set_title("Hard Negatives — Mean Score by Syllable Group", fontweight="bold")
ax3.set_ylabel("Mean score"); ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(OUT_PLOT / "v3_layermix_postcal_hard_negatives.png", dpi=200, bbox_inches="tight")
plt.close(fig3)

# ── Print summary ─────────────────────────────────────────────
print(f"\n===== Post-Calibration Results =====")
print(f"\n{'Strategy':25s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'BalAcc':>7s} {'Brier':>7s}")
for name, m in methods.items():
    print(f"{name:25s} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['balanced_accuracy']:>7.4f} {m.get('brier','N/A'):>7}")

print(f"\nBest temperature: {best_temp:.1f}")
print(f"Syllable thresholds: {syl_thresholds}")
print(f"\nHard negatives summary:")
hn_agg = hn_df.groupby("subject_id")[["mean_score", "fp_orig_rate", "fp_syl_rate"]].mean()
print(hn_agg.round(4).to_string())
print(f"\nOutputs: {OUT_CSV}")
print(f"Plots:   {OUT_PLOT}")
