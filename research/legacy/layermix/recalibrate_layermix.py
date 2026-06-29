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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)

from research.src.config import CONFIG

OUT_CSV = CONFIG.paths.reports_dir / "CSV"
OUT_PLOT = CONFIG.paths.reports_dir / "attention_viz" / "final_comparison"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

# ── Load V3 LayerMix predictions ─────────────────────────────
pred = pd.read_csv("research/research_results/CSV/predictions_v3_layermix.csv")
print(f"V3 LayerMix predictions: {len(pred)} samples, {pred['subject_id'].nunique()} subjects")

# ── Fine threshold sweep ─────────────────────────────────────
thresholds = np.arange(0.05, 1.0, 0.01).round(4).tolist()
sample_rows, subject_rows = [], []

for thr in thresholds:
    pred["pred_num"] = (pred["score_patients"] >= thr).astype(int)
    yt = pred["label"].astype(float).values
    yp = pred["pred_num"].values
    ys = pred["score_patients"].values

    srow = {"threshold": thr,
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "fp_count": int((pred["pred_num"] == 1).sum() - (yt == 1).sum())}
    try: srow["auc"] = round(float(roc_auc_score(yt, ys)), 4)
    except: srow["auc"] = None
    sample_rows.append(srow)

    subj = pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj["pred_num"] = (subj["score_patients"] >= thr).astype(int)
    yu = subj["label"].astype(float).values
    ypu = subj["pred_num"].values
    ysu = subj["score_patients"].values

    urow = {"threshold": thr,
            "accuracy": round(float(accuracy_score(yu, ypu)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yu, ypu)), 4),
            "precision": round(float(precision_score(yu, ypu, zero_division=0)), 4),
            "recall": round(float(recall_score(yu, ypu, zero_division=0)), 4),
            "f1": round(float(f1_score(yu, ypu, zero_division=0)), 4),
            "fp_subjects": int((subj["pred_num"] == 1).sum() - (yu == 1).sum())}
    try: urow["auc"] = round(float(roc_auc_score(yu, ysu)), 4)
    except: urow["auc"] = None
    subject_rows.append(urow)

sweep_df = pd.DataFrame(sample_rows)
subj_sweep_df = pd.DataFrame(subject_rows)
sweep_df.to_csv(OUT_CSV / "threshold_sweep_v3_layermix.csv", index=False)
subj_sweep_df.to_csv(OUT_CSV / "subject_level_threshold_tradeoff_v3_layermix.csv", index=False)

# ── Find best candidates ─────────────────────────────────────
current_thr = 0.20
current_s = sweep_df[sweep_df["threshold"] == current_thr].iloc[0]
current_u = subj_sweep_df[subj_sweep_df["threshold"] == current_thr].iloc[0]

best_bal_s = sweep_df.loc[sweep_df["balanced_accuracy"].idxmax()]
best_bal_u = subj_sweep_df.loc[subj_sweep_df["balanced_accuracy"].idxmax()]

best_f1_s = sweep_df.loc[sweep_df["f1"].idxmax()]
best_f1_u = subj_sweep_df.loc[subj_sweep_df["f1"].idxmax()]

constrained = subj_sweep_df[subj_sweep_df["recall"] >= 0.95].copy()
if len(constrained) > 0:
    best_prec_u = constrained.loc[constrained["precision"].idxmax()]
else:
    best_prec_u = best_f1_u

constrained_s = sweep_df[sweep_df["recall"] >= 0.95].copy()
if len(constrained_s) > 0:
    best_prec_s = constrained_s.loc[constrained_s["precision"].idxmax()]
else:
    best_prec_s = best_f1_s

# ── Comparison table ─────────────────────────────────────────
comparison = pd.DataFrame([
    {"policy": "Current (recall-max, thr=0.20)", "threshold": current_thr,
     "sample_bal_acc": current_s["balanced_accuracy"], "sample_recall": current_s["recall"],
     "sample_precision": current_s["precision"], "sample_f1": current_s["f1"],
     "subject_bal_acc": current_u["balanced_accuracy"], "subject_recall": current_u["recall"],
     "subject_precision": current_u["precision"], "subject_f1": current_u["f1"]},
    {"policy": "Best Balanced Accuracy", "threshold": best_bal_u["threshold"],
     "sample_bal_acc": best_bal_s["balanced_accuracy"], "sample_recall": best_bal_s["recall"],
     "sample_precision": best_bal_s["precision"], "sample_f1": best_bal_s["f1"],
     "subject_bal_acc": best_bal_u["balanced_accuracy"], "subject_recall": best_bal_u["recall"],
     "subject_precision": best_bal_u["precision"], "subject_f1": best_bal_u["f1"]},
    {"policy": "Best F1", "threshold": best_f1_u["threshold"],
     "sample_bal_acc": best_f1_s["balanced_accuracy"], "sample_recall": best_f1_s["recall"],
     "sample_precision": best_f1_s["precision"], "sample_f1": best_f1_s["f1"],
     "subject_bal_acc": best_f1_u["balanced_accuracy"], "subject_recall": best_f1_u["recall"],
     "subject_precision": best_f1_u["precision"], "subject_f1": best_f1_u["f1"]},
    {"policy": "Best Precision (recall>=0.95)", "threshold": best_prec_u["threshold"],
     "sample_bal_acc": best_prec_s["balanced_accuracy"], "sample_recall": best_prec_s["recall"],
     "sample_precision": best_prec_s["precision"], "sample_f1": best_prec_s["f1"],
     "subject_bal_acc": best_prec_u["balanced_accuracy"], "subject_recall": best_prec_u["recall"],
     "subject_precision": best_prec_u["precision"], "subject_f1": best_prec_u["f1"]},
])
comparison.to_csv(OUT_CSV / "v3_layermix_recalibration_comparison.csv", index=False)

# ── Critical subjects ────────────────────────────────────────
critical = pred[pred["subject_id"].isin(["H26", "H27", "H28", "H29"])]
critical_data = []
for sid in ["H26", "H27", "H28", "H29"]:
    ss = critical[critical["subject_id"] == sid]
    mean_score = float(ss["score_patients"].mean())
    n_total = len(ss)

    for policy_name, thr_val in [("Current", current_thr), ("Best BalAcc", best_bal_u["threshold"]),
                                  ("Best F1", best_f1_u["threshold"]), ("Best Prec", best_prec_u["threshold"])]:
        fp_at_thr = int((ss["score_patients"] >= thr_val).sum())
        critical_data.append({"subject_id": sid, "policy": policy_name, "threshold": thr_val,
                              "mean_score": round(mean_score, 4), "n_total": n_total,
                              "fp": fp_at_thr, "fp_rate": round(fp_at_thr / max(n_total, 1), 4)})

crit_df = pd.DataFrame(critical_data)
crit_df.to_csv(OUT_CSV / "v3_layermix_critical_subjects_recalibration.csv", index=False)

# ── Final metrics JSON ───────────────────────────────────────
best_policy = comparison.iloc[-1]
final_metrics = {
    "config": {"arch": "V3_LayerMix", "recalibration": "threshold_optimization",
               "original_threshold": current_thr, "best_policy": best_policy["policy"],
               "recommended_threshold": best_policy["threshold"]},
    "sample_level": {"original": current_s.to_dict(), "recalibrated": best_prec_s.to_dict()},
    "subject_level": {"original": current_u.to_dict(), "recalibrated": best_prec_u.to_dict()},
    "improvements": {
        "sample_precision_delta": round(best_prec_s["precision"] - current_s["precision"], 4),
        "subject_precision_delta": round(best_prec_u["precision"] - current_u["precision"], 4),
        "sample_recall_delta": round(best_prec_s["recall"] - current_s["recall"], 4),
        "subject_recall_delta": round(best_prec_u["recall"] - current_u["recall"], 4),
    },
    "threshold_comparison": comparison.to_dict(orient="records"),
    "critical_subjects_summary": crit_df.to_dict(orient="records"),
}
(OUT_CSV / "metrics_v3_layermix_recalibrated.json").write_text(
    json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")

# ── Chart 1: Threshold tradeoff ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, df, title in [(axes[0], sweep_df, "Sample-level"),
                       (axes[1], subj_sweep_df, "Subject-level")]:
    ax.plot(df["threshold"], df["recall"], "b-", linewidth=2, label="Recall")
    ax.plot(df["threshold"], df["precision"], "r-", linewidth=2, label="Precision")
    ax.plot(df["threshold"], df["f1"], "g--", linewidth=2, label="F1")
    ax.plot(df["threshold"], df["balanced_accuracy"], "orange", linewidth=1.5, label="Bal Acc")
    ax.axvline(x=current_thr, color="gray", linestyle=":", linewidth=2, label=f"Current thr={current_thr}")
    ax.axvline(x=best_prec_u["threshold"], color="green", linestyle="--", linewidth=2,
               label=f"Best Prec thr={best_prec_u['threshold']}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=7)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.70, 1.02)

fig.suptitle("V3 LayerMix — Threshold Trade-off: Recall vs Precision", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_PLOT / "v3_layermix_threshold_tradeoff.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── Chart 2: Precision-Recall curve ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 8))
ax2.plot(subj_sweep_df["recall"], subj_sweep_df["precision"], "b.-", linewidth=1.5, markersize=3)
ax2.scatter(current_u["recall"], current_u["precision"], s=200, c="red", edgecolors="black",
            zorder=10, label=f"Current (thr={current_thr})")
ax2.scatter(best_bal_u["recall"], best_bal_u["precision"], s=150, c="orange", edgecolors="black",
            zorder=10, label=f"Best BalAcc (thr={best_bal_u['threshold']})")
ax2.scatter(best_f1_u["recall"], best_f1_u["precision"], s=150, c="green", edgecolors="black",
            zorder=10, label=f"Best F1 (thr={best_f1_u['threshold']})")
ax2.scatter(best_prec_u["recall"], best_prec_u["precision"], s=200, c="blue", edgecolors="black",
            zorder=10, marker="D", label=f"Best Prec (thr={best_prec_u['threshold']})")
ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Recall", fontsize=12); ax2.set_ylabel("Precision", fontsize=12)
ax2.set_title("V3 LayerMix — Subject-level Precision-Recall Curve", fontsize=14, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(0.70, 1.02); ax2.set_ylim(0.75, 1.02)
fig2.tight_layout()
fig2.savefig(OUT_PLOT / "v3_layermix_precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

# ── Print summary ────────────────────────────────────────────
print(f"\n===== V3 LayerMix Recalibration =====")
print(f"\nOriginal (thr={current_thr}):")
print(f"  Sample: bal_acc={current_s['balanced_accuracy']}  rec={current_s['recall']}  prec={current_s['precision']}  f1={current_s['f1']}")
print(f"  Subject: bal_acc={current_u['balanced_accuracy']}  rec={current_u['recall']}  prec={current_u['precision']}  f1={current_u['f1']}")

print(f"\nBest Precision constrained (recall>=0.95, thr={best_prec_u['threshold']}):")
print(f"  Sample: bal_acc={best_prec_s['balanced_accuracy']}  rec={best_prec_s['recall']}  prec={best_prec_s['precision']}  f1={best_prec_s['f1']}")
print(f"  Subject: bal_acc={best_prec_u['balanced_accuracy']}  rec={best_prec_u['recall']}  prec={best_prec_u['precision']}  f1={best_prec_u['f1']}")

print(f"\nImprovements:")
print(f"  Sample precision: {current_s['precision']} -> {best_prec_s['precision']}  (+{best_prec_s['precision']-current_s['precision']:.4f})")
print(f"  Subject precision: {current_u['precision']} -> {best_prec_u['precision']}  (+{best_prec_u['precision']-current_u['precision']:.4f})")
print(f"  Subject recall: {current_u['recall']} -> {best_prec_u['recall']}  ({best_prec_u['recall']-current_u['recall']:+.4f})")

print(f"\nCritical subjects (Best Prec thr={best_prec_u['threshold']}):")
for sid in ["H26", "H27", "H28", "H29"]:
    sub = critical[critical["subject_id"] == sid]
    ms = float(sub["score_patients"].mean())
    fp_curr = int((sub["score_patients"] >= current_thr).sum())
    fp_new = int((sub["score_patients"] >= best_prec_u["threshold"]).sum())
    print(f"  {sid}: mean_score={ms:.4f}  FP_current={fp_curr}/{len(sub)}  FP_new={fp_new}/{len(sub)}")

print(f"\nOutputs: {OUT_CSV}")
print(f"Plots:   {OUT_PLOT}")
