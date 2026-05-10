import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)

word_pred = pd.read_csv("research/research_results/CSV/predictions_word_expert.csv")
cons_pred = pd.read_csv("research/research_results/CSV/predictions_consonant_expert.csv")

# Ensemble: union of both experts
ensemble = pd.concat([word_pred, cons_pred], ignore_index=True)
print(f"Ensemble: {len(ensemble)} samples (word={len(word_pred)}, cons={len(cons_pred)})")

# Sample-level metrics at threshold 0.5
def fold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    m["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    return m

sample_m = fold_metrics(ensemble["label"].values, ensemble["score_patients"].values, threshold=0.5)
print("\n=== SAMPLE-LEVEL (threshold=0.5) ===")
for k, v in sample_m.items():
    print(f"  {k:>20s}: {v:.4f}" if v else f"  {k:>20s}: None")

# Subject-level
subject_agg = (
    ensemble.groupby(["subject_id", "label"], as_index=False)["score_patients"]
    .mean()
    .rename(columns={"score_patients": "score_mean"})
)
subject_agg["pred_label"] = subject_agg["score_mean"].apply(
    lambda s: "Patients" if s >= 0.5 else "Healthy"
)
subject_m = fold_metrics(subject_agg["label"].values, subject_agg["score_mean"].values, threshold=0.5)
print("\n=== SUBJECT-LEVEL (threshold=0.5) ===")
for k, v in subject_m.items():
    print(f"  {k:>20s}: {v:.4f}" if v else f"  {k:>20s}: None")

# Critical subjects
print("\n=== CRITICAL SUBJECTS (word + consonant combined) ===")
for s in ["H27", "H29", "H26", "H28"]:
    subj = subject_agg[subject_agg.subject_id == s]
    if len(subj) > 0:
        row = subj.iloc[0]
        print(f"  {s}: score_mean={row['score_mean']:.4f}  label={row['label']}  pred={row['pred_label']}")

# Save ensemble predictions
ensemble.to_csv("research/research_results/CSV/predictions_ensemble.csv", index=False)
subject_agg.to_csv("research/research_results/CSV/predictions_ensemble_subjects.csv", index=False)

# FP count
fp_mask = (ensemble["label"] == 0.0) & (ensemble["pred_label"] == "Patients")
fp_count = fp_mask.sum()
total_healthy = (ensemble["label"] == 0.0).sum()
print(f"\nFPs: {fp_count}/{total_healthy} = {fp_count/total_healthy*100:.1f}%")

# Compare tables
print("\n=== COMPARISON: CRITICAL SUBJECTS ===")
print(f"           CLEANED-SINGLE    WORD-EXPERT    CONS-EXPERT")
for s in ["H27", "H29", "H26", "H28"]:
    clean_info = f"{23}/{27}" if s == "H27" else (f"{27}/{27}" if s == "H29" else (f"{23}/{24}" if s == "H26" else "33/33"))
    w_info = f"{21}/{21}" if s == "H27" else (f"{21}/{21}" if s == "H29" else (f"{20}/{20}" if s == "H26" else "23/23"))
    c_info = f"{6}/{6}" if s == "H27" else (f"{6}/{6}" if s == "H29" else (f"{4}/{4}" if s == "H26" else "10/10"))
    print(f"  {s:>4s}:    {clean_info} FP          {w_info} FP        {c_info} FP")

print("\n=== METRICS COMPARISON ===")
print(f"{'':>20s}  {'CLEAN-SINGLE':>14s}  {'WORD-EXP':>12s}  {'CONS-EXP':>12s}  {'ENSEMBLE':>12s}")
for metric in ["accuracy", "balanced_accuracy", "recall", "f1", "auc"]:
    clean_v = 0.859 if metric == "accuracy" else (0.843 if metric == "balanced_accuracy" else (0.954 if metric == "recall" else (0.886 if metric == "f1" else 0.924)))
    word_v = word_pred if False else None # computed below
    print(f"  {metric:>20s}:  ...")

# Compute per-model metrics
print("\nSample-level per expert:")
for name, pred in [("WORD", word_pred), ("CONS", cons_pred)]:
    m = fold_metrics(pred["label"].values, pred["score_patients"].values, threshold=0.5)
    print(f"  {name}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

print(f"\n  ENSEMBLE: acc={sample_m['accuracy']:.4f} bal_acc={sample_m['balanced_accuracy']:.4f} recall={sample_m['recall']:.4f} f1={sample_m['f1']:.4f} auc={sample_m['auc']:.4f}")

print("\nSubject-level per expert:")
for name, pred in [("WORD", word_pred), ("CONS", cons_pred)]:
    sagg = pred.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    sagg["pred_label"] = sagg["score_patients"].apply(lambda s: "Patients" if s >= 0.5 else "Healthy")
    m = fold_metrics(sagg["label"].values, sagg["score_patients"].values, threshold=0.5)
    print(f"  {name}: acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")

print(f"\n  ENSEMBLE: acc={subject_m['accuracy']:.4f} bal_acc={subject_m['balanced_accuracy']:.4f} recall={subject_m['recall']:.4f} f1={subject_m['f1']:.4f} auc={subject_m['auc']:.4f}")
