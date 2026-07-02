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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)

from research.src.config import CONFIG

OUT_DIR = CONFIG.paths.reports_dir / "CSV" / "pronunciation_audit"
PLOT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "pronunciation_audit"

SYLLABLE_MAP = {
    "02SOUHL": ("consonant", "consonant", "medium"),
    "03_1SL": ("one_syllable", "one_syllable_items", "low"),
    "04_2SL": ("word", "word_2syll", "medium"),
    "05_3SL": ("word", "word_3syll", "high"),
    "06_4SL": ("word", "word_4syll", "high"),
    "07_VSL": ("word", "word_difficult", "high"),
}

MODEL_SPECS = [
    ("Fase1_baseline", "research/research_results/CSV/predictions_contrastive_l0.1.csv"),
    ("V3_control", "research/research_results/CSV/predictions_v3.csv"),
    ("V3_LayerMix", "research/research_results/CSV/predictions_v3_layermix.csv"),
]


def build_master_table():
    df = pd.read_csv("research/manifests/cz_main_cleaned.csv")
    df["source_block_code"] = df["notes"].fillna("").astype(str).str.strip()
    groups = df["source_block_code"].map(lambda x: SYLLABLE_MAP.get(x, ("unknown", "unknown", "unknown")))
    df["task_group"] = [g[0] for g in groups]
    df["syllable_group"] = [g[1] for g in groups]
    df["articulatory_complexity"] = [g[2] for g in groups]
    df["label_num"] = (df["label"] == "Patients").astype(int)
    cols = ["sample_id", "subject_id", "label", "label_num", "task_type",
            "task_group", "syllable_group", "articulatory_complexity",
            "source_block_code", "duration_s"]
    return df[cols].copy()


def harmonize_predictions(pred_path, model_name, taxonomy_map):
    pred = pd.read_csv(pred_path)
    pred["label"] = pred["label"].astype(float)
    pred["score_patients"] = pred["score_patients"].astype(float)
    if "pred_label" in pred.columns:
        pred["pred_label_str"] = pred["pred_label"].astype(str)
    else:
        pred["pred_label_str"] = pred["score_patients"].apply(lambda s: "Patients" if s >= 0.5 else "Healthy")

    pred["task_group"] = pred["sample_id"].map(lambda sid: taxonomy_map.get(sid, {}).get("task_group", "unknown"))
    pred["syllable_group"] = pred["sample_id"].map(lambda sid: taxonomy_map.get(sid, {}).get("syllable_group", "unknown"))
    pred["articulatory_complexity"] = pred["sample_id"].map(lambda sid: taxonomy_map.get(sid, {}).get("articulatory_complexity", "unknown"))
    pred["task_type"] = pred["sample_id"].map(lambda sid: taxonomy_map.get(sid, {}).get("task_type", "unknown"))
    pred["model_name"] = model_name
    pred["pred_label_num"] = (pred["pred_label_str"] == "Patients").astype(int)
    pred["is_fp"] = (pred["label"] == 0) & (pred["pred_label_num"] == 1)
    pred["is_fn"] = (pred["label"] == 1) & (pred["pred_label_num"] == 0)
    pred["is_tp"] = (pred["label"] == 1) & (pred["pred_label_num"] == 1)
    pred["is_tn"] = (pred["label"] == 0) & (pred["pred_label_num"] == 0)
    return pred[
        ["sample_id", "subject_id", "label", "score_patients", "pred_label_str",
         "pred_label_num", "model_name", "task_type", "task_group",
         "syllable_group", "articulatory_complexity",
         "is_fp", "is_fn", "is_tp", "is_tn"]
    ]


def metrics_by_group(df, group_col):
    results = []
    for group_val in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == group_val]
        if len(sub) == 0:
            continue
        yt = sub["label"].values
        ys = sub["score_patients"].values
        yp = sub["pred_label_num"].values
        n_healthy = int((yt == 0).sum())
        fp_count = int(sub["is_fp"].sum()) if "is_fp" in sub.columns else int(((yt == 0) & (yp == 1)).sum())
        row = {
            "group": group_val, "n_samples": len(sub),
            "n_healthy": n_healthy, "n_patients": int((yt == 1).sum()),
            "n_subjects": int(sub["subject_id"].nunique()),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "false_positive_rate": round(fp_count / max(n_healthy, 1), 4),
        }
        try:
            row["auc"] = round(float(roc_auc_score(yt, ys)), 4)
        except ValueError:
            row["auc"] = None
        results.append(row)
    return pd.DataFrame(results)


def error_distribution(all_preds, group_col):
    rows = []
    for mn in all_preds["model_name"].unique():
        sub = all_preds[all_preds["model_name"] == mn]
        for gv in sorted(sub[group_col].dropna().unique()):
            gs = sub[sub[group_col] == gv]
            fp = int(gs["is_fp"].sum())
            fn = int(gs["is_fn"].sum())
            tp = int(gs["is_tp"].sum())
            tn = int(gs["is_tn"].sum())
            total = len(gs)
            rows.append({
                "model_name": mn, "group_col": group_col, "group_value": str(gv),
                "n_total": total, "n_fp": fp, "n_fn": fn, "n_tp": tp, "n_tn": tn,
                "fp_rate": round(fp / max(fp + tn, 1), 4),
                "fn_rate": round(fn / max(fn + tp, 1), 4),
            })
    return pd.DataFrame(rows)


def compute_deltas(metrics_all, group_col):
    rows = []
    for gv in sorted(metrics_all[metrics_all["group_col"] == group_col]["group"].unique()):
        f1 = metrics_all[(metrics_all["group_col"] == group_col) & (metrics_all["model_name"] == "Fase1_baseline") & (metrics_all["group"] == gv)]
        v3 = metrics_all[(metrics_all["group_col"] == group_col) & (metrics_all["model_name"] == "V3_control") & (metrics_all["group"] == gv)]
        lm = metrics_all[(metrics_all["group_col"] == group_col) & (metrics_all["model_name"] == "V3_LayerMix") & (metrics_all["group"] == gv)]
        if len(f1) == 0 or len(v3) == 0 or len(lm) == 0:
            continue
        f1r, v3r, lmr = f1.iloc[0], v3.iloc[0], lm.iloc[0]
        for metric in ["balanced_accuracy", "recall", "f1", "auc", "precision"]:
            rows.append({
                "group_col": group_col, "group_value": str(gv),
                "metric": metric,
                "Fase1_baseline": f1r.get(metric),
                "V3_control": v3r.get(metric),
                "V3_LayerMix": lmr.get(metric),
                "delta_V3_vs_Fase1": round((v3r.get(metric) or 0) - (f1r.get(metric) or 0), 4),
                "delta_LayerMix_vs_V3": round((lmr.get(metric) or 0) - (v3r.get(metric) or 0), 4),
                "delta_LayerMix_vs_Fase1": round((lmr.get(metric) or 0) - (f1r.get(metric) or 0), 4),
            })
    return pd.DataFrame(rows)


def plot_score_distribution(all_preds, group_col, out_path, title_prefix):
    groups = sorted(all_preds[group_col].dropna().unique())
    n_groups = len(groups)
    if n_groups == 0:
        return
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), squeeze=False)
    for i, gv in enumerate(groups):
        ax = axes[0, i]
        gs = all_preds[all_preds[group_col] == gv]
        h = gs[gs["label"] == 0]["score_patients"].values
        p = gs[gs["label"] == 1]["score_patients"].values
        ax.hist(h, bins=30, alpha=0.5, color="steelblue", label=f"Healthy (n={len(h)})")
        ax.hist(p, bins=30, alpha=0.5, color="crimson", label=f"Patients (n={len(p)})")
        ax.set_title(f"{gv}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle(f"{title_prefix} - Score Distribution by {group_col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_profile(error_df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

    model_order = ["Fase1_baseline", "V3_control", "V3_LayerMix"]
    metrics_plot = [("fp_rate", "False Positive Rate"), ("fn_rate", "False Negative Rate")]

    for row_idx, (col_name, ylabel) in enumerate(metrics_plot):
        for col_idx, mn in enumerate(model_order):
            ax = axes[row_idx, col_idx]
            sub = error_df[(error_df["model_name"] == mn) & (error_df["group_col"] == "syllable_group")]
            if len(sub) == 0:
                continue
            groups = sub["group_value"].tolist()
            vals = sub[col_name].tolist()
            colors = ["#66c2a5" if v < 0.3 else "#fc8d62" if v < 0.5 else "#e78ac3" for v in vals]
            ax.bar(range(len(groups)), vals, color=colors, edgecolor="black")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{mn}", fontweight="bold", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_ylim(0, max(1.0, max(vals) * 1.15))

    fig.suptitle("Error Profile by Syllable Group", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def subject_error_report(all_preds):
    rows = []
    for mn in all_preds["model_name"].unique():
        sub = all_preds[all_preds["model_name"] == mn]
        for sid in sorted(sub["subject_id"].unique()):
            ss = sub[sub["subject_id"] == sid]
            fp = int(ss["is_fp"].sum())
            fn = int(ss["is_fn"].sum())
            tp = int(ss["is_tp"].sum())
            tn = int(ss["is_tn"].sum())
            lab = ss["label"].iloc[0]
            rows.append({
                "model_name": mn, "subject_id": str(sid), "label": int(lab),
                "n_samples": len(ss), "fp": fp, "fn": fn, "tp": tp, "tn": tn,
                "error_rate": round((fp + fn) / max(len(ss), 1), 4),
            })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Pronunciation audit across existing models.")
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--plot-dir", default=str(PLOT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir, plot_dir = Path(args.output_dir), Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Building pronunciation master table ===")
    master = build_master_table()
    print(f"  Samples: {len(master)}")
    print(f"  task_type: {master['task_type'].value_counts().to_dict()}")
    print(f"  syllable_group: {master['syllable_group'].value_counts().to_dict()}")
    print(f"  complexity: {master['articulatory_complexity'].value_counts().to_dict()}")
    master.to_csv(out_dir / "pronunciation_master_table.csv", index=False)

    taxonomy_map = {}
    for _, r in master.iterrows():
        taxonomy_map[r["sample_id"]] = {
            "task_group": r["task_group"], "syllable_group": r["syllable_group"],
            "articulatory_complexity": r["articulatory_complexity"], "task_type": r["task_type"],
        }

    print("\n=== 2. Harmonizing model predictions ===")
    all_preds = []
    for mn, path in MODEL_SPECS:
        try:
            df = harmonize_predictions(path, mn, taxonomy_map)
            all_preds.append(df)
            print(f"  {mn}: {len(df)} predictions")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {mn}: ERROR - {e}")
    all_df = pd.concat(all_preds, ignore_index=True)
    print(f"  Total predictions: {len(all_df)}")

    print("\n=== 3. Metrics by pronunciation group ===")
    group_cols = ["task_type", "syllable_group", "articulatory_complexity"]
    all_metrics = []
    for gcol in group_cols:
        for mn in all_df["model_name"].unique():
            sub = all_df[all_df["model_name"] == mn]
            mdf = metrics_by_group(sub, gcol)
            mdf.insert(0, "model_name", mn)
            mdf.insert(0, "group_col", gcol)
            all_metrics.append(mdf)

        mdf_full = pd.concat([m for m in all_metrics if m["group_col"].values[0] == gcol], ignore_index=True)
        fname = f"metrics_by_{gcol}.csv"
        mdf_full.to_csv(out_dir / fname, index=False)
        print(f"  {fname}: {len(mdf_full)} rows")

    metrics_all = pd.concat(all_metrics, ignore_index=True)

    print("\n=== 4. Model deltas by pronunciation ===")
    deltas = []
    for gcol in group_cols:
        dd = compute_deltas(metrics_all, gcol)
        if len(dd) > 0:
            deltas.append(dd)
    if deltas:
        delta_df = pd.concat(deltas, ignore_index=True)
        delta_df.to_csv(out_dir / "model_delta_by_pronunciation_group.csv", index=False)
        print(f"  model_delta: {len(delta_df)} rows")

    print("\n=== 5. Error distribution ===")
    error_rows = []
    for gcol in group_cols:
        ed = error_distribution(all_df, gcol)
        if len(ed) > 0:
            error_rows.append(ed)
    if error_rows:
        error_df = pd.concat(error_rows, ignore_index=True)
        error_df.to_csv(out_dir / "error_distribution_by_group.csv", index=False)
        print(f"  errors: {len(error_df)} rows")

    subj_err = subject_error_report(all_df)
    subj_err.to_csv(out_dir / "subject_error_report.csv", index=False)
    critical = subj_err[subj_err["subject_id"].isin(["H26", "H27", "H28", "H29"])]
    print(f"\n  Critical subjects:")
    for _, r in critical.iterrows():
        print(f"    {r['subject_id']} | {r['model_name']:>20s} | label={r['label']} | "
              f"fp={r['fp']} fn={r['fn']} | error_rate={r['error_rate']:.4f}")

    print("\n=== 6. Score distributions (V3 LayerMix) ===")
    v3lm = all_df[all_df["model_name"] == "V3_LayerMix"]
    plot_score_distribution(v3lm, "syllable_group",
                            plot_dir / "score_distribution_by_syllable_group.png",
                            "V3 LayerMix")
    plot_score_distribution(v3lm, "task_type",
                            plot_dir / "score_distribution_by_task_type.png",
                            "V3 LayerMix")
    if error_rows:
        plot_error_profile(pd.concat(error_rows, ignore_index=True),
                           plot_dir / "error_profile_by_group.png")
    print("  Plots saved.")

    print("\n=== 7. Key findings ===")
    print()
    for gcol in group_cols:
        print(f"--- {gcol} ---")
        bm = metrics_all[(metrics_all["group_col"] == gcol) & (metrics_all["model_name"] == "V3_LayerMix")]
        if len(bm) == 0:
            continue
        best = bm.sort_values("balanced_accuracy", ascending=False).iloc[0]
        worst = bm.sort_values("balanced_accuracy", ascending=True).iloc[0]
        print(f"  Best:  {best['group']} -> bal_acc={best['balanced_accuracy']}  auc={best['auc']}")
        print(f"  Worst: {worst['group']} -> bal_acc={worst['balanced_accuracy']}  auc={worst['auc']}")

    print()
    print("Raw diffs (LayerMix - V3_control) by syllable_group:")
    if deltas:
        dd = pd.concat(deltas, ignore_index=True)
        dd_bal = dd[(dd["metric"] == "balanced_accuracy") & (dd["group_col"] == "syllable_group")]
        for _, r in dd_bal.iterrows():
            print(f"  {r['group_value']:>18s}: {r['delta_LayerMix_vs_V3']:+.4f}")
    print()
    print(f"Outputs: {out_dir}")
    print(f"Plots:   {plot_dir}")


if __name__ == "__main__":
    main()
