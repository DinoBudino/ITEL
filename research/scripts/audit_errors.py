from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from research.src.config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error profiling: False Positive audit for Attention-BiLSTM natural-length model.")
    parser.add_argument(
        "--predictions",
        default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_natural.csv"),
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
    )
    parser.add_argument(
        "--sequence-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(CONFIG.paths.reports_dir / "CSV" / "error_audit_report.csv"),
    )
    parser.add_argument(
        "--output-summary",
        default=str(CONFIG.paths.reports_dir / "CSV" / "error_audit_summary.txt"),
    )
    return parser.parse_args()


def _logit(prob: float, eps: float = 1e-8) -> float:
    p = max(eps, min(1.0 - eps, float(prob)))
    return float(np.log(p / (1.0 - p)))


def _duration_bin(duration_ms: float) -> str:
    if duration_ms < 200:
        return "<0.2s"
    if duration_ms <= 500:
        return "0.2-0.5s"
    return ">0.5s"


def main() -> None:
    args = parse_args()

    predictions = pd.read_csv(args.predictions)
    samples = pd.read_csv(args.sample_manifest)
    seq_manifest = pd.read_csv(args.sequence_manifest)

    df = predictions.merge(
        samples[["sample_id", "task_type", "subject_id", "raw_audio_path"]],
        on="sample_id", how="left", suffixes=("", "_sample"),
    )
    df["label_str"] = df["label"].apply(lambda x: "Patients" if x == 1.0 else "Healthy")

    seq_info = seq_manifest[["sample_id", "sequence_length"]].copy()
    seq_info.rename(columns={"sequence_length": "num_hubert_frames"}, inplace=True)
    df = df.merge(seq_info, on="sample_id", how="left")

    print(f"Computing real durations for {len(df)} samples ...")
    durations_ms: list[float] = []
    for path in tqdm(df["raw_audio_path"].tolist(), desc="durations"):
        try:
            durations_ms.append(librosa.get_duration(path=path) * 1000.0)
        except Exception:
            durations_ms.append(np.nan)
    df["duration_ms"] = durations_ms

    df["score_logit"] = df["score_patients"].apply(_logit)
    df["duration_bin"] = df["duration_ms"].apply(_duration_bin)

    fp_mask = (df["label_str"] == "Healthy") & (df["pred_label"] == "Patients")
    tn_mask = (df["label_str"] == "Healthy") & (df["pred_label"] == "Healthy")

    fp = df[fp_mask].copy()
    tn = df[tn_mask].copy()

    fp_report = fp[[
        "subject_id", "sample_id", "duration_ms", "duration_bin",
        "task_type", "score_patients", "score_logit", "num_hubert_frames", "fold",
    ]].sort_values("score_patients", ascending=False).reset_index(drop=True)

    fp_report.to_csv(args.output_csv, index=False)
    print(f"\nFP report saved: {args.output_csv}  ({len(fp_report)} rows)")

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("ERROR AUDIT REPORT — AttentionBiLSTM (natural length)")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"Total samples:                    {len(df)}")
    lines.append(f"True  Healthy (label=H):          {len(df[df['label_str'] == 'Healthy'])}")
    lines.append(f"True  Patients (label=P):         {len(df[df['label_str'] == 'Patients'])}")
    lines.append("")

    lines.append(f"False Positives (H->P):           {len(fp)}  ({len(fp)/max(len(df[df['label_str']=='Healthy']),1)*100:.1f}% of Healthy)")
    lines.append(f"True Negatives  (H->H):           {len(tn)}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("DURATION COMPARISON: FP vs TN")
    lines.append("-" * 72)
    lines.append(
        f"FP mean duration:  {fp['duration_ms'].mean():.1f} ms  "
        f"(median: {fp['duration_ms'].median():.1f} ms, std: {fp['duration_ms'].std():.1f})"
    )
    lines.append(
        f"TN mean duration:  {tn['duration_ms'].mean():.1f} ms  "
        f"(median: {tn['duration_ms'].median():.1f} ms, std: {tn['duration_ms'].std():.1f})"
    )

    fp_dur = fp["duration_ms"].dropna().values
    tn_dur = tn["duration_ms"].dropna().values
    mw_stat, mw_p = stats.mannwhitneyu(fp_dur, tn_dur, alternative="two-sided") if len(fp_dur) and len(tn_dur) else (None, None)

    lines.append("")
    lines.append(f"Mann-Whitney U: {mw_stat:.1f}, p={mw_p:.4f}" if mw_stat is not None else "MW: N/A")
    lines.append("")

    lines.append("-" * 72)
    lines.append("FP DISTRIBUTION BY DURATION BIN")
    lines.append("-" * 72)
    for bin_label in ["<0.2s", "0.2-0.5s", ">0.5s"]:
        count = len(fp[fp["duration_bin"] == bin_label])
        total_healthy_in_bin = len(df[(df["label_str"] == "Healthy") & (df["duration_bin"] == bin_label)])
        rate = count / max(total_healthy_in_bin, 1) * 100
        lines.append(f"  {bin_label:>10s}: {count:>4d} FP / {total_healthy_in_bin:>5d} Healthy = {rate:.1f}% FP rate")

    lines.append("")
    lines.append("-" * 72)
    lines.append("FP DISTRIBUTION BY TASK TYPE")
    lines.append("-" * 72)
    for task in fp["task_type"].value_counts().index.tolist():
        count = fp["task_type"].value_counts()[task]
        total_task = len(df[(df["label_str"] == "Healthy") & (df["task_type"] == task)])
        rate = count / max(total_task, 1) * 100
        lines.append(f"  {task:>25s}: {count:>4d} FP / {total_task:>5d} Healthy = {rate:.1f}% FP rate")

    lines.append("")
    lines.append("-" * 72)
    lines.append("TOP FALSE POSITIVES (by score)")
    lines.append("-" * 72)
    for _, row in fp_report.head(10).iterrows():
        lines.append(
            f"  {row['subject_id']:>6s} | {row['sample_id']:<35s} | "
            f"{row['duration_ms']:6.0f}ms | {row['task_type']:<22s} | "
            f"score={row['score_patients']:.4f} logit={row['score_logit']:+.2f} | "
            f"frames={int(row['num_hubert_frames']):>3d}"
        )

    lines.append("")
    lines.append("-" * 72)
    lines.append("SUBJECT-LEVEL FP BREAKDOWN")
    lines.append("-" * 72)
    fp_by_subj = fp.groupby("subject_id").agg(
        fp_count=("sample_id", "count"),
        mean_score=("score_patients", "mean"),
        mean_duration_ms=("duration_ms", "mean"),
    ).sort_values("fp_count", ascending=False)
    for subj, row in fp_by_subj.iterrows():
        total_for_subj = len(df[(df["subject_id"] == subj) & (df["label_str"] == "Healthy")])
        pct = int(row["fp_count"]) / max(total_for_subj, 1) * 100
        lines.append(
            f"  {subj:>6s}: {int(row['fp_count']):>3d} FP / {total_for_subj:>3d} samples = {pct:.0f}% | "
            f"mean_score={row['mean_score']:.4f} | mean_dur={row['mean_duration_ms']:.0f}ms"
        )

    report = "\n".join(lines)
    print(report)

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(report, encoding="utf-8")
    print(f"\nSummary saved: {args.output_summary}")


if __name__ == "__main__":
    main()
