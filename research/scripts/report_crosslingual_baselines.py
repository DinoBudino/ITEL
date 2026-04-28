from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from research.src.config import CONFIG


FIG_DIR = CONFIG.paths.reports_dir / "Figuras"
CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual cross-lingual reports from Czech task-variant comparisons.")
    parser.add_argument(
        "--variant-summary",
        default=str(CSV_DIR / "compare_cz_task_variants_summary.csv"),
        help="CSV summary produced by compare_cz_task_variants.py.",
    )
    parser.add_argument(
        "--variant-report",
        default=str(CSV_DIR / "compare_cz_task_variants.json"),
        help="JSON report produced by compare_cz_task_variants.py.",
    )
    parser.add_argument(
        "--mlp-metrics",
        default=str(CSV_DIR / "mlp_speech_sounds_only" / "metrics_english_external.json"),
        help="Optional external metrics JSON for the MLP speech-sounds model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIG_DIR),
        help="Directory where the figures will be written.",
    )
    return parser.parse_args()


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_variant_overview(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values(by="en_balanced_accuracy", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=ordered, x="variant", y="cz_subject_balanced_accuracy", ax=axes[0], palette="Blues_d")
    axes[0].set_title("CZ Internal by Task Variant")
    axes[0].set_ylabel("Subject Balanced Accuracy")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=ordered, x="variant", y="en_balanced_accuracy", ax=axes[1], palette="YlOrBr")
    axes[1].set_title("EN External by Task Variant")
    axes[1].set_ylabel("Approved-only Balanced Accuracy")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    _save(fig, output_dir / "crosslingual_variant_overview.png")


def plot_baseline_focus(summary: pd.DataFrame, output_dir: Path) -> None:
    focus = summary.loc[summary["variant"].isin(["all_tasks", "speech_sounds_only"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    internal = focus[["variant", "cz_subject_balanced_accuracy", "cz_subject_auc"]].melt(id_vars="variant", var_name="metric", value_name="value")
    external = focus[["variant", "en_balanced_accuracy", "en_auc"]].melt(id_vars="variant", var_name="metric", value_name="value")

    sns.barplot(data=internal, x="variant", y="value", hue="metric", ax=axes[0], palette="Blues")
    axes[0].set_title("Czech Internal")
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("")

    sns.barplot(data=external, x="variant", y="value", hue="metric", ax=axes[1], palette="YlOrBr")
    axes[1].set_title("English External (seg_5s)")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("")

    _save(fig, output_dir / "crosslingual_all_vs_speech_sounds.png")


def plot_threshold_curves(report: dict[str, object], output_dir: Path) -> None:
    rows: list[dict[str, float | str]] = []
    for variant in ["all_tasks", "speech_sounds_only"]:
        metrics = report["variants"][variant]["external"]["seg_5s"]["metrics"]
        for row in metrics["threshold_sweep"]["grid"]:
            rows.append(
                {
                    "variant": variant,
                    "threshold": row["threshold"],
                    "balanced_accuracy": row["balanced_accuracy"],
                }
            )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=frame, x="threshold", y="balanced_accuracy", hue="variant", marker="o", ax=ax)
    ax.set_title("Threshold Sweep on EN Approved-only (seg_5s)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Decision Threshold")
    _save(fig, output_dir / "crosslingual_threshold_sweep_seg5s.png")


def plot_model_family(summary: pd.DataFrame, mlp_metrics_path: Path, output_dir: Path) -> None:
    if not mlp_metrics_path.exists():
        return
    mlp_metrics = json.loads(mlp_metrics_path.read_text(encoding="utf-8"))
    speech_logreg = summary.loc[summary["variant"] == "speech_sounds_only"].iloc[0]
    family = pd.DataFrame(
        [
            {
                "model_family": "logreg",
                "en_balanced_accuracy": speech_logreg["en_balanced_accuracy"],
                "en_auc": speech_logreg["en_auc"],
            },
            {
                "model_family": "mlp",
                "en_balanced_accuracy": mlp_metrics["seg_5s"]["balanced_accuracy"],
                "en_auc": mlp_metrics["seg_5s"]["auc"],
            },
        ]
    )
    plot_frame = family.melt(id_vars="model_family", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=plot_frame, x="model_family", y="value", hue="metric", ax=ax, palette="crest")
    ax.set_title("Speech-sounds Model Family Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("EN Approved-only Score")
    _save(fig, output_dir / "crosslingual_model_family_comparison.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary = pd.read_csv(args.variant_summary)
    report = json.loads(Path(args.variant_report).read_text(encoding="utf-8"))

    _style()
    plot_variant_overview(summary, output_dir)
    plot_baseline_focus(summary, output_dir)
    plot_threshold_curves(report, output_dir)
    plot_model_family(summary, Path(args.mlp_metrics), output_dir)

    print(f"Saved cross-lingual figures to: {output_dir}")


if __name__ == "__main__":
    main()
