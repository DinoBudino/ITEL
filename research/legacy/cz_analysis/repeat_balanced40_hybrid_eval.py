from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat balanced40 experiments across seeds for XLSR-only vs hybrid top3.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 21, 42, 84, 123],
        help="Random seeds used to sample balanced 40-audio English subsets.",
    )
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument(
        "--marker-report",
        default=str(REPORT_CSV_DIR / "compare_cz_english_markers_top3.json"),
        help="Marker report that defines the selected top-k marker set.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPORT_CSV_DIR / "repeat_balanced40_hybrid_eval_summary.csv"),
    )
    parser.add_argument(
        "--aggregate-json",
        default=str(REPORT_CSV_DIR / "repeat_balanced40_hybrid_eval.json"),
    )
    return parser.parse_args()


def _python_executable() -> str:
    return sys.executable


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command))
    subprocess.run(command, check=True)


def _read_hybrid_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = [
        "model_type",
        "setup",
        "mode",
        "balanced_accuracy",
        "auc",
        "f1",
        "best_threshold",
        "best_threshold_balanced_accuracy",
        "best_threshold_accuracy",
    ]
    return frame[keep].copy()


def _normalized_mode(value: str) -> str:
    marker = "_seed"
    if marker in value:
        return value.split(marker, 1)[0]
    return value


def _subset_tag(per_label: int) -> str:
    return f"balanced{per_label * 2}"


def main() -> None:
    args = parse_args()
    py = _python_executable()
    summary_rows: list[dict[str, object]] = []
    subset_tag = _subset_tag(args.per_label)

    for seed in args.seeds:
        subset_manifest = CONFIG.paths.manifests_dir / f"english_external_samples_{subset_tag}_seed{seed}.csv"
        embedding_dir = CONFIG.paths.processed_dir / CONFIG.datasets.en_dataset_id / "xlsr" / f"seg_5s_{subset_tag}_seed{seed}"
        embedding_manifest = CONFIG.paths.processed_manifest_dir / f"english_external_embeddings_seg_5s_{subset_tag}_seed{seed}.csv"
        hybrid_report = REPORT_CSV_DIR / f"compare_hybrid_xlsr_markers_{subset_tag}_seed{seed}.json"
        hybrid_summary = REPORT_CSV_DIR / f"compare_hybrid_xlsr_markers_{subset_tag}_seed{seed}_summary.csv"

        _run(
            [
                py,
                str(ROOT_DIR / "research" / "scripts" / "build_balanced_en_subset.py"),
                "--seed",
                str(seed),
                "--per-label",
                str(args.per_label),
                "--output-manifest",
                str(subset_manifest),
            ]
        )
        _run(
            [
                py,
                str(ROOT_DIR / "research" / "scripts" / "build_embeddings.py"),
                "--dataset",
                "english_external",
                "--manifest",
                str(subset_manifest),
                "--eval-mode",
                "seg_5s",
                "--output-dir",
                str(embedding_dir),
                "--output-manifest",
                str(embedding_manifest),
                "--no-skip-existing",
            ]
        )
        _run(
            [
                py,
                str(ROOT_DIR / "research" / "scripts" / "compare_hybrid_xlsr_markers.py"),
                "--en-sample-manifest",
                str(subset_manifest),
                "--en-embedding-manifests",
                str(embedding_manifest),
                "--model-types",
                "logreg_tuned",
                "--marker-report",
                str(args.marker_report),
                "--report-json",
                str(hybrid_report),
                "--summary-csv",
                str(hybrid_summary),
            ]
        )

        seed_summary = _read_hybrid_summary(hybrid_summary)
        seed_summary.insert(0, "seed", seed)
        summary_rows.extend(seed_summary.to_dict(orient="records"))

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["normalized_mode"] = summary_frame["mode"].map(_normalized_mode)
    aggregate = (
        summary_frame.groupby(["model_type", "setup", "normalized_mode"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_best_threshold_balanced_accuracy=("best_threshold_balanced_accuracy", "mean"),
            std_best_threshold_balanced_accuracy=("best_threshold_balanced_accuracy", "std"),
        )
        .sort_values(["model_type", "setup", "normalized_mode"])
        .reset_index(drop=True)
    )

    summary_path = Path(args.summary_csv)
    aggregate_path = Path(args.aggregate_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_path, index=False)
    aggregate_payload = {
        "config": vars(args),
        "aggregate": aggregate.to_dict(orient="records"),
    }
    aggregate_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")

    print(f"Saved repeated-seed summary to: {summary_path}")
    print(f"Saved repeated-seed aggregate to: {aggregate_path}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
