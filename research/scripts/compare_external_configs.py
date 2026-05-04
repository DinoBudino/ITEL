from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.scripts.compare_cz_task_variants import VARIANT_SPECS
from research.scripts.evaluate_external import _evaluate_manifest, _filter_sample_manifest
from research.src.config import CONFIG
from research.src.train import MODEL_TRAINERS, load_embeddings


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare external English configurations without rerunning internal CV.")
    parser.add_argument("--cz-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name))
    parser.add_argument("--cz-embedding-manifest", default=str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"))
    parser.add_argument("--en-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name))
    parser.add_argument(
        "--embedding-manifests",
        nargs="+",
        default=[
            str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_2s.csv"),
            str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_5s.csv"),
        ],
    )
    parser.add_argument(
        "--task-variant",
        default="speech_sounds_only",
        choices=list(VARIANT_SPECS.keys()),
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["logreg", "logreg_tuned", "linear_svm", "xgboost", "mlp_small"],
        choices=list(MODEL_TRAINERS.keys()),
    )
    parser.add_argument(
        "--aggregation-methods",
        nargs="+",
        default=["mean", "median"],
        choices=["mean", "median"],
    )
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--report-json",
        default=str(REPORT_CSV_DIR / "compare_external_configs.json"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPORT_CSV_DIR / "compare_external_configs_summary.csv"),
    )
    return parser.parse_args()


def _filter_cz_variant(frame: pd.DataFrame, task_types: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[frame["task_type"].isin(task_types)].copy()


def _filter_embedding_rows(embedding_manifest: pd.DataFrame, sample_manifest: pd.DataFrame) -> pd.DataFrame:
    allowed_ids = set(sample_manifest["sample_id"].tolist())
    return embedding_manifest.loc[embedding_manifest["source_sample_id"].isin(allowed_ids)].copy()


def _train_model(sample_manifest: pd.DataFrame, embedding_manifest: pd.DataFrame, model_type: str):
    merged = embedding_manifest.merge(sample_manifest[["sample_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")
    X, merged = load_embeddings(merged)
    return MODEL_TRAINERS[model_type](X, merged["label"])


def main() -> None:
    args = parse_args()
    cz_samples = pd.read_csv(args.cz_sample_manifest)
    cz_embeddings = pd.read_csv(args.cz_embedding_manifest)
    en_samples = _filter_sample_manifest(pd.read_csv(args.en_sample_manifest), approved_only=args.approved_only)

    task_types = VARIANT_SPECS[args.task_variant]
    variant_samples = _filter_cz_variant(cz_samples, task_types)
    variant_embeddings = _filter_embedding_rows(cz_embeddings, variant_samples)

    report: dict[str, object] = {
        "config": vars(args),
        "task_types": list(task_types),
        "results": {},
    }
    summary_rows: list[dict[str, object]] = []

    for model_type in args.model_types:
        model = _train_model(variant_samples, variant_embeddings, model_type)
        per_model: dict[str, object] = {}
        for manifest_str in args.embedding_manifests:
            manifest_path = Path(manifest_str)
            mode_name = manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
            per_mode: dict[str, object] = {}
            for aggregation_method in args.aggregation_methods:
                metrics, _ = _evaluate_manifest(
                    model,
                    en_samples,
                    manifest_path,
                    aggregation_method=aggregation_method,
                    decision_threshold=args.decision_threshold,
                )
                best_threshold = metrics.get("threshold_sweep", {}).get("best_by_balanced_accuracy")
                per_mode[aggregation_method] = metrics
                summary_rows.append(
                    {
                        "task_variant": args.task_variant,
                        "model_type": model_type,
                        "mode": mode_name,
                        "aggregation_method": aggregation_method,
                        "n_rows": metrics["n_rows"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "auc": metrics["auc"],
                        "f1": metrics["f1"],
                        "accuracy_line": metrics["classification_report"].splitlines()[6].strip() if metrics.get("classification_report") else None,
                        "best_threshold": None if not best_threshold else best_threshold["threshold"],
                        "best_threshold_balanced_accuracy": None if not best_threshold else best_threshold["balanced_accuracy"],
                        "best_threshold_accuracy": None if not best_threshold else best_threshold["accuracy"],
                    }
                )
                print(
                    f"{model_type}/{mode_name}/{aggregation_method}: "
                    f"bal_acc={metrics['balanced_accuracy']:.4f} auc={metrics['auc']:.4f} f1={metrics['f1']:.4f} "
                    f"best_bal_acc={None if not best_threshold else best_threshold['balanced_accuracy']:.4f}"
                )
            per_model[mode_name] = per_mode
        report["results"][model_type] = per_model

    report_path = Path(args.report_json)
    summary_path = Path(args.summary_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved external config report to: {report_path}")
    print(f"Saved external config summary to: {summary_path}")


if __name__ == "__main__":
    main()
