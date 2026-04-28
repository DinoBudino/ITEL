from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.scripts.evaluate_external import _evaluate_manifest, _filter_sample_manifest
from research.scripts.evaluate_internal_cz import run_cv
from research.src.config import CONFIG
from research.src.train import fit_logreg_classifier, load_embeddings


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"

VARIANT_SPECS: dict[str, tuple[str, ...]] = {
    "all_tasks": ("controlled_vowel", "controlled_consonant", "controlled_word"),
    "word_only": ("controlled_word",),
    "vowel_only": ("controlled_vowel",),
    "consonant_only": ("controlled_consonant",),
    "speech_sounds_only": ("controlled_vowel", "controlled_consonant"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Czech task-type training variants against internal CV and English external evaluation.")
    parser.add_argument(
        "--cz-sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Path to the Czech sample manifest.",
    )
    parser.add_argument(
        "--cz-embedding-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"),
        help="Path to the Czech embedding manifest.",
    )
    parser.add_argument(
        "--en-sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name),
        help="Path to the English sample manifest.",
    )
    parser.add_argument(
        "--approved-only",
        action="store_true",
        help="Evaluate only approved English rows.",
    )
    parser.add_argument(
        "--embedding-manifests",
        nargs="+",
        default=[str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_5s.csv")],
        help="English embedding manifests to evaluate for each Czech variant.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_SPECS.keys()),
        choices=list(VARIANT_SPECS.keys()),
        help="Task-type variants to compare.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold splits for internal Czech CV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for internal Czech CV.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for external evaluation.",
    )
    parser.add_argument(
        "--report-json",
        default=str(REPORT_CSV_DIR / "compare_cz_task_variants.json"),
        help="Where to save the full comparison report.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(REPORT_CSV_DIR / "compare_cz_task_variants_summary.csv"),
        help="Where to save the compact summary CSV.",
    )
    return parser.parse_args()


def _filter_cz_variant(frame: pd.DataFrame, task_types: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[frame["task_type"].isin(task_types)].copy()


def _filter_embedding_rows(embedding_manifest: pd.DataFrame, sample_manifest: pd.DataFrame) -> pd.DataFrame:
    allowed_ids = set(sample_manifest["sample_id"].tolist())
    return embedding_manifest.loc[embedding_manifest["source_sample_id"].isin(allowed_ids)].copy()


def _train_variant_model(sample_manifest: pd.DataFrame, embedding_manifest: pd.DataFrame):
    merged = embedding_manifest.merge(sample_manifest[["sample_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")
    X, merged = load_embeddings(merged)
    model = fit_logreg_classifier(X, merged["label"])
    return model


def main() -> None:
    args = parse_args()
    cz_sample_manifest = pd.read_csv(args.cz_sample_manifest)
    cz_embedding_manifest = pd.read_csv(args.cz_embedding_manifest)
    en_sample_manifest = _filter_sample_manifest(pd.read_csv(args.en_sample_manifest), approved_only=args.approved_only)

    report: dict[str, object] = {
        "config": {
            "approved_only": args.approved_only,
            "variants": args.variants,
            "decision_threshold": args.decision_threshold,
            "embedding_manifests": args.embedding_manifests,
            "n_splits": args.n_splits,
            "random_state": args.random_state,
        },
        "variants": {},
    }
    summary_rows: list[dict[str, object]] = []

    for variant_name in args.variants:
        task_types = VARIANT_SPECS[variant_name]
        variant_cz_samples = _filter_cz_variant(cz_sample_manifest, task_types)
        variant_cz_embeddings = _filter_embedding_rows(cz_embedding_manifest, variant_cz_samples)

        sample_predictions, subject_predictions, internal_metrics = run_cv(
            sample_manifest=variant_cz_samples,
            embedding_manifest=variant_cz_embeddings,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        model = _train_variant_model(variant_cz_samples, variant_cz_embeddings)

        external_by_mode: dict[str, object] = {}
        for manifest_str in args.embedding_manifests:
            manifest_path = Path(manifest_str)
            mode_name = manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
            external_metrics, external_predictions = _evaluate_manifest(
                model,
                en_sample_manifest,
                manifest_path,
                decision_threshold=args.decision_threshold,
            )
            external_by_mode[mode_name] = {
                "metrics": external_metrics,
                "n_prediction_rows": int(len(external_predictions)),
            }
            best_threshold = external_metrics.get("threshold_sweep", {}).get("best_by_balanced_accuracy")
            summary_rows.append(
                {
                    "variant": variant_name,
                    "task_types": ",".join(task_types),
                    "mode": mode_name,
                    "cz_subject_balanced_accuracy": internal_metrics["subject_level"]["balanced_accuracy"],
                    "cz_subject_auc": internal_metrics["subject_level"]["auc"],
                    "en_balanced_accuracy": external_metrics["balanced_accuracy"],
                    "en_auc": external_metrics["auc"],
                    "best_threshold": None if not best_threshold else best_threshold["threshold"],
                    "best_threshold_balanced_accuracy": None if not best_threshold else best_threshold["balanced_accuracy"],
                }
            )

        report["variants"][variant_name] = {
            "task_types": list(task_types),
            "dataset": {
                "n_samples": int(len(variant_cz_samples)),
                "n_subjects": int(variant_cz_samples["subject_id"].nunique()),
                "samples_by_label": variant_cz_samples["label"].value_counts().to_dict(),
            },
            "internal_cv": internal_metrics,
            "external": external_by_mode,
        }

        print(
            f"{variant_name}: cz_subject_bal_acc={internal_metrics['subject_level']['balanced_accuracy']:.4f} "
            f"cz_subject_auc={internal_metrics['subject_level']['auc']:.4f}"
        )
        for mode_name, mode_payload in external_by_mode.items():
            metrics = mode_payload["metrics"]
            best_threshold = metrics.get("threshold_sweep", {}).get("best_by_balanced_accuracy")
            print(
                f"  {mode_name}: en_bal_acc={metrics['balanced_accuracy']:.4f} en_auc={metrics['auc']:.4f} "
                f"best_threshold={None if not best_threshold else best_threshold['threshold']}"
            )

    report_json = Path(args.report_json)
    summary_csv = Path(args.summary_csv)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"Saved variant comparison report to: {report_json}")
    print(f"Saved variant summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
