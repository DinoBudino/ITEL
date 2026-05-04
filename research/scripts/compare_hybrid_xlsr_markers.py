from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from research.scripts.compare_cz_task_variants import VARIANT_SPECS
from research.scripts.evaluate_external import (
    _default_aggregation_method,
    _filter_sample_manifest,
    _positive_class_index,
    _predictions_from_threshold,
    _subset_frames,
    _threshold_sweep,
)
from research.src.config import CONFIG
from research.src.evaluate import binary_metrics
from research.src.segmentation import aggregate_probabilities
from research.src.train import MODEL_TRAINERS, load_embeddings


REPORT_CSV_DIR = CONFIG.paths.reports_dir / "CSV"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XLSR-only vs XLSR+marker hybrid transfer from Czech to English.")
    parser.add_argument("--cz-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name))
    parser.add_argument(
        "--cz-embedding-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"),
    )
    parser.add_argument("--en-sample-manifest", default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name))
    parser.add_argument(
        "--en-embedding-manifests",
        nargs="+",
        default=[str(CONFIG.paths.processed_manifest_dir / f"{CONFIG.datasets.en_dataset_id}_embeddings_seg_5s.csv")],
    )
    parser.add_argument(
        "--cz-feature-cache",
        default=str(CONFIG.paths.processed_manifest_dir / "cz_main_recording_proxy_features.csv"),
    )
    parser.add_argument(
        "--en-feature-cache",
        default=str(CONFIG.paths.processed_manifest_dir / "english_external_recording_proxy_features.csv"),
    )
    parser.add_argument(
        "--marker-report",
        default=str(REPORT_CSV_DIR / "compare_cz_english_markers.json"),
    )
    parser.add_argument("--task-variant", default="speech_sounds_only", choices=list(VARIANT_SPECS.keys()))
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["logreg_tuned", "mlp_small"],
        choices=list(MODEL_TRAINERS.keys()),
    )
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--aggregation-methods",
        nargs="+",
        default=[],
        choices=["mean", "median", "trimmed_mean"],
    )
    parser.add_argument(
        "--segment-selection",
        default="all",
        choices=["all", "passed_only"],
        help="Whether to aggregate all segments or only those with segment_qc_status=passed.",
    )
    parser.add_argument("--report-json", default=str(REPORT_CSV_DIR / "compare_hybrid_xlsr_markers.json"))
    parser.add_argument("--summary-csv", default=str(REPORT_CSV_DIR / "compare_hybrid_xlsr_markers_summary.csv"))
    return parser.parse_args()


def _selected_features(marker_report_path: Path) -> list[str]:
    if not marker_report_path.exists():
        raise FileNotFoundError(f"Marker report not found: {marker_report_path}")
    report = json.loads(marker_report_path.read_text(encoding="utf-8"))
    features = report.get("config", {}).get("selected_features", [])
    if not features:
        raise ValueError(f"No selected_features found in marker report: {marker_report_path}")
    return [str(feature) for feature in features]


def _filter_cz_variant(frame: pd.DataFrame, task_types: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[frame["task_type"].isin(task_types)].copy()


def _filter_embedding_rows(embedding_manifest: pd.DataFrame, sample_manifest: pd.DataFrame) -> pd.DataFrame:
    allowed_ids = set(sample_manifest["sample_id"].tolist())
    return embedding_manifest.loc[embedding_manifest["source_sample_id"].isin(allowed_ids)].copy()


def _prepare_cz_training_matrix(
    sample_manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
    feature_cache: pd.DataFrame,
    marker_features: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    merged = embedding_manifest.merge(sample_manifest[["sample_id", "label"]], left_on="source_sample_id", right_on="sample_id", how="left")
    embed_X, merged = load_embeddings(merged)
    feature_frame = feature_cache[["sample_id", *marker_features]].copy()
    merged = merged.merge(feature_frame, left_on="source_sample_id", right_on="sample_id", how="left", suffixes=("", "_feature"))
    marker_X = merged[marker_features].to_numpy(dtype=np.float32)
    hybrid_X = np.hstack([embed_X.astype(np.float32), marker_X])
    return embed_X.astype(np.float32), hybrid_X, merged["label"]


def _evaluate_hybrid_manifest(
    model,
    sample_manifest: pd.DataFrame,
    embedding_manifest_path: Path,
    feature_cache: pd.DataFrame,
    marker_features: list[str],
    *,
    aggregation_method: str | None = None,
    decision_threshold: float = 0.5,
    segment_selection: str = "all",
) -> tuple[dict[str, object], pd.DataFrame]:
    embedding_manifest = pd.read_csv(embedding_manifest_path)
    mode_name = embedding_manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
    aggregation = aggregation_method or _default_aggregation_method(mode_name)
    merged = embedding_manifest.merge(
        sample_manifest[["sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_sample"),
    )
    merged = merged.dropna(subset=["label"]).copy()
    if segment_selection == "passed_only" and "segment_qc_status" in merged.columns:
        passed = merged["segment_qc_status"].fillna("").astype(str).str.lower().eq("passed")
        filtered = merged.loc[passed].copy()
        if not filtered.empty:
            merged = filtered
    marker_frame = feature_cache[["sample_id", *marker_features]].copy()
    merged = merged.merge(marker_frame, left_on="source_sample_id", right_on="sample_id", how="left", suffixes=("", "_feature"))
    embed_X, merged = load_embeddings(merged)
    marker_X = merged[marker_features].to_numpy(dtype=np.float32)
    hybrid_X = np.hstack([embed_X.astype(np.float32), marker_X])

    positive_index = _positive_class_index(model)
    frame = merged.copy()
    frame["score_patients"] = model.predict_proba(hybrid_X)[:, positive_index]

    per_source = (
        frame.groupby(
            ["source_sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"],
            as_index=False,
            dropna=False,
        )["score_patients"]
        .agg(lambda values: aggregate_probabilities([float(value) for value in values], method=aggregation))
        .rename(columns={"source_sample_id": "sample_id"})
    )
    per_source["aggregation_method"] = aggregation
    if mode_name in {"seg_2s", "seg_5s", "seg_5s_balanced40"}:
        segment_counts = frame.groupby("source_sample_id", dropna=False).size().rename("n_segments_used").reset_index()
        per_source = per_source.merge(segment_counts, left_on="sample_id", right_on="source_sample_id", how="left").drop(columns=["source_sample_id"])
    per_source["decision_threshold"] = decision_threshold
    per_source["pred_label"] = _predictions_from_threshold(per_source["score_patients"], decision_threshold)

    metrics = binary_metrics(per_source["label"], per_source["pred_label"], per_source["score_patients"])
    metrics["aggregation_method"] = aggregation
    metrics["decision_threshold"] = decision_threshold
    metrics["threshold_sweep"] = _threshold_sweep(per_source)
    subset_metrics: dict[str, object] = {}
    for subset_name, subset_frame in _subset_frames(per_source).items():
        subset_metrics[subset_name] = binary_metrics(subset_frame["label"], subset_frame["pred_label"], subset_frame["score_patients"])
        subset_metrics[subset_name]["aggregation_method"] = aggregation
        subset_metrics[subset_name]["decision_threshold"] = decision_threshold
        subset_metrics[subset_name]["threshold_sweep"] = _threshold_sweep(subset_frame)
    metrics["subsets"] = subset_metrics
    return metrics, per_source


def _evaluate_xlsr_manifest(
    model,
    sample_manifest: pd.DataFrame,
    embedding_manifest_path: Path,
    *,
    aggregation_method: str | None = None,
    decision_threshold: float = 0.5,
    segment_selection: str = "all",
) -> tuple[dict[str, object], pd.DataFrame]:
    embedding_manifest = pd.read_csv(embedding_manifest_path)
    mode_name = embedding_manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
    aggregation = aggregation_method or _default_aggregation_method(mode_name)
    merged = embedding_manifest.merge(
        sample_manifest[["sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"]],
        left_on="source_sample_id",
        right_on="sample_id",
        how="left",
        suffixes=("", "_sample"),
    )
    merged = merged.dropna(subset=["label"]).copy()
    if segment_selection == "passed_only" and "segment_qc_status" in merged.columns:
        passed = merged["segment_qc_status"].fillna("").astype(str).str.lower().eq("passed")
        filtered = merged.loc[passed].copy()
        if not filtered.empty:
            merged = filtered
    embed_X, frame = load_embeddings(merged)
    positive_index = _positive_class_index(model)
    frame = frame.copy()
    frame["score_patients"] = model.predict_proba(embed_X.astype(np.float32))[:, positive_index]

    per_source = (
        frame.groupby(
            ["source_sample_id", "subject_id", "label", "review_status", "review_reason", "auto_qc_flag", "qc_flag"],
            as_index=False,
            dropna=False,
        )["score_patients"]
        .agg(lambda values: aggregate_probabilities([float(value) for value in values], method=aggregation))
        .rename(columns={"source_sample_id": "sample_id"})
    )
    per_source["aggregation_method"] = aggregation
    if mode_name.startswith("seg_"):
        segment_counts = frame.groupby("source_sample_id", dropna=False).size().rename("n_segments_used").reset_index()
        per_source = per_source.merge(segment_counts, left_on="sample_id", right_on="source_sample_id", how="left").drop(columns=["source_sample_id"])
    per_source["decision_threshold"] = decision_threshold
    per_source["pred_label"] = _predictions_from_threshold(per_source["score_patients"], decision_threshold)

    metrics = binary_metrics(per_source["label"], per_source["pred_label"], per_source["score_patients"])
    metrics["aggregation_method"] = aggregation
    metrics["decision_threshold"] = decision_threshold
    metrics["threshold_sweep"] = _threshold_sweep(per_source)
    subset_metrics: dict[str, object] = {}
    for subset_name, subset_frame in _subset_frames(per_source).items():
        subset_metrics[subset_name] = binary_metrics(subset_frame["label"], subset_frame["pred_label"], subset_frame["score_patients"])
        subset_metrics[subset_name]["aggregation_method"] = aggregation
        subset_metrics[subset_name]["decision_threshold"] = decision_threshold
        subset_metrics[subset_name]["threshold_sweep"] = _threshold_sweep(subset_frame)
    metrics["subsets"] = subset_metrics
    return metrics, per_source


def main() -> None:
    args = parse_args()
    marker_features = _selected_features(Path(args.marker_report))

    cz_samples = pd.read_csv(args.cz_sample_manifest)
    cz_embeddings = pd.read_csv(args.cz_embedding_manifest)
    en_samples = _filter_sample_manifest(pd.read_csv(args.en_sample_manifest), approved_only=args.approved_only)
    cz_feature_cache = pd.read_csv(args.cz_feature_cache)
    en_feature_cache = pd.read_csv(args.en_feature_cache)

    task_types = VARIANT_SPECS[args.task_variant]
    variant_cz_samples = _filter_cz_variant(cz_samples, task_types)
    variant_cz_embeddings = _filter_embedding_rows(cz_embeddings, variant_cz_samples)
    embed_X, hybrid_X, y = _prepare_cz_training_matrix(variant_cz_samples, variant_cz_embeddings, cz_feature_cache, marker_features)

    summary_rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "config": {
            **vars(args),
            "selected_marker_features": marker_features,
        },
        "results": {},
    }

    for model_type in args.model_types:
        report["results"][model_type] = {}
        xlsr_model = MODEL_TRAINERS[model_type](embed_X, y)
        hybrid_model = MODEL_TRAINERS[model_type](hybrid_X, y)

        aggregation_methods = args.aggregation_methods or [None]
        for manifest_str in args.en_embedding_manifests:
            manifest_path = Path(manifest_str)
            mode_name = manifest_path.stem.replace(f"{CONFIG.datasets.en_dataset_id}_embeddings_", "")
            report["results"][model_type][mode_name] = {}
            for aggregation_method in aggregation_methods:
                label = aggregation_method or _default_aggregation_method(mode_name)
                xlsr_metrics, _ = _evaluate_xlsr_manifest(
                    xlsr_model,
                    en_samples,
                    manifest_path,
                    aggregation_method=aggregation_method,
                    decision_threshold=args.decision_threshold,
                    segment_selection=args.segment_selection,
                )
                hybrid_metrics, _ = _evaluate_hybrid_manifest(
                    hybrid_model,
                    en_samples,
                    manifest_path,
                    en_feature_cache,
                    marker_features,
                    aggregation_method=aggregation_method,
                    decision_threshold=args.decision_threshold,
                    segment_selection=args.segment_selection,
                )

                report["results"][model_type][mode_name][label] = {
                    "xlsr_only": xlsr_metrics,
                    "hybrid_xlsr_markers": hybrid_metrics,
                }

                for setup_name, metrics in (("xlsr_only", xlsr_metrics), ("hybrid_xlsr_markers", hybrid_metrics)):
                    best_threshold = metrics.get("threshold_sweep", {}).get("best_by_balanced_accuracy")
                    summary_rows.append(
                        {
                            "task_variant": args.task_variant,
                            "model_type": model_type,
                            "setup": setup_name,
                            "mode": mode_name,
                            "aggregation_method": label,
                            "segment_selection": args.segment_selection,
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "auc": metrics["auc"],
                            "f1": metrics["f1"],
                            "best_threshold": None if not best_threshold else best_threshold["threshold"],
                            "best_threshold_balanced_accuracy": None if not best_threshold else best_threshold["balanced_accuracy"],
                            "best_threshold_accuracy": None if not best_threshold else best_threshold["accuracy"],
                        }
                    )

                print(
                    f"{model_type}/{mode_name}/{label}/{args.segment_selection}: "
                    f"xlsr bal_acc={xlsr_metrics['balanced_accuracy']:.4f} auc={xlsr_metrics['auc']:.4f} | "
                    f"hybrid bal_acc={hybrid_metrics['balanced_accuracy']:.4f} auc={hybrid_metrics['auc']:.4f}"
                )

    report_path = Path(args.report_json)
    summary_path = Path(args.summary_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved hybrid report to: {report_path}")
    print(f"Saved hybrid summary to: {summary_path}")


if __name__ == "__main__":
    main()
