from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.embedding_pipeline import build_embedding_records, save_embedding_manifest
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_params


def default_manifest_for_dataset(dataset_id: str) -> Path:
    if dataset_id == CONFIG.datasets.cz_dataset_id:
        return CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name
    if dataset_id == CONFIG.datasets.en_dataset_id:
        return CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name
    raise ValueError(f"Unsupported dataset_id: {dataset_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen multilingual XLS-R embeddings from a sample manifest.")
    parser.add_argument("--dataset", required=True, choices=[CONFIG.datasets.cz_dataset_id, CONFIG.datasets.en_dataset_id])
    parser.add_argument("--eval-mode", required=True, choices=["train_default", "seg_2s", "seg_5s", "full_audio"])
    parser.add_argument("--manifest", default=None, help="Optional override for the sample manifest path.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the embedding output directory.")
    parser.add_argument("--output-manifest", default=None, help="Optional override for the embedding manifest CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick pilot runs.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Recompute embeddings even if .npy files already exist.")
    parser.add_argument("--mlflow-experiment", default="itel-embeddings", help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to a local file store under research_results/mlruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    sample_manifest = Path(args.manifest) if args.manifest else default_manifest_for_dataset(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.paths.processed_dir / args.dataset / "xlsr300m" / args.eval_mode
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else CONFIG.paths.processed_manifest_dir / f"{args.dataset}_embeddings_{args.eval_mode}.csv"
    )

    with mlflow.start_run(run_name=args.mlflow_run_name):
        frame = build_embedding_records(
            sample_manifest,
            output_dir,
            args.eval_mode,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
        )
        saved_manifest = save_embedding_manifest(frame, output_manifest)

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "build_embeddings.py",
                "dataset_id": args.dataset,
                "eval_mode": args.eval_mode,
                "encoder_model": CONFIG.encoder.model_id,
            }
        )
        log_params(
            {
                "dataset": args.dataset,
                "eval_mode": args.eval_mode,
                "manifest": str(sample_manifest),
                "output_dir": str(output_dir),
                "output_manifest": str(saved_manifest),
                "skip_existing": not args.no_skip_existing,
                "limit": args.limit,
                "tracking_uri": tracking_uri,
            }
        )
        log_dict(
            "metadata/build_embeddings_summary.json",
            {
                "n_rows": int(len(frame)),
                "n_source_samples": 0 if frame.empty else int(frame["source_sample_id"].nunique()),
                "embedding_dim": None if frame.empty else int(frame["embedding_dim"].iloc[0]),
                "preproc_versions": [] if frame.empty else sorted(frame["preproc_version"].dropna().astype(str).unique().tolist()),
                "chunk_policies": [] if frame.empty else sorted(frame["chunk_policy"].dropna().astype(str).unique().tolist()),
            },
        )
        log_artifact(saved_manifest, artifact_path="embedding_manifests")

        print(f"Saved embedding manifest to: {saved_manifest}")
        print(f"Generated {len(frame)} embedding rows in: {output_dir}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
