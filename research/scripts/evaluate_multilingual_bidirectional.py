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

from research.src.config import CONFIG
from research.src.multilingual_evaluation import aggregate_subject_embeddings, run_target_loso
from research.src.multilingual_features import load_statistical_sequence_embedding


REPORT_DIR = CONFIG.paths.reports_dir / "CSV" / "multilingual_transfer"


def _load_sequence_embedding(sequence_path: str | Path, layer_policy: str) -> np.ndarray:
    return load_statistical_sequence_embedding(sequence_path, layer_policy)


def _load_manifest_embeddings(manifest_path: str | Path, layer_policy: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        rows.append(
            {
                "sample_id": getattr(row, "sample_id"),
                "subject_id": str(getattr(row, "subject_id")),
                "label": str(getattr(row, "label")),
                "embedding": _load_sequence_embedding(getattr(row, "sequence_path"), layer_policy=layer_policy),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one multilingual head on both target EN and target CZ under the same protocol.")
    parser.add_argument("--cz-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural_multilingual_light.csv"))
    parser.add_argument("--en-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_v2.csv"))
    parser.add_argument("--layer-policy", choices=["last", "mean_layers"], default="last")
    parser.add_argument("--model-type", default="logreg", choices=["logreg", "linear_svm", "logreg_pca", "mlp_small", "mlp_medium"])
    parser.add_argument("--alignment-method", default="centroid_shift", choices=["raw", "centroid_shift"])
    parser.add_argument("--output-json", default=str(REPORT_DIR / "multilingual_bidirectional_eval.json"))
    parser.add_argument("--output-csv", default=str(REPORT_DIR / "multilingual_bidirectional_eval_predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cz_subject = aggregate_subject_embeddings(_load_manifest_embeddings(args.cz_sequence_manifest, args.layer_policy))
    en_subject = aggregate_subject_embeddings(_load_manifest_embeddings(args.en_sequence_manifest, args.layer_policy))

    en_metrics, en_pred = run_target_loso(cz_subject, en_subject, args.model_type, args.alignment_method)
    cz_metrics, cz_pred = run_target_loso(en_subject, cz_subject, args.model_type, args.alignment_method)

    en_pred["target_domain"] = "english"
    cz_pred["target_domain"] = "czech"

    report = {
        "config": {
            "model_type": args.model_type,
            "alignment_method": args.alignment_method,
            "layer_policy": args.layer_policy,
            "cz_sequence_manifest": args.cz_sequence_manifest,
            "en_sequence_manifest": args.en_sequence_manifest,
        },
        "datasets": {
            "cz_subjects": int(cz_subject["subject_id"].nunique()),
            "cz_labels": cz_subject["label"].value_counts().to_dict(),
            "en_subjects": int(en_subject["subject_id"].nunique()),
            "en_labels": en_subject["label"].value_counts().to_dict(),
        },
        "target_english": en_metrics,
        "target_czech": cz_metrics,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.concat([en_pred, cz_pred], ignore_index=True).to_csv(out_csv, index=False)

    print("===== MULTILINGUAL BIDIRECTIONAL EVAL =====")
    print(f"Model={args.model_type}  Alignment={args.alignment_method}")
    print(f"EN target: bal_acc={en_metrics['balanced_accuracy']:.4f}  auc={en_metrics['auc']:.4f}  recall={en_metrics['recall']:.4f}  best_bal_acc={en_metrics['best_balanced_accuracy_global']:.4f}")
    print(f"CZ target: bal_acc={cz_metrics['balanced_accuracy']:.4f}  auc={cz_metrics['auc']:.4f}  recall={cz_metrics['recall']:.4f}  best_bal_acc={cz_metrics['best_balanced_accuracy_global']:.4f}")
    print(f"\nJSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
