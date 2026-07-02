from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG


DEFAULT_SPECS = [
    (
        "natural_multilingual_light",
        CONFIG.paths.reports_dir / "CSV" / "metrics_attention_bilstm_natural.json",
        CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_natural_subjects.csv",
    ),
    (
        "natural_anti_confound",
        CONFIG.paths.reports_dir / "CSV" / "metrics_attention_bilstm_natural_anti_confound.json",
        CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_natural_anti_confound_subjects.csv",
    ),
    (
        "fixed_2s_multilingual_light",
        CONFIG.paths.reports_dir / "CSV" / "metrics_attention_bilstm_fixed_2s_multilingual_light.json",
        CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_fixed_2s_multilingual_light_subjects.csv",
    ),
]


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _confusion_counts(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "subject_fp": None,
            "subject_fn": None,
            "subject_tp": None,
            "subject_tn": None,
            "fp_subjects": "",
            "fn_subjects": "",
        }
    frame = pd.read_csv(path)
    y_true = frame["label"].astype(int)
    y_pred = (frame["pred_label"].astype(str) == "Patients").astype(int)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    return {
        "subject_fp": int(fp_mask.sum()),
        "subject_fn": int(fn_mask.sum()),
        "subject_tp": int(tp_mask.sum()),
        "subject_tn": int(tn_mask.sum()),
        "fp_subjects": ";".join(frame.loc[fp_mask, "subject_id"].astype(str).tolist()),
        "fn_subjects": ";".join(frame.loc[fn_mask, "subject_id"].astype(str).tolist()),
    }


def _row_from_metrics(name: str, metrics_path: Path, subjects_path: Path) -> dict[str, object]:
    metrics = _load_metrics(metrics_path)
    row: dict[str, object] = {
        "profile": name,
        "metrics_path": str(metrics_path),
        "subjects_path": str(subjects_path),
        "available": metrics is not None,
    }
    if metrics is None:
        row.update(_confusion_counts(subjects_path))
        return row

    sample_metrics = metrics.get("sample_level", {}).get("metrics", {})
    subject_metrics = metrics.get("subject_level", {}).get("metrics", {})
    dataset = metrics.get("dataset", {})
    row.update(
        {
            "n_samples": dataset.get("n_samples"),
            "n_subjects": dataset.get("n_subjects"),
            "sample_accuracy": sample_metrics.get("accuracy"),
            "sample_balanced_accuracy": sample_metrics.get("balanced_accuracy"),
            "sample_auc": sample_metrics.get("auc"),
            "sample_recall": sample_metrics.get("recall"),
            "sample_precision": sample_metrics.get("precision"),
            "sample_f1": sample_metrics.get("f1"),
            "subject_accuracy": subject_metrics.get("accuracy"),
            "subject_balanced_accuracy": subject_metrics.get("balanced_accuracy"),
            "subject_auc": subject_metrics.get("auc"),
            "subject_recall": subject_metrics.get("recall"),
            "subject_precision": subject_metrics.get("precision"),
            "subject_f1": subject_metrics.get("f1"),
        }
    )
    row.update(_confusion_counts(subjects_path))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare preprocessing profile evaluation metrics.")
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.reports_dir / "CSV" / "preprocessing_comparison.csv"),
        help="Output comparison CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [_row_from_metrics(name, Path(metrics), Path(subjects)) for name, metrics, subjects in DEFAULT_SPECS]
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
