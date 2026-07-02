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


REPORT_JSON = CONFIG.paths.reports_dir / "CSV" / "language_shift" / "language_shift_transfer_report.json"
PREDICTIONS_CSV = CONFIG.paths.reports_dir / "CSV" / "language_shift" / "language_shift_transfer_subject_predictions.csv"
OUT_DIR = CONFIG.paths.reports_dir / "CSV" / "language_shift"


def _derive_thresholds(sweep: list[dict]) -> tuple[float, float]:
    if not sweep:
        return 0.30, 0.70
    high_thr = float(max(sweep, key=lambda row: float(row["balanced_accuracy"]))["threshold"])

    candidates = [row for row in sweep if float(row["recall"]) >= 0.90]
    if candidates:
        low_thr = float(max(candidates, key=lambda row: float(row["balanced_accuracy"]))["threshold"])
    else:
        low_thr = min(0.30, high_thr)

    if low_thr >= high_thr:
        low_thr = max(0.05, high_thr - 0.20)
    return round(low_thr, 2), round(high_thr, 2)


def _risk_label(score: float, low_thr: float, high_thr: float) -> str:
    if score >= high_thr:
        return "Alto"
    if score >= low_thr:
        return "Intermedio"
    return "Bajo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-subject risk levels from the best language-shift transfer strategy.")
    parser.add_argument("--report-json", default=str(REPORT_JSON))
    parser.add_argument("--predictions-csv", default=str(PREDICTIONS_CSV))
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "svm_rbf"])
    parser.add_argument("--strategy", default="cz_transfer_logreg")
    parser.add_argument("--variant", default="centroid_shift")
    parser.add_argument("--output-csv", default=str(OUT_DIR / "language_shift_risk_levels.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions_csv)

    section = report["czech_to_english_transfer"][args.classifier][args.variant]
    sweep = section.get("threshold_sweep", [])
    low_thr, high_thr = _derive_thresholds(sweep)

    subset = predictions[(predictions["strategy"] == args.strategy) & (predictions["variant"] == args.variant)].copy()
    subset["score_patients"] = subset["score_patients"].astype(float)
    subset["risk_level"] = subset["score_patients"].apply(lambda s: _risk_label(float(s), low_thr, high_thr))
    subset["low_threshold"] = low_thr
    subset["high_threshold"] = high_thr
    subset["risk_percent"] = (subset["score_patients"] * 100.0).round(2)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False)

    print("===== LANGUAGE SHIFT RISK LEVELS =====")
    print(f"classifier={args.classifier} strategy={args.strategy} variant={args.variant}")
    print(f"low_threshold={low_thr:.2f}  high_threshold={high_thr:.2f}")
    print(subset[["subject_id", "label", "score_patients", "risk_level"]].to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
