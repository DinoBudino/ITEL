from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "research" / "research_results" / "CSV"
MULTILINGUAL_DIR = RESULTS_DIR / "multilingual_transfer"
LANGUAGE_SHIFT_DIR = RESULTS_DIR / "language_shift"

FINAL_PREDICTIONS = MULTILINGUAL_DIR / "multilingual_bidirectional_logreg_pca_centroid_predictions.csv"
RAW_PREDICTIONS = MULTILINGUAL_DIR / "multilingual_bidirectional_logreg_pca_raw_predictions.csv"
ENGLISH_ONLY_PREDICTIONS = LANGUAGE_SHIFT_DIR / "language_shift_transfer_subject_predictions.csv"
OUTPUT_CSV = MULTILINGUAL_DIR / "english_prediction_comparison.csv"

LABEL_MAP = {"Healthy": "TD", "Patients": "LI"}


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")


def _validate_unique_subjects(frame: pd.DataFrame, name: str) -> None:
    if len(frame) != 48:
        raise ValueError(f"Expected 48 rows for {name}, found {len(frame)}")
    duplicates = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate subject_id values in {name}: {duplicates}")


def _load_multilingual(path: Path, alignment_method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    required = {
        "subject_id",
        "label",
        "score_patients",
        "pred_label",
        "threshold_fold",
        "model_type",
        "alignment_method",
        "target_domain",
    }
    _require_columns(frame, required, path)
    frame = frame.loc[
        frame["target_domain"].eq("english")
        & frame["model_type"].eq("logreg_pca")
        & frame["alignment_method"].eq(alignment_method)
    ].copy()
    _validate_unique_subjects(frame, f"logreg_pca + {alignment_method}")
    return frame


def _load_english_only(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    required = {
        "subject_id",
        "label",
        "score_patients",
        "pred_label",
        "strategy",
        "alignment_method",
        "variant",
        "best_C",
    }
    _require_columns(frame, required, path)
    frame = frame.loc[
        frame["strategy"].eq("english_loso_logreg")
        & frame["alignment_method"].eq("raw")
        & frame["variant"].eq("raw")
    ].copy()
    _validate_unique_subjects(frame, "English-only logreg")
    return frame


def build_comparison() -> pd.DataFrame:
    final = _load_multilingual(FINAL_PREDICTIONS, "centroid_shift")
    raw = _load_multilingual(RAW_PREDICTIONS, "raw")
    english_only = _load_english_only(ENGLISH_ONLY_PREDICTIONS)

    subject_sets = [set(frame["subject_id"]) for frame in (final, raw, english_only)]
    if not subject_sets[0] == subject_sets[1] == subject_sets[2]:
        raise ValueError("The three configurations do not contain identical subject_id sets")

    comparison = final[["subject_id", "label", "score_patients", "pred_label"]].rename(
        columns={
            "label": "final_label",
            "score_patients": "final_score",
            "pred_label": "final_pred",
        }
    )
    comparison = comparison.merge(
        raw[["subject_id", "label", "score_patients", "pred_label"]].rename(
            columns={
                "label": "raw_label",
                "score_patients": "raw_score",
                "pred_label": "raw_pred",
            }
        ),
        on="subject_id",
        validate="one_to_one",
    )
    comparison = comparison.merge(
        english_only[["subject_id", "label", "score_patients", "pred_label"]].rename(
            columns={
                "label": "english_only_label",
                "score_patients": "english_only_score",
                "pred_label": "english_only_pred",
            }
        ),
        on="subject_id",
        validate="one_to_one",
    )

    expected_numeric_labels = comparison["final_label"].map({"Healthy": 0, "Patients": 1})
    english_only_labels = pd.to_numeric(comparison["english_only_label"], errors="raise")
    if not comparison["final_label"].equals(comparison["raw_label"]):
        raise ValueError("Real labels disagree between final and raw predictions")
    if not expected_numeric_labels.equals(english_only_labels):
        raise ValueError("Real labels disagree between multilingual and English-only predictions")

    comparison["y_true"] = comparison["final_label"].map(LABEL_MAP)
    comparison["final_pred"] = comparison["final_pred"].map(LABEL_MAP)
    comparison["raw_pred"] = comparison["raw_pred"].map(LABEL_MAP)
    comparison["english_only_pred"] = comparison["english_only_pred"].map(LABEL_MAP)
    comparison["age"] = pd.to_numeric(
        comparison["subject_id"].str.extract(r"^([4-9])\d{2}$", expand=False),
        errors="coerce",
    ).astype("Int64")

    output_columns = [
        "subject_id",
        "y_true",
        "final_score",
        "final_pred",
        "raw_score",
        "raw_pred",
        "english_only_score",
        "english_only_pred",
        "age",
    ]
    comparison = comparison[output_columns].sort_values("subject_id", key=lambda values: values.astype(int))
    if comparison.isna().any().any():
        missing = comparison.columns[comparison.isna().any()].tolist()
        raise ValueError(f"Consolidated comparison contains missing values in: {missing}")
    return comparison.reset_index(drop=True)


def main() -> None:
    comparison = build_comparison()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(comparison)} aligned English predictions to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
