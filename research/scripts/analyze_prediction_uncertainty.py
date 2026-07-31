from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "research" / "research_results" / "CSV"
MULTILINGUAL_DIR = RESULTS_DIR / "multilingual_transfer"
LANGUAGE_SHIFT_DIR = RESULTS_DIR / "language_shift"

DEFAULT_FINAL = MULTILINGUAL_DIR / "multilingual_bidirectional_logreg_pca_centroid_predictions.csv"
DEFAULT_RAW = MULTILINGUAL_DIR / "multilingual_bidirectional_logreg_pca_raw_predictions.csv"
DEFAULT_ENGLISH_ONLY = LANGUAGE_SHIFT_DIR / "language_shift_transfer_subject_predictions.csv"
DEFAULT_CZECH_OUTPUT = MULTILINGUAL_DIR / "czech_final_loso_predictions.csv"
DEFAULT_REPORT_OUTPUT = MULTILINGUAL_DIR / "prediction_uncertainty_report.json"

POSITIVE_LABEL = "Patients"
CONFIDENCE_LEVEL = 0.95


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")


def _validate_prediction_frame(frame: pd.DataFrame, expected_rows: int, name: str) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows for {name}, found {len(frame)}")
    duplicates = frame.loc[frame["subject_id"].duplicated(), "subject_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate subject_id values in {name}: {duplicates}")
    required = ["subject_id", "label", "score_patients", "pred_label"]
    if frame[required].isna().any().any():
        raise ValueError(f"Missing prediction values in {name}")
    unknown_labels = set(frame["label"]).difference({"Healthy", "Patients"})
    unknown_predictions = set(frame["pred_label"]).difference({"Healthy", "Patients"})
    if unknown_labels or unknown_predictions:
        raise ValueError(
            f"Unsupported labels in {name}: true={sorted(unknown_labels)}, predicted={sorted(unknown_predictions)}"
        )


def _load_multilingual(path: Path, target_domain: str, alignment_method: str, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    _require_columns(
        frame,
        {
            "subject_id",
            "label",
            "score_patients",
            "pred_label",
            "model_type",
            "alignment_method",
            "target_domain",
        },
        path,
    )
    frame = frame.loc[
        frame["target_domain"].eq(target_domain)
        & frame["model_type"].eq("logreg_pca")
        & frame["alignment_method"].eq(alignment_method)
    ].copy()
    _validate_prediction_frame(frame, expected_rows, f"{target_domain} logreg_pca + {alignment_method}")
    return frame


def _load_english_only(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    _require_columns(
        frame,
        {"subject_id", "label", "score_patients", "pred_label", "strategy", "alignment_method", "variant"},
        path,
    )
    frame = frame.loc[
        frame["strategy"].eq("english_loso_logreg")
        & frame["alignment_method"].eq("raw")
        & frame["variant"].eq("raw")
    ].copy()
    numeric_labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["label"] = numeric_labels.map({0: "Healthy", 1: "Patients"})
    _validate_prediction_frame(frame, 48, "English-only logreg raw")
    return frame


def _binary_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = frame["label"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    y_score = frame["score_patients"].astype(float).to_numpy()
    y_pred = frame["pred_label"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    return y_true, y_score, y_pred


def wilson_interval(successes: int, total: int, confidence_level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * sqrt((proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def stratified_f1_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repetitions: int,
    seed: int,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float]:
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Stratified bootstrap requires both classes")

    bootstrap_values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        bootstrap_values[index] = f1_score(y_true[sampled], y_pred[sampled], pos_label=1, zero_division=0)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(bootstrap_values, [alpha / 2.0, 1.0 - alpha / 2.0])
    estimate = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return float(estimate), float(lower), float(upper)


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks + 1.0
    return result


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    model_count, sample_count = predictions.shape
    negative_count = sample_count - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong requires at least two observations in each class")

    positive_scores = predictions[:, :positive_count]
    negative_scores = predictions[:, positive_count:]
    positive_ranks = np.empty((model_count, positive_count), dtype=float)
    negative_ranks = np.empty((model_count, negative_count), dtype=float)
    all_ranks = np.empty((model_count, sample_count), dtype=float)

    for model_index in range(model_count):
        positive_ranks[model_index] = _compute_midrank(positive_scores[model_index])
        negative_ranks[model_index] = _compute_midrank(negative_scores[model_index])
        all_ranks[model_index] = _compute_midrank(predictions[model_index])

    aucs = all_ranks[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (all_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1.0 - (all_ranks[:, positive_count:] - negative_ranks) / positive_count
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance


def delong_statistics(y_true: np.ndarray, score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    score_matrix = np.atleast_2d(np.asarray(score_matrix, dtype=float))
    if score_matrix.shape[1] != len(y_true):
        raise ValueError("DeLong score matrix and labels have incompatible lengths")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("DeLong requires binary labels encoded as 0 and 1")

    order = np.argsort(-y_true, kind="stable")
    positive_count = int(y_true.sum())
    return _fast_delong(score_matrix[:, order], positive_count)


def delong_auc_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float, float]:
    aucs, covariance = delong_statistics(y_true, y_score)
    variance = max(float(covariance[0, 0]), 0.0)
    standard_error = sqrt(variance)
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    lower = max(0.0, float(aucs[0]) - z * standard_error)
    upper = min(1.0, float(aucs[0]) + z * standard_error)
    return float(aucs[0]), lower, upper, standard_error


def delong_paired_test(y_true: np.ndarray, first_score: np.ndarray, second_score: np.ndarray) -> dict[str, float]:
    aucs, covariance = delong_statistics(y_true, np.vstack([first_score, second_score]))
    contrast = np.array([1.0, -1.0])
    variance = max(float(contrast @ covariance @ contrast), 0.0)
    difference = float(aucs[0] - aucs[1])
    if variance == 0.0:
        z_value = 0.0 if difference == 0.0 else float("inf") * np.sign(difference)
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z_value = difference / sqrt(variance)
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(z_value)))
    standard_error = sqrt(variance)
    critical_value = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2.0)
    return {
        "auc_first": float(aucs[0]),
        "auc_second": float(aucs[1]),
        "auc_difference_first_minus_second": difference,
        "difference_standard_error": standard_error,
        "difference_ci_lower": difference - critical_value * standard_error,
        "difference_ci_upper": difference + critical_value * standard_error,
        "z": float(z_value),
        "p_value_two_sided": float(p_value),
    }


def mcnemar_exact(y_true: np.ndarray, first_pred: np.ndarray, second_pred: np.ndarray) -> dict[str, int | float]:
    first_correct = first_pred == y_true
    second_correct = second_pred == y_true
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, value) for value in range(min(first_only, second_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_pairs": discordant,
        "p_value_two_sided_exact": float(p_value),
    }


def _paired_english_comparison(final: pd.DataFrame, comparator: pd.DataFrame, comparator_name: str) -> dict:
    merged = final[["subject_id", "label", "score_patients", "pred_label"]].merge(
        comparator[["subject_id", "label", "score_patients", "pred_label"]],
        on="subject_id",
        suffixes=("_final", "_comparator"),
        validate="one_to_one",
    )
    if len(merged) != 48:
        raise ValueError(f"Expected 48 aligned English rows for {comparator_name}, found {len(merged)}")
    if not merged["label_final"].equals(merged["label_comparator"]):
        raise ValueError(f"Real labels disagree in English comparison with {comparator_name}")

    y_true = merged["label_final"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    final_score = merged["score_patients_final"].astype(float).to_numpy()
    comparator_score = merged["score_patients_comparator"].astype(float).to_numpy()
    final_pred = merged["pred_label_final"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    comparator_pred = merged["pred_label_comparator"].eq(POSITIVE_LABEL).astype(int).to_numpy()
    return {
        "n_subjects": int(len(merged)),
        "positive_class": POSITIVE_LABEL,
        "first_model": "centroid_shift + logreg_pca",
        "second_model": comparator_name,
        "delong": delong_paired_test(y_true, final_score, comparator_score),
        "mcnemar": mcnemar_exact(y_true, final_pred, comparator_pred),
    }


def _interval_report(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict:
    y_true, y_score, y_pred = _binary_columns(frame)
    accuracy = float(accuracy_score(y_true, y_pred))
    accuracy_lower, accuracy_upper = wilson_interval(int(np.sum(y_true == y_pred)), len(y_true))
    f1, f1_lower, f1_upper = stratified_f1_bootstrap(
        y_true,
        y_pred,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    auc, auc_lower, auc_upper, auc_standard_error = delong_auc_interval(y_true, y_score)

    sklearn_auc = float(roc_auc_score(y_true, y_score))
    if not np.isclose(auc, sklearn_auc, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"DeLong AUC {auc} disagrees with sklearn AUC {sklearn_auc}")

    return {
        "n_subjects": int(len(frame)),
        "n_patients": int(y_true.sum()),
        "n_healthy": int(len(y_true) - y_true.sum()),
        "accuracy": {
            "estimate": accuracy,
            "ci_lower": accuracy_lower,
            "ci_upper": accuracy_upper,
            "method": "Wilson score interval",
        },
        "f1_patients": {
            "estimate": f1,
            "ci_lower": f1_lower,
            "ci_upper": f1_upper,
            "method": "Stratified percentile bootstrap",
        },
        "auc_roc": {
            "estimate": auc,
            "ci_lower": auc_lower,
            "ci_upper": auc_upper,
            "standard_error": auc_standard_error,
            "method": "DeLong normal-approximation interval",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute uncertainty intervals for Czech LOSO predictions and paired English model tests."
    )
    parser.add_argument("--final-predictions", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--raw-predictions", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--english-only-predictions", type=Path, default=DEFAULT_ENGLISH_ONLY)
    parser.add_argument("--czech-output", type=Path, default=DEFAULT_CZECH_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_czech = _load_multilingual(args.final_predictions, "czech", "centroid_shift", 98)
    final_english = _load_multilingual(args.final_predictions, "english", "centroid_shift", 48)
    raw_english = _load_multilingual(args.raw_predictions, "english", "raw", 48)
    english_only = _load_english_only(args.english_only_predictions)

    czech_output = final_czech[["subject_id", "label", "score_patients", "pred_label"]].rename(
        columns={"label": "y_true", "score_patients": "y_score", "pred_label": "y_pred"}
    )
    czech_intervals = _interval_report(final_czech, args.bootstrap_repetitions, args.seed)
    english_intervals = _interval_report(final_english, args.bootstrap_repetitions, args.seed)

    report = {
        "config": {
            "positive_class": POSITIVE_LABEL,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.seed,
            "final_predictions": str(args.final_predictions),
            "raw_predictions": str(args.raw_predictions),
            "english_only_predictions": str(args.english_only_predictions),
        },
        "czech_final": czech_intervals,
        "english_final": english_intervals,
        "english_paired_comparisons": {
            "final_vs_raw": _paired_english_comparison(final_english, raw_english, "raw + logreg_pca"),
            "final_vs_english_only": _paired_english_comparison(final_english, english_only, "English-only"),
        },
    }

    args.czech_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    czech_output.to_csv(args.czech_output, index=False)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Czech LOSO predictions: {args.czech_output}")
    print(f"Statistical report: {args.report_output}")
    print(
        "Czech final: "
        f"Accuracy={czech_intervals['accuracy']['estimate']:.4f} "
        f"[{czech_intervals['accuracy']['ci_lower']:.4f}, {czech_intervals['accuracy']['ci_upper']:.4f}], "
        f"F1={czech_intervals['f1_patients']['estimate']:.4f} "
        f"[{czech_intervals['f1_patients']['ci_lower']:.4f}, {czech_intervals['f1_patients']['ci_upper']:.4f}], "
        f"AUC={czech_intervals['auc_roc']['estimate']:.4f} "
        f"[{czech_intervals['auc_roc']['ci_lower']:.4f}, {czech_intervals['auc_roc']['ci_upper']:.4f}]"
    )
    print(
        "English final: "
        f"Accuracy={english_intervals['accuracy']['estimate']:.4f} "
        f"[{english_intervals['accuracy']['ci_lower']:.4f}, {english_intervals['accuracy']['ci_upper']:.4f}], "
        f"F1={english_intervals['f1_patients']['estimate']:.4f} "
        f"[{english_intervals['f1_patients']['ci_lower']:.4f}, {english_intervals['f1_patients']['ci_upper']:.4f}], "
        f"AUC={english_intervals['auc_roc']['estimate']:.4f} "
        f"[{english_intervals['auc_roc']['ci_lower']:.4f}, {english_intervals['auc_roc']['ci_upper']:.4f}]"
    )


if __name__ == "__main__":
    main()
