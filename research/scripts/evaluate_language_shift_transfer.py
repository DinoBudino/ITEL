from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research.src.config import CONFIG


REPORT_DIR = CONFIG.paths.reports_dir / "CSV" / "language_shift"


def _load_sequence_embedding(sequence_path: str | Path, layer_policy: str) -> np.ndarray:
    payload = torch.load(sequence_path, map_location="cpu", weights_only=True)
    seq = payload["sequence"].float()

    if seq.dim() == 3:
        if layer_policy == "last":
            seq = seq[-1]
        elif layer_policy == "mean_layers":
            seq = seq.mean(dim=0)
        else:
            raise ValueError(f"Unsupported layer_policy for 3D sequence: {layer_policy}")

    if seq.dim() != 2:
        raise ValueError(f"Expected 2D sequence after layer selection, got shape {tuple(seq.shape)}")

    time_steps = seq.size(0)
    mean_emb = seq.mean(dim=0)
    std_emb = seq.std(dim=0)
    first_25 = seq[: max(1, time_steps // 4)].mean(dim=0)
    last_25 = seq[-max(1, time_steps // 4) :].mean(dim=0)
    delta = first_25 - last_25

    x = torch.arange(time_steps, dtype=torch.float32)
    x_centered = x - x.mean()
    x_norm = torch.sqrt((x_centered ** 2).sum())
    if time_steps < 2 or float(x_norm) == 0.0:
        slope = torch.zeros(seq.size(1), dtype=torch.float32)
    else:
        slope_vals = []
        for dim in range(seq.size(1)):
            y = seq[:, dim]
            y_centered = y - y.mean()
            y_norm = torch.sqrt((y_centered ** 2).sum())
            if float(y_norm) == 0.0:
                slope_vals.append(torch.tensor(0.0))
            else:
                slope_vals.append(torch.dot(x_centered, y_centered) / (x_norm * y_norm))
        slope = torch.stack(slope_vals)

    emb = torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope]).numpy().astype(np.float64)
    return emb


def _load_manifest_embeddings(manifest_path: str | Path, layer_policy: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        emb = _load_sequence_embedding(getattr(row, "sequence_path"), layer_policy=layer_policy)
        rows.append(
            {
                "sample_id": getattr(row, "sample_id"),
                "subject_id": str(getattr(row, "subject_id")),
                "label": str(getattr(row, "label")),
                "embedding": emb,
            }
        )
    return pd.DataFrame(rows)


def _healthy_centroid(frame: pd.DataFrame) -> np.ndarray:
    healthy = frame.loc[frame["label"] == "Healthy", "embedding"]
    if healthy.empty:
        raise ValueError("No Healthy samples available to compute centroid.")
    return np.vstack(healthy.to_list()).mean(axis=0)


def _aggregate_subject_embeddings(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject_id, label), group in frame.groupby(["subject_id", "label"], dropna=False):
        emb = np.vstack(group["embedding"].to_list()).mean(axis=0)
        rows.append({
            "subject_id": str(subject_id),
            "label": str(label),
            "n_samples": int(len(group)),
            "embedding": emb,
        })
    return pd.DataFrame(rows)


def _shift_english(frame: pd.DataFrame, cz_healthy_centroid: np.ndarray, en_healthy_centroid: np.ndarray) -> pd.DataFrame:
    shifted = frame.copy()
    shifted["embedding"] = shifted["embedding"].apply(lambda z: z - en_healthy_centroid + cz_healthy_centroid)
    return shifted


def _variance_vector(embeddings: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    var = np.var(embeddings, axis=0)
    return np.clip(var, eps, None)


def _coral_shift_english(frame: pd.DataFrame, cz_frame: pd.DataFrame, en_healthy_centroid: np.ndarray, cz_healthy_centroid: np.ndarray) -> pd.DataFrame:
    cz_healthy = np.vstack(cz_frame.loc[cz_frame["label"] == "Healthy", "embedding"].to_list())
    en_healthy = np.vstack(frame.loc[frame["label"] == "Healthy", "embedding"].to_list())
    cz_cov = _covariance_matrix(cz_healthy)
    en_cov = _covariance_matrix(en_healthy)

    en_inv_sqrt = _matrix_inv_sqrt(en_cov)
    cz_sqrt = _matrix_sqrt(cz_cov)

    shifted = frame.copy()

    def transform(z: np.ndarray) -> np.ndarray:
        centered = z - en_healthy_centroid
        aligned = centered @ en_inv_sqrt @ cz_sqrt
        return aligned + cz_healthy_centroid

    shifted["embedding"] = shifted["embedding"].apply(transform)
    return shifted


def _build_alignment_transform(cz_frame: pd.DataFrame, en_reference_frame: pd.DataFrame, method: str):
    if method == "raw":
        return lambda z: z

    cz_subject = _aggregate_subject_embeddings(cz_frame)
    en_subject = _aggregate_subject_embeddings(en_reference_frame)
    cz_h = _healthy_centroid(cz_subject)
    en_h = _healthy_centroid(en_subject)

    if method == "centroid_shift":
        return lambda z: z - en_h + cz_h

    if method == "coral_shift":
        cz_healthy = np.vstack(cz_subject.loc[cz_subject["label"] == "Healthy", "embedding"].to_list())
        en_healthy = np.vstack(en_subject.loc[en_subject["label"] == "Healthy", "embedding"].to_list())
        cz_var = _variance_vector(cz_healthy)
        en_var = _variance_vector(en_healthy)
        scale = np.sqrt(cz_var / en_var)

        def transform(z: np.ndarray) -> np.ndarray:
            centered = z - en_h
            aligned = centered * scale
            return aligned + cz_h

        return transform

    raise ValueError(f"Unsupported alignment method: {method}")


def _scores_to_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | int | None] = {
        "n_rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    return metrics


def _threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[dict[str, float | int | None], list[dict[str, float | int | None]]]:
    rows = []
    best = None
    for thr in np.arange(0.05, 1.0, 0.01):
        thr = round(float(thr), 2)
        m = _scores_to_metrics(y_true, y_score, threshold=thr)
        rows.append(m)
        if best is None or float(m["balanced_accuracy"]) > float(best["balanced_accuracy"]):
            best = m
    return best or _scores_to_metrics(y_true, y_score, threshold=0.5), rows


def _build_classifier(classifier_name: str, c_value: float = 1.0, gamma: str = "scale"):
    if classifier_name == "logreg":
        return LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced", random_state=42)
    if classifier_name == "svm_rbf":
        return SVC(C=c_value, kernel="rbf", gamma=gamma, probability=True, class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def _select_best_c(X_train: np.ndarray, y_train: np.ndarray, classifier_name: str, c_values: tuple[float, ...]) -> float:
    # With one row per subject in English, nested LOSO on single-subject folds is too noisy.
    # Use small stratified 3-fold CV when class counts allow it; otherwise fall back to 1.0.
    from sklearn.model_selection import StratifiedKFold

    class_counts = np.bincount(y_train.astype(int), minlength=2)
    if class_counts.min() < 3:
        return 1.0

    best_c = c_values[0]
    best_score = -1.0
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for c_val in c_values:
        fold_scores: list[float] = []
        for tr_idx, vl_idx in splitter.split(X_train, y_train):
            clf = _build_classifier(classifier_name, c_value=c_val)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_train[tr_idx], y_train[tr_idx])
            pred = clf.predict(X_train[vl_idx])
            fold_scores.append(float(balanced_accuracy_score(y_train[vl_idx], pred)))
        mean_score = float(np.mean(fold_scores)) if fold_scores else -1.0
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val
    return best_c


def _english_only_loso(frame: pd.DataFrame, cz_frame: pd.DataFrame, classifier_name: str, alignment_method: str, c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)) -> tuple[dict, pd.DataFrame]:
    subject_frame = _aggregate_subject_embeddings(frame)
    subjects = sorted(set(subject_frame["subject_id"].astype(str).tolist()))
    X_all = np.vstack(subject_frame["embedding"].to_list())
    y_all = (subject_frame["label"] == "Patients").astype(int).to_numpy()
    subj_all = subject_frame["subject_id"].astype(str).to_numpy()

    rows: list[dict[str, object]] = []
    for test_subject in subjects:
        train_mask = subj_all != test_subject
        test_mask = subj_all == test_subject
        train_frame = subject_frame.loc[train_mask].copy()
        test_frame = subject_frame.loc[test_mask].copy()
        transform = _build_alignment_transform(cz_frame, train_frame, alignment_method)
        train_frame["embedding"] = train_frame["embedding"].apply(transform)
        test_frame["embedding"] = test_frame["embedding"].apply(transform)

        X_train_raw = np.vstack(train_frame["embedding"].to_list())
        X_test_raw = np.vstack(test_frame["embedding"].to_list())
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        best_c = _select_best_c(X_train, y_train, classifier_name, c_values)

        clf = _build_classifier(classifier_name, c_value=best_c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
        score = float(clf.predict_proba(X_test)[0, list(clf.classes_).index(1)])
        rows.append(
            {
                "subject_id": test_subject,
                "label": int(y_test[0]),
                "score_patients": score,
                "pred_label": "Patients" if score >= 0.5 else "Healthy",
                "strategy": f"english_loso_{classifier_name}",
                "alignment_method": alignment_method,
                "best_C": best_c,
            }
        )

    pred = pd.DataFrame(rows)
    y_true = pred["label"].to_numpy()
    y_score = pred["score_patients"].to_numpy()
    metrics = _scores_to_metrics(y_true, y_score, threshold=0.5)
    return metrics, pred


def _fit_cz_transfer(cz_frame: pd.DataFrame, en_frame: pd.DataFrame, classifier_name: str, alignment_method: str, train_unit: str, threshold: float = 0.5) -> tuple[dict, pd.DataFrame]:
    if train_unit == "subject":
        cz_train_frame = _aggregate_subject_embeddings(cz_frame)
    elif train_unit == "sample":
        cz_train_frame = cz_frame.copy()
    else:
        raise ValueError(f"Unsupported train_unit: {train_unit}")

    en_subject = _aggregate_subject_embeddings(en_frame)

    X_cz = np.vstack(cz_train_frame["embedding"].to_list())
    y_cz = (cz_train_frame["label"] == "Patients").astype(int).to_numpy()
    scaler = StandardScaler()
    X_cz_s = scaler.fit_transform(X_cz)
    clf = _build_classifier(classifier_name, c_value=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_cz_s, y_cz)

    rows: list[dict[str, object]] = []
    for test_subject in sorted(set(en_subject["subject_id"].astype(str).tolist())):
        ref_frame = en_subject.loc[en_subject["subject_id"].astype(str) != test_subject].copy()
        test_frame = en_subject.loc[en_subject["subject_id"].astype(str) == test_subject].copy()
        transform = _build_alignment_transform(cz_frame, ref_frame, alignment_method)
        test_emb = transform(test_frame.iloc[0]["embedding"])
        test_emb_s = scaler.transform(test_emb.reshape(1, -1))
        score = float(clf.predict_proba(test_emb_s)[0, list(clf.classes_).index(1)])
        rows.append(
            {
                "subject_id": str(test_frame.iloc[0]["subject_id"]),
                "label": str(test_frame.iloc[0]["label"]),
                "score_patients": score,
                "pred_label": "Patients" if score >= threshold else "Healthy",
                "strategy": f"cz_transfer_{classifier_name}",
                "alignment_method": alignment_method,
                "train_unit": train_unit,
            }
        )

    pred = pd.DataFrame(rows)
    y_en = (pred["label"] == "Patients").astype(int).to_numpy()
    en_scores = pred["score_patients"].to_numpy()
    metrics = _scores_to_metrics(y_en, en_scores, threshold=threshold)
    best, sweep = _threshold_sweep(y_en, en_scores)
    metrics["best_threshold_on_english"] = best["threshold"]
    metrics["best_balanced_accuracy_on_english"] = best["balanced_accuracy"]
    metrics["best_recall_on_english"] = best["recall"]
    metrics["best_precision_on_english"] = best["precision"]
    metrics["train_unit"] = train_unit
    metrics["threshold_sweep"] = sweep
    return metrics, pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw vs centroid-shift transfer using HuBERT sequence pooling + logistic regression.")
    parser.add_argument(
        "--cz-sequence-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_fixed_2s_multilingual_light.csv"),
        help="Czech sequence manifest.",
    )
    parser.add_argument(
        "--en-sequence-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_v2_fixed_2s.csv"),
        help="English sequence manifest.",
    )
    parser.add_argument(
        "--layer-policy",
        choices=["last", "mean_layers"],
        default="last",
        help="How to reduce 12-layer sequence tensors if present.",
    )
    parser.add_argument(
        "--output-json",
        default=str(REPORT_DIR / "language_shift_transfer_report.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--output-subjects-csv",
        default=str(REPORT_DIR / "language_shift_transfer_subject_predictions.csv"),
        help="Where to write per-subject predictions.",
    )
    parser.add_argument(
        "--alignment-methods",
        nargs="+",
        default=["raw", "centroid_shift", "coral_shift"],
        choices=["raw", "centroid_shift", "coral_shift"],
        help="Embedding alignment methods to compare.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["logreg", "svm_rbf"],
        choices=["logreg", "svm_rbf"],
        help="Classifiers to compare after raw/shift adaptation.",
    )
    parser.add_argument(
        "--cz-train-units",
        nargs="+",
        default=["subject", "sample"],
        choices=["subject", "sample"],
        help="How to train the Czech-side classifier before transferring to English.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cz_frame = _load_manifest_embeddings(args.cz_sequence_manifest, layer_policy=args.layer_policy)
    en_frame = _load_manifest_embeddings(args.en_sequence_manifest, layer_policy=args.layer_policy)

    cz_h = _healthy_centroid(_aggregate_subject_embeddings(cz_frame))
    en_h = _healthy_centroid(_aggregate_subject_embeddings(en_frame))

    english_only_report: dict[str, dict[str, dict]] = {}
    transfer_report: dict[str, dict[str, dict[str, dict]]] = {}
    pred_frames: list[pd.DataFrame] = []

    for classifier_name in args.classifiers:
        english_only_report[classifier_name] = {}
        transfer_report[classifier_name] = {}
        for alignment_name in args.alignment_methods:
            loso_metrics, loso_pred = _english_only_loso(en_frame, cz_frame, classifier_name=classifier_name, alignment_method=alignment_name)

            english_only_report[classifier_name][alignment_name] = loso_metrics
            transfer_report[classifier_name][alignment_name] = {}

            for train_unit in args.cz_train_units:
                transfer_metrics, transfer_pred = _fit_cz_transfer(
                    cz_frame,
                    en_frame,
                    classifier_name=classifier_name,
                    alignment_method=alignment_name,
                    train_unit=train_unit,
                )
                transfer_report[classifier_name][alignment_name][train_unit] = transfer_metrics
                transfer_pred["train_unit"] = train_unit
                transfer_pred["variant"] = alignment_name
                pred_frames.append(transfer_pred)

            loso_pred["variant"] = alignment_name
            pred_frames.append(loso_pred)

    report = {
        "config": {
            "cz_sequence_manifest": args.cz_sequence_manifest,
            "en_sequence_manifest": args.en_sequence_manifest,
            "layer_policy": args.layer_policy,
        },
        "datasets": {
            "cz_n_rows": int(len(cz_frame)),
            "cz_n_subjects": int(cz_frame["subject_id"].nunique()),
            "cz_labels": cz_frame["label"].value_counts().to_dict(),
            "en_n_rows": int(len(en_frame)),
            "en_n_subjects": int(en_frame["subject_id"].nunique()),
            "en_labels": en_frame["label"].value_counts().to_dict(),
        },
        "centroids": {
            "cz_healthy_norm": float(np.linalg.norm(cz_h)),
            "en_healthy_norm": float(np.linalg.norm(en_h)),
            "shift_magnitude": float(np.linalg.norm(cz_h - en_h)),
        },
        "alignment_methods": args.alignment_methods,
        "english_only_loso": english_only_report,
        "czech_to_english_transfer": transfer_report,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_subjects_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.concat(pred_frames, ignore_index=True).to_csv(out_csv, index=False)

    print("===== LANGUAGE SHIFT TRANSFER REPORT =====")
    print(f"CZ rows={len(cz_frame)}  EN rows={len(en_frame)}  shift_norm={np.linalg.norm(cz_h - en_h):.4f}")
    for classifier_name in args.classifiers:
        print(f"\nEnglish-only LOSO [{classifier_name}]")
        for alignment_name in args.alignment_methods:
            m = english_only_report[classifier_name][alignment_name]
            print(f"  {alignment_name:<14} bal_acc={m['balanced_accuracy']:.4f}  auc={m['auc']:.4f}  recall={m['recall']:.4f}")
        print(f"\nCzech -> English transfer [{classifier_name}]")
        for alignment_name in args.alignment_methods:
            for train_unit in args.cz_train_units:
                m = transfer_report[classifier_name][alignment_name][train_unit]
                print(f"  {alignment_name:<14} train={train_unit:<7} bal_acc={m['balanced_accuracy']:.4f}  auc={m['auc']:.4f}  recall={m['recall']:.4f}  best_EN_bal_acc={m['best_balanced_accuracy_on_english']:.4f} @thr={m['best_threshold_on_english']}")
    print(f"\nJSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
