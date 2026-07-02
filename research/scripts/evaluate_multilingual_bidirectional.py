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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from research.src.config import CONFIG
from research.src.train import LabelEncodedClassifier


REPORT_DIR = CONFIG.paths.reports_dir / "CSV" / "multilingual_transfer"


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

    return torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope]).numpy().astype(np.float64)


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


def _healthy_centroid(frame: pd.DataFrame) -> np.ndarray:
    healthy = frame.loc[frame["label"] == "Healthy", "embedding"]
    if healthy.empty:
        raise ValueError("No Healthy rows available for centroid computation.")
    return np.vstack(healthy.to_list()).mean(axis=0)


def _shift_source_toward_target(source_frame: pd.DataFrame, target_train_frame: pd.DataFrame) -> pd.DataFrame:
    src_h = _healthy_centroid(source_frame)
    tgt_h = _healthy_centroid(target_train_frame)
    shifted = source_frame.copy()
    shifted["embedding"] = shifted["embedding"].apply(lambda z: z - src_h + tgt_h)
    return shifted


def _fit_model(model_type: str, X: np.ndarray, y: pd.Series):
    if model_type == "logreg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs")),
        ])
    elif model_type == "linear_svm":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=42)),
        ])
    elif model_type == "logreg_pca":
        n_components = int(min(64, X.shape[0] - 1, X.shape[1]))
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, solver="lbfgs")),
        ])
    elif model_type == "mlp_small":
        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LabelEncodedClassifier(
                    MLPClassifier(
                        hidden_layer_sizes=(128, 32),
                        activation="relu",
                        solver="adam",
                        alpha=5e-4,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=600,
                        early_stopping=False,
                        batch_size="auto",
                        n_iter_no_change=20,
                        random_state=42,
                    )
                ),
            ),
        ])
    elif model_type == "mlp_medium":
        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LabelEncodedClassifier(
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        learning_rate="adaptive",
                        learning_rate_init=7.5e-4,
                        max_iter=800,
                        early_stopping=False,
                        batch_size="auto",
                        n_iter_no_change=30,
                        random_state=42,
                    )
                ),
            ),
        ])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps") and "clf" in model.named_steps:
        classes = list(getattr(model.named_steps["clf"], "classes_", []))
    if "Patients" in classes:
        return classes.index("Patients")
    if 1 in classes:
        return classes.index(1)
    if len(classes) == 2:
        return 1
    raise ValueError(f"Could not resolve positive class in {classes}")


def _scores_to_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int | None]:
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


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float | int | None]]:
    best_thr = 0.5
    best = None
    for thr in np.unique(np.round(y_score, 4)):
        m = _scores_to_metrics(y_true, y_score, threshold=float(thr))
        if best is None or float(m["balanced_accuracy"]) > float(best["balanced_accuracy"]):
            best = m
            best_thr = float(thr)
    return best_thr, best or _scores_to_metrics(y_true, y_score, threshold=0.5)


def _run_target_loso(source_subject: pd.DataFrame, target_subject: pd.DataFrame, model_type: str, alignment_method: str) -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    target_ids = sorted(set(target_subject["subject_id"].astype(str).tolist()))

    for test_subject in target_ids:
        tgt_train = target_subject.loc[target_subject["subject_id"].astype(str) != test_subject].copy()
        tgt_test = target_subject.loc[target_subject["subject_id"].astype(str) == test_subject].copy()

        src_train = source_subject.copy()
        if alignment_method == "centroid_shift":
            src_train = _shift_source_toward_target(src_train, tgt_train)

        train_frame = pd.concat([src_train, tgt_train], ignore_index=True)
        X_train = np.vstack(train_frame["embedding"].to_list())
        y_train = train_frame["label"].reset_index(drop=True)
        model = _fit_model(model_type, X_train, y_train)
        positive_idx = _positive_index(model)

        train_scores = model.predict_proba(np.vstack(tgt_train["embedding"].to_list()))[:, positive_idx]
        train_y = (tgt_train["label"] == "Patients").astype(int).to_numpy()
        best_thr, _ = _best_threshold(train_y, train_scores)

        test_score = float(model.predict_proba(np.vstack(tgt_test["embedding"].to_list()))[:, positive_idx][0])
        rows.append(
            {
                "subject_id": str(tgt_test.iloc[0]["subject_id"]),
                "label": str(tgt_test.iloc[0]["label"]),
                "score_patients": test_score,
                "pred_label": "Patients" if test_score >= best_thr else "Healthy",
                "threshold_fold": best_thr,
                "model_type": model_type,
                "alignment_method": alignment_method,
            }
        )

    pred = pd.DataFrame(rows)
    y_true = (pred["label"] == "Patients").astype(int).to_numpy()
    y_score = pred["score_patients"].to_numpy()
    y_pred = (pred["pred_label"] == "Patients").astype(int).to_numpy()
    metrics = {
        "n_rows": int(len(pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
    }
    best_thr_global, best_global = _best_threshold(y_true, y_score)
    metrics["best_threshold_global"] = best_thr_global
    metrics["best_balanced_accuracy_global"] = best_global["balanced_accuracy"]
    metrics["best_recall_global"] = best_global["recall"]
    metrics["best_precision_global"] = best_global["precision"]
    return metrics, pred


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
    cz_subject = _aggregate_subject_embeddings(_load_manifest_embeddings(args.cz_sequence_manifest, args.layer_policy))
    en_subject = _aggregate_subject_embeddings(_load_manifest_embeddings(args.en_sequence_manifest, args.layer_policy))

    en_metrics, en_pred = _run_target_loso(cz_subject, en_subject, args.model_type, args.alignment_method)
    cz_metrics, cz_pred = _run_target_loso(en_subject, cz_subject, args.model_type, args.alignment_method)

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
