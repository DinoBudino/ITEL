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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.src.config import CONFIG
from research.src.train import LabelEncodedClassifier, MODEL_TRAINERS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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
        rows.append(
            {
                "subject_id": str(subject_id),
                "label": str(label),
                "n_samples": int(len(group)),
                "embedding": emb,
            }
        )
    return pd.DataFrame(rows)


def _healthy_centroid(frame: pd.DataFrame) -> np.ndarray:
    healthy = frame.loc[frame["label"] == "Healthy", "embedding"]
    if healthy.empty:
        raise ValueError("No Healthy rows available for centroid computation.")
    return np.vstack(healthy.to_list()).mean(axis=0)


def _shift_czech_toward_english(cz_frame: pd.DataFrame, en_train_frame: pd.DataFrame) -> pd.DataFrame:
    cz_h = _healthy_centroid(cz_frame)
    en_h = _healthy_centroid(en_train_frame)
    shifted = cz_frame.copy()
    shifted["embedding"] = shifted["embedding"].apply(lambda z: z - cz_h + en_h)
    return shifted


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


def _fit_model(model_type: str, X: np.ndarray, y: pd.Series):
    if model_type in {"logreg_pca", "linear_svm_pca", "mlp_pca"}:
        n_components = int(min(64, X.shape[0] - 1, X.shape[1]))
        if n_components < 2:
            raise ValueError(f"Not enough samples for PCA in model {model_type}: {X.shape}")
        if model_type == "logreg_pca":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=42)),
                    ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=4000, solver="lbfgs")),
                ]
            )
        elif model_type == "linear_svm_pca":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=42)),
                    ("clf", SVC(kernel="linear", C=0.5, class_weight="balanced", probability=True, random_state=42)),
                ]
            )
        else:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=42)),
                    (
                        "clf",
                        LabelEncodedClassifier(
                            MLPClassifier(
                                hidden_layer_sizes=(128, 32),
                                activation="relu",
                                solver="adam",
                                alpha=1e-2,
                                learning_rate="adaptive",
                                learning_rate_init=1e-3,
                                max_iter=800,
                                early_stopping=False,
                                batch_size="auto",
                                n_iter_no_change=30,
                                random_state=42,
                            )
                        ),
                    ),
                ]
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return model

    trainer = MODEL_TRAINERS[model_type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trainer(X, y)


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


def _run_multilingual_loso(cz_subject: pd.DataFrame, en_subject: pd.DataFrame, model_type: str, alignment_method: str) -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    english_subjects = sorted(set(en_subject["subject_id"].astype(str).tolist()))

    for test_subject in english_subjects:
        en_train = en_subject.loc[en_subject["subject_id"].astype(str) != test_subject].copy()
        en_test = en_subject.loc[en_subject["subject_id"].astype(str) == test_subject].copy()

        cz_train = cz_subject.copy()
        if alignment_method == "centroid_shift":
            cz_train = _shift_czech_toward_english(cz_train, en_train)

        train_frame = pd.concat([cz_train, en_train], ignore_index=True)
        X_train = np.vstack(train_frame["embedding"].to_list())
        y_train = train_frame["label"].reset_index(drop=True)

        model = _fit_model(model_type, X_train, y_train)
        positive_idx = _positive_index(model)

        # choose threshold on EN-train only, with the same fitted multilingual model
        en_train_scores = model.predict_proba(np.vstack(en_train["embedding"].to_list()))[:, positive_idx]
        en_train_y = (en_train["label"] == "Patients").astype(int).to_numpy()
        best_thr, _ = _best_threshold(en_train_y, en_train_scores)

        test_scores = model.predict_proba(np.vstack(en_test["embedding"].to_list()))[:, positive_idx]
        score = float(test_scores[0])
        rows.append(
            {
                "subject_id": str(en_test.iloc[0]["subject_id"]),
                "label": str(en_test.iloc[0]["label"]),
                "score_patients": score,
                "pred_label": "Patients" if score >= best_thr else "Healthy",
                "model_type": model_type,
                "alignment_method": alignment_method,
                "threshold_fold": best_thr,
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
    parser = argparse.ArgumentParser(description="Benchmark multilingual CZ+EN transfer heads over subject-level pooled embeddings.")
    parser.add_argument("--cz-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural_multilingual_light.csv"))
    parser.add_argument("--en-sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_v2.csv"))
    parser.add_argument("--layer-policy", choices=["last", "mean_layers"], default="last")
    parser.add_argument("--alignment-methods", nargs="+", default=["raw", "centroid_shift"], choices=["raw", "centroid_shift"])
    parser.add_argument("--model-types", nargs="+", default=["logreg", "linear_svm", "logreg_pca", "mlp_small", "mlp_medium"], choices=tuple(MODEL_TRAINERS.keys()))
    parser.add_argument("--output-json", default=str(REPORT_DIR / "multilingual_cz_en_heads.json"))
    parser.add_argument("--output-csv", default=str(REPORT_DIR / "multilingual_cz_en_heads_predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cz_frame = _aggregate_subject_embeddings(_load_manifest_embeddings(args.cz_sequence_manifest, args.layer_policy))
    en_frame = _aggregate_subject_embeddings(_load_manifest_embeddings(args.en_sequence_manifest, args.layer_policy))

    leaderboard_rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    report: dict[str, object] = {
        "config": {
            "cz_sequence_manifest": args.cz_sequence_manifest,
            "en_sequence_manifest": args.en_sequence_manifest,
            "layer_policy": args.layer_policy,
            "alignment_methods": args.alignment_methods,
            "model_types": args.model_types,
        },
        "datasets": {
            "cz_subjects": int(cz_frame["subject_id"].nunique()),
            "cz_labels": cz_frame["label"].value_counts().to_dict(),
            "en_subjects": int(en_frame["subject_id"].nunique()),
            "en_labels": en_frame["label"].value_counts().to_dict(),
        },
        "results": {},
    }

    for model_type in args.model_types:
        report["results"][model_type] = {}
        print(f"\n===== Model: {model_type} =====")
        for alignment_method in args.alignment_methods:
            metrics, pred = _run_multilingual_loso(cz_frame, en_frame, model_type=model_type, alignment_method=alignment_method)
            report["results"][model_type][alignment_method] = metrics
            pred_frames.append(pred)
            leaderboard_rows.append(
                {
                    "model_type": model_type,
                    "alignment_method": alignment_method,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "auc": metrics["auc"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "best_balanced_accuracy_global": metrics["best_balanced_accuracy_global"],
                    "best_threshold_global": metrics["best_threshold_global"],
                }
            )
            print(
                f"  {alignment_method:<14} bal_acc={metrics['balanced_accuracy']:.4f}  auc={metrics['auc']:.4f}  "
                f"recall={metrics['recall']:.4f}  best_bal_acc={metrics['best_balanced_accuracy_global']:.4f} @thr={metrics['best_threshold_global']}"
            )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["best_balanced_accuracy_global", "auc", "balanced_accuracy"],
        ascending=[False, False, False],
    )
    report["leaderboard"] = leaderboard.to_dict(orient="records")

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.concat(pred_frames, ignore_index=True).to_csv(out_csv, index=False)

    print("\n===== TOP CONFIGURATIONS =====")
    print(leaderboard.head(10).to_string(index=False))
    print(f"\nJSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
