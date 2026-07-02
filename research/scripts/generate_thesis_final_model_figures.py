from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "research" / "research_results" / "CSV"
REPORTS_DIR = ROOT_DIR / "research" / "reports" / "thesis_chapter4"

BASE_METRICS_PATH = RESULTS_DIR / "metrics_v3_layermix.json"
FINAL_METRICS_PATH = RESULTS_DIR / "metrics_v3_layermix_hardneg_hnweight2x.json"
FINAL_PREDICTIONS_PATH = RESULTS_DIR / "predictions_v3_layermix_hardneg_hnweight2x.csv"


def savefig(path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_confusion_matrix(predictions: pd.DataFrame) -> None:
    y_true = predictions["label"].astype(int).to_numpy()
    y_pred = (predictions["pred_label"] == "Patients").astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=200)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Patients"])
    disp.plot(ax=ax, cmap="Greys", colorbar=False, values_format="d")
    ax.set_title("Figura 13.\nMatriz de confusión del modelo refinado final", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    note = f"TN={tn}, FP={fp}, FN={fn}, TP={tp}. Fuente: predictions_v3_layermix_hardneg_hnweight2x.csv."
    fig.text(0.02, 0.01, f"Nota. {note}", fontsize=9.5)
    savefig(REPORTS_DIR / "figura_13_matriz_confusion_modelo_final.png")


def render_roc_curve(predictions: pd.DataFrame, metrics: dict) -> None:
    y_true = predictions["label"].astype(int).to_numpy()
    y_score = predictions["score_patients"].astype(float).to_numpy()
    auc_val = float(metrics["sample_level"]["metrics"]["auc"])
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=200)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val, name="Modelo refinado final").plot(
        ax=ax,
        curve_kwargs={"color": "black", "lw": 2.2},
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7a7a7a", linewidth=1.2)
    ax.set_title("Figura 14.\nCurva ROC del modelo refinado final", fontsize=15, fontweight="bold", pad=16)
    ax.grid(alpha=0.18, color="#555555")
    ax.text(
        0.62,
        0.12,
        f"AUC = {auc_val:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#777777", "boxstyle": "round,pad=0.3"},
    )
    fig.text(0.02, 0.01, "Nota. La curva ROC se calculó con los scores del modelo refinado final.", fontsize=9.5)
    savefig(REPORTS_DIR / "figura_14_curva_roc_modelo_final.png")


def metric_value(metrics: dict, level: str, key: str) -> float:
    return float(metrics[level]["metrics"][key])


def render_metric_comparison(base_metrics: dict, final_metrics: dict, level: str, rows: list[tuple[str, str]], title: str, out_name: str) -> None:
    labels = [label for label, _ in rows]
    base_vals = np.array([metric_value(base_metrics, level, key) for _, key in rows])
    final_vals = np.array([metric_value(final_metrics, level, key) for _, key in rows])
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=200)
    ax.bar(x - width / 2, base_vals, width, label="Modelo base", color="#bdbdbd", edgecolor="black", linewidth=0.8)
    ax.bar(x + width / 2, final_vals, width, label="Modelo refinado final", color="#4d4d4d", edgecolor="black", linewidth=0.8)
    ax.set_ylim(0.5, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Valor")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.grid(axis="y", alpha=0.18, color="#555555")
    ax.legend(frameon=False)

    for xpos, value in zip(x - width / 2, base_vals):
        ax.text(xpos, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2, final_vals):
        ax.text(xpos, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.text(0.02, 0.01, "Nota. Comparación construida a partir de los archivos de métricas del modelo base y del modelo refinado final.", fontsize=9.5)
    savefig(REPORTS_DIR / out_name)


def main() -> None:
    base_metrics = load_json(BASE_METRICS_PATH)
    final_metrics = load_json(FINAL_METRICS_PATH)
    predictions = pd.read_csv(FINAL_PREDICTIONS_PATH)

    render_confusion_matrix(predictions)
    render_roc_curve(predictions, final_metrics)
    render_metric_comparison(
        base_metrics,
        final_metrics,
        "sample_level",
        [
            ("Accuracy", "accuracy"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-Score", "f1"),
            ("AUC-ROC", "auc"),
        ],
        "Figura 15.\nComparación de métricas a nivel de muestra",
        "figura_15_metricas_sample_level_modelo_final.png",
    )
    render_metric_comparison(
        base_metrics,
        final_metrics,
        "subject_level",
        [
            ("Subject Accuracy", "accuracy"),
            ("Subject Balanced Accuracy", "balanced_accuracy"),
            ("Subject Precision", "precision"),
            ("Subject Recall", "recall"),
            ("Subject F1-Score", "f1"),
            ("Subject AUC", "auc"),
        ],
        "Figura 16.\nComparación de métricas a nivel de sujeto",
        "figura_16_metricas_subject_level_modelo_final.png",
    )

    for file_name in [
        "figura_13_matriz_confusion_modelo_final.png",
        "figura_14_curva_roc_modelo_final.png",
        "figura_15_metricas_sample_level_modelo_final.png",
        "figura_16_metricas_subject_level_modelo_final.png",
    ]:
        print(REPORTS_DIR / file_name)


if __name__ == "__main__":
    main()
