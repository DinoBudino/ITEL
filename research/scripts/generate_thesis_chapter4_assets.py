from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
RESULTS_DIR = RESEARCH_DIR / "research_results" / "CSV"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"
METADATA_PATH = RESEARCH_DIR / "checkpoints" / "v3_final" / "layermix" / "metadata.json"


@dataclass(slots=True)
class MetricsBundle:
    control: dict
    layermix: dict
    predictions_control: pd.DataFrame
    predictions_layermix: pd.DataFrame
    metadata: dict


def load_bundle() -> MetricsBundle:
    control = json.loads((RESULTS_DIR / "metrics_v3.json").read_text(encoding="utf-8"))
    layermix = json.loads((RESULTS_DIR / "metrics_v3_layermix.json").read_text(encoding="utf-8"))
    predictions_control = pd.read_csv(RESULTS_DIR / "predictions_v3.csv")
    predictions_layermix = pd.read_csv(RESULTS_DIR / "predictions_v3_layermix.csv")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return MetricsBundle(control, layermix, predictions_control, predictions_layermix, metadata)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 300) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def fmt4(value: float, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render_architecture_figure(bundle: MetricsBundle) -> None:
    model_cfg = bundle.metadata["model_config"]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.text(0.06, 0.96, "Figura 1.", fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925, "Arquitectura del modelo V3 LayerMix", fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.89, "V3_HuBERT_BASE -> AttentionBiLSTM + SupCon (LayerMix)", fontsize=11, ha="left", va="top", color="#444444")

    headers = [
        (0.10, "ENTRADA"),
        (0.28, "CODIFICADOR + LAYERMIX"),
        (0.50, "MODELO SECUENCIAL"),
        (0.71, "CLASIFICADOR"),
        (0.89, "SALIDA"),
    ]
    for x, label in headers:
        ax.text(x, 0.83, label, ha="center", va="center", fontsize=12, fontweight="bold", color="black")

    layer_nodes = [
        (0.10, [("Audio", 0.65), ("Infantil", 0.54), ("Frames", 0.43)]),
        (0.28, [("HuBERT", 0.74), ("C1-C4", 0.64), ("C5-C8", 0.54), ("C9-C12", 0.44), ("LayerMix", 0.34)]),
        (0.50, [("T x 768", 0.68), ("Bi", 0.56), ("LSTM", 0.44), ("Attention", 0.32)]),
        (0.71, [("Contexto", 0.60), ("FC128", 0.48), ("Logit", 0.36)]),
        (0.89, [("Healthy", 0.58), ("Patient", 0.40)]),
    ]
    shades = [0.05, 0.20, 0.35, 0.55, 0.05]
    radius = 0.03
    centers: list[list[tuple[float, float]]] = []

    for x, nodes in layer_nodes:
        current = []
        for idx, (label, y) in enumerate(nodes):
            edge = str(shades[min(idx, len(shades) - 1)])
            ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=edge, linewidth=2.2))
            ax.add_patch(Circle((x, y), radius * 0.58, facecolor=str(min(shades[min(idx, len(shades) - 1)] + 0.08, 0.75)), edgecolor="none", alpha=0.18))
            ax.text(x, y - 0.055, label, ha="center", va="top", fontsize=10, color="black")
            current.append((x, y))
        centers.append(current)

    gray_edges = ["#d0d0d0", "#ababab", "#7a7a7a", "#4e4e4e"]
    for layer_idx in range(len(centers) - 1):
        for src_idx, (x1, y1) in enumerate(centers[layer_idx]):
            for dst_idx, (x2, y2) in enumerate(centers[layer_idx + 1]):
                rad = 0.15 * np.sign(y2 - y1)
                color = gray_edges[(src_idx + dst_idx + layer_idx) % len(gray_edges)]
                lw = 0.9 + ((src_idx * 3 + dst_idx) % 3) * 0.55
                alpha = 0.28 + ((src_idx + dst_idx) % 3) * 0.17
                patch = FancyArrowPatch(
                    (x1 + radius * 0.95, y1),
                    (x2 - radius * 0.95, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                )
                ax.add_patch(patch)

    ax.text(0.06, 0.18, "Señales del modelo", fontsize=13, fontweight="bold", ha="left")
    signal_lines = [
        "El audio infantil se transforma en una secuencia de embeddings por frame.",
        "HuBERT-base extrae 12 capas y LayerMix aprende una fusion ponderada.",
        f"La Bi-LSTM bidireccional usa {model_cfg['lstm_layers']} capas y hidden size {model_cfg['lstm_hidden']} por direccion.",
        "Single-Head Attention resume la secuencia en un vector de contexto.",
        f"El clasificador binario aplica Linear(512->128)->ReLU->Linear(128->1) y sigmoid.",
    ]
    for idx, line in enumerate(signal_lines):
        ax.text(0.06, 0.145 - idx * 0.038, line, fontsize=10.5, ha="left", color="#222222")

    panel = FancyBboxPatch((0.67, 0.05), 0.28, 0.18, boxstyle="round,pad=0.018,rounding_size=0.02", linewidth=1.0, edgecolor="#9a9a9a", facecolor="white")
    ax.add_patch(panel)
    ax.text(0.69, 0.205, "Hiperparámetros", fontsize=13, fontweight="bold", ha="left")
    hyper = [
        f"dropout={model_cfg['dropout']}",
        f"fc_hidden={model_cfg['fc_hidden']}",
        f"epochs={bundle.metadata['training_config']['epochs']}",
        f"lr={bundle.metadata['training_config']['lr']}",
        f"weight_decay={bundle.metadata['training_config']['weight_decay']}",
        f"batch_size=16",
    ]
    for idx, value in enumerate(hyper):
        col = idx // 3
        row = idx % 3
        ax.text(0.69 + col * 0.13, 0.17 - row * 0.045, value, fontsize=10, ha="left", color="#222222")

    note = (
        "Nota. La figura muestra el diseño general del modelo propuesto, compuesto por la entrada de audio, "
        "extraccion de representaciones mediante HuBERT, fusion de capas con LayerMix, procesamiento secuencial "
        "mediante Bi-LSTM, mecanismo de atencion y clasificador binario para la salida Healthy/Patient. "
        "Elaboracion propia."
    )
    fig.text(0.06, 0.01, note, fontsize=10, ha="left", va="bottom")

    savefig(REPORTS_DIR / "figura_1_arquitectura_v3_layermix.png", dpi=400)


def render_confusion_figure(bundle: MetricsBundle) -> dict[str, int]:
    y_true = bundle.predictions_layermix["label"].astype(int).to_numpy()
    y_pred = (bundle.predictions_layermix["pred_label"] == "Patients").astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=200)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Patient"])
    disp.plot(ax=ax, cmap="Greys", colorbar=False, values_format="d")
    ax.set_title("Figura 2.\nMatriz de confusion del modelo V3 LayerMix", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor real")
    fig.text(0.02, 0.01, "Nota. La matriz de confusion se calculo a partir de las predicciones reales del archivo predictions_v3_layermix.csv. Elaboracion propia.", fontsize=10)
    savefig(REPORTS_DIR / "figura_2_matriz_confusion_v3_layermix.png", dpi=400)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def render_roc_figure(bundle: MetricsBundle) -> None:
    y_true = bundle.predictions_layermix["label"].astype(int).to_numpy()
    y_score = bundle.predictions_layermix["score_patients"].astype(float).to_numpy()
    auc_val = float(bundle.layermix["sample_level"]["metrics"]["auc"])
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=200)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val, name="V3 LayerMix").plot(
        ax=ax,
        curve_kwargs={"color": "black", "lw": 2.2},
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7a7a7a", linewidth=1.2)
    ax.set_title("Figura 3.\nCurva ROC del modelo V3 LayerMix", fontsize=15, fontweight="bold", pad=16)
    ax.grid(alpha=0.18, color="#555555")
    ax.text(0.62, 0.12, f"AUC = {auc_val:.4f}", transform=ax.transAxes, fontsize=12, bbox={"facecolor": "white", "edgecolor": "#777777", "boxstyle": "round,pad=0.3"})
    fig.text(0.02, 0.01, "Nota. La curva ROC se calculo con los scores reales de score_patients del modelo V3 LayerMix. Elaboracion propia.", fontsize=10)
    savefig(REPORTS_DIR / "figura_3_curva_roc_v3_layermix.png", dpi=400)


def render_metric_comparison_figures(bundle: MetricsBundle) -> None:
    sample_metrics = [
        ("Accuracy", bundle.control["sample_level"]["metrics"]["accuracy"], bundle.layermix["sample_level"]["metrics"]["accuracy"]),
        ("Balanced Accuracy", bundle.control["sample_level"]["metrics"]["balanced_accuracy"], bundle.layermix["sample_level"]["metrics"]["balanced_accuracy"]),
        ("Precision", bundle.control["sample_level"]["metrics"]["precision"], bundle.layermix["sample_level"]["metrics"]["precision"]),
        ("Recall", bundle.control["sample_level"]["metrics"]["recall"], bundle.layermix["sample_level"]["metrics"]["recall"]),
        ("F1-Score", bundle.control["sample_level"]["metrics"]["f1"], bundle.layermix["sample_level"]["metrics"]["f1"]),
        ("AUC-ROC", bundle.control["sample_level"]["metrics"]["auc"], bundle.layermix["sample_level"]["metrics"]["auc"]),
    ]
    subject_metrics = [
        ("Subject Accuracy", bundle.control["subject_level"]["metrics"]["accuracy"], bundle.layermix["subject_level"]["metrics"]["accuracy"]),
        ("Subject Balanced Accuracy", bundle.control["subject_level"]["metrics"]["balanced_accuracy"], bundle.layermix["subject_level"]["metrics"]["balanced_accuracy"]),
        ("Subject Precision", bundle.control["subject_level"]["metrics"]["precision"], bundle.layermix["subject_level"]["metrics"]["precision"]),
        ("Subject Recall", bundle.control["subject_level"]["metrics"]["recall"], bundle.layermix["subject_level"]["metrics"]["recall"]),
        ("Subject F1-Score", bundle.control["subject_level"]["metrics"]["f1"], bundle.layermix["subject_level"]["metrics"]["f1"]),
        ("Subject AUC", bundle.control["subject_level"]["metrics"]["auc"], bundle.layermix["subject_level"]["metrics"]["auc"]),
    ]

    def plot_grouped(metrics: list[tuple[str, float, float]], title: str, out_name: str, note: str) -> None:
        labels = [row[0] for row in metrics]
        control_vals = np.array([row[1] for row in metrics])
        layermix_vals = np.array([row[2] for row in metrics])
        x = np.arange(len(labels))
        width = 0.36

        fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=200)
        ax.bar(x - width / 2, control_vals, width, label="V3 Control", color="#bdbdbd", edgecolor="black", linewidth=0.8)
        ax.bar(x + width / 2, layermix_vals, width, label="V3 LayerMix", color="#4d4d4d", edgecolor="black", linewidth=0.8)
        ax.set_ylim(0.5, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("Valor")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
        ax.grid(axis="y", alpha=0.18, color="#555555")
        ax.legend(frameon=False)

        for xpos, value in zip(x - width / 2, control_vals):
            ax.text(xpos, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        for xpos, value in zip(x + width / 2, layermix_vals):
            ax.text(xpos, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

        fig.text(0.02, 0.01, note, fontsize=10)
        savefig(REPORTS_DIR / out_name, dpi=400)

    plot_grouped(
        sample_metrics,
        "Figura 4.\nComparacion grafica de metricas sample-level",
        "figura_4_metricas_sample_level.png",
        "Nota. Comparacion entre V3 Control y V3 LayerMix a nivel de muestra, construida a partir de metrics_v3.json y metrics_v3_layermix.json. Elaboracion propia.",
    )
    plot_grouped(
        subject_metrics,
        "Figura 5.\nComparacion grafica de metricas subject-level",
        "figura_5_metricas_subject_level.png",
        "Nota. Comparacion entre V3 Control y V3 LayerMix a nivel de sujeto, construida a partir de metrics_v3.json y metrics_v3_layermix.json. Elaboracion propia.",
    )


def build_tables_markdown(bundle: MetricsBundle, cm_values: dict[str, int]) -> str:
    control = bundle.control
    layermix = bundle.layermix
    dataset = layermix["dataset"]
    model_cfg = bundle.metadata["model_config"]
    train_cfg = bundle.metadata["training_config"]
    fold_control = control["sample_level"]["fold_summary"]
    fold_layermix = layermix["sample_level"]["fold_summary"]

    def table_block(number: int, name: str, header: list[str], rows: list[list[str]]) -> str:
        lines = [f"Tabla {number}.", name, ""]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("Nota. Elaboracion propia.")
        lines.append("")
        return "\n".join(lines)

    t2_rows = [
        ["Modelo extractor", "HuBERT-base"],
        ["Estado del extractor", "Congelado"],
        ["Capas extraidas", "12"],
        ["Dimension por capa", "768"],
        ["Formato de representacion", "[12, T, 768]"],
        ["Muestras totales", str(dataset["n_samples"])],
        ["Muestras Patient", str(dataset["label_distribution"]["Patients"])],
        ["Muestras Healthy", str(dataset["label_distribution"]["Healthy"])],
        ["Sujetos", str(dataset["n_subjects"])],
        ["Estrategia temporal", "Natural-length"],
    ]

    t3_rows = [
        ["Extractor acustico", "HuBERT-base congelado"],
        ["Capas utilizadas", "12 capas de HuBERT"],
        ["Fusion de capas", "LayerMix / ScalarMix"],
        ["Normalizacion", "LayerNorm"],
        ["Modelo secuencial", "Bi-LSTM bidireccional"],
        ["Capas Bi-LSTM", str(model_cfg["lstm_layers"])],
        ["Tamano oculto", str(model_cfg["lstm_hidden"])],
        ["Salida Bi-LSTM", str(model_cfg["lstm_hidden"] * 2) + " dimensiones"],
        ["Mecanismo de atencion", "Single-Head Attention"],
        ["Clasificador", "Linear(512->128) -> ReLU -> Linear(128->1)"],
        ["Funcion de perdida", "BCE + Supervised Contrastive Loss"],
        ["Optimizador", "Adam"],
        ["Learning rate", str(train_cfg["lr"])],
        ["Weight decay", "1e-4"],
        ["Batch size", "16"],
        ["Epocas de entrenamiento", str(train_cfg["epochs"])],
    ]

    t4_rows = [
        ["V3 Control", "Utiliza unicamente la ultima capa de HuBERT como entrada para la Bi-LSTM."],
        ["V3 LayerMix", "Utiliza una combinacion aprendible de las 12 capas de HuBERT."],
    ]

    sample_pairs = [
        ("Accuracy", control["sample_level"]["metrics"]["accuracy"], layermix["sample_level"]["metrics"]["accuracy"]),
        ("Balanced Accuracy", control["sample_level"]["metrics"]["balanced_accuracy"], layermix["sample_level"]["metrics"]["balanced_accuracy"]),
        ("Precision", control["sample_level"]["metrics"]["precision"], layermix["sample_level"]["metrics"]["precision"]),
        ("Recall", control["sample_level"]["metrics"]["recall"], layermix["sample_level"]["metrics"]["recall"]),
        ("F1-Score", control["sample_level"]["metrics"]["f1"], layermix["sample_level"]["metrics"]["f1"]),
        ("AUC-ROC", control["sample_level"]["metrics"]["auc"], layermix["sample_level"]["metrics"]["auc"]),
    ]
    t5_rows = [[label, fmt4(ctrl), fmt4(lmix), fmt4(lmix - ctrl, signed=True)] for label, ctrl, lmix in sample_pairs]

    subject_pairs = [
        ("Subject Accuracy", control["subject_level"]["metrics"]["accuracy"], layermix["subject_level"]["metrics"]["accuracy"]),
        ("Subject Balanced Accuracy", control["subject_level"]["metrics"]["balanced_accuracy"], layermix["subject_level"]["metrics"]["balanced_accuracy"]),
        ("Subject Precision", control["subject_level"]["metrics"]["precision"], layermix["subject_level"]["metrics"]["precision"]),
        ("Subject Recall", control["subject_level"]["metrics"]["recall"], layermix["subject_level"]["metrics"]["recall"]),
        ("Subject F1-Score", control["subject_level"]["metrics"]["f1"], layermix["subject_level"]["metrics"]["f1"]),
        ("Subject AUC", control["subject_level"]["metrics"]["auc"], layermix["subject_level"]["metrics"]["auc"]),
    ]
    t6_rows = [[label, fmt4(ctrl), fmt4(lmix), fmt4(lmix - ctrl, signed=True)] for label, ctrl, lmix in subject_pairs]

    t7_rows = [
        ["Mean Fold Balanced Accuracy", f"{fold_control['balanced_accuracy']['mean']:.4f} ± {fold_control['balanced_accuracy']['std']:.4f}", f"{fold_layermix['balanced_accuracy']['mean']:.4f} ± {fold_layermix['balanced_accuracy']['std']:.4f}"],
        ["Mean Fold AUC", f"{fold_control['auc']['mean']:.4f} ± {fold_control['auc']['std']:.4f}", f"{fold_layermix['auc']['mean']:.4f} ± {fold_layermix['auc']['std']:.4f}"],
        ["Mean Fold F1", f"{fold_control['f1']['mean']:.4f} ± {fold_control['f1']['std']:.4f}", f"{fold_layermix['f1']['mean']:.4f} ± {fold_layermix['f1']['std']:.4f}"],
        ["Mean Fold Recall", f"{fold_control['recall']['mean']:.4f} ± {fold_control['recall']['std']:.4f}", f"{fold_layermix['recall']['mean']:.4f} ± {fold_layermix['recall']['std']:.4f}"],
    ]

    t8_rows = [
        ["Hasib et al.", "DCNN", "90.47% Accuracy"],
        ["Kaushik et al.", "2D-CNN", "99.09% Accuracy"],
        ["Kotarba & Kotarba", "ResNet-18", "99.48% Accuracy"],
        ["Reddy et al.", "FFNN + caracteristicas glotales", "98.82% Accuracy"],
        ["Sharma et al.", "LBP + caracteristicas acusticas", "97.36% Accuracy"],
        ["Modelo propuesto", "HuBERT + LayerMix + Bi-LSTM + Attention", "89.48% Sample Accuracy / 94.90% Subject Accuracy"],
    ]

    figure_lines = [
        "Figuras generadas",
        "",
        "Figura 1.",
        "Arquitectura del modelo V3 LayerMix",
        "",
        "Nota. La figura muestra el diseño general del modelo propuesto, compuesto por la entrada de audio, extraccion de representaciones mediante HuBERT, fusion de capas con LayerMix, procesamiento secuencial mediante Bi-LSTM, mecanismo de atencion y clasificador binario para la salida Healthy/Patient. Elaboracion propia.",
        "",
        "Figura 2.",
        "Matriz de confusion del modelo V3 LayerMix",
        "",
        f"Nota. La matriz de confusion se calculo a partir de las predicciones reales del modelo V3 LayerMix: TN={cm_values['tn']}, FP={cm_values['fp']}, FN={cm_values['fn']}, TP={cm_values['tp']}. Elaboracion propia.",
        "",
        "Figura 3.",
        "Curva ROC del modelo V3 LayerMix",
        "",
        f"Nota. La curva ROC se calculo con los scores reales del archivo predictions_v3_layermix.csv. AUC = {layermix['sample_level']['metrics']['auc']:.4f}. Elaboracion propia.",
        "",
        "Figura 4.",
        "Comparacion grafica de metricas sample-level",
        "",
        "Nota. Compara Accuracy, Balanced Accuracy, Precision, Recall, F1-Score y AUC-ROC entre V3 Control y V3 LayerMix. Elaboracion propia.",
        "",
        "Figura 5.",
        "Comparacion grafica de metricas subject-level",
        "",
        "Nota. Compara Subject Accuracy, Subject Balanced Accuracy, Subject Precision, Subject Recall, Subject F1-Score y Subject AUC entre V3 Control y V3 LayerMix. Elaboracion propia.",
        "",
    ]

    blocks = [
        "# Capitulo 4 - Tablas y figuras de resultados",
        "",
        "Este archivo reune las tablas y notas de figuras generadas para el Capitulo 4, utilizando como fuentes `metrics_v3.json`, `metrics_v3_layermix.json`, `predictions_v3.csv`, `predictions_v3_layermix.csv` y `metadata.json` del checkpoint final `v3_final/layermix`.",
        "",
        "Ubicacion sugerida de la Figura 1: 4.2.3 Objetivo especifico 3, despues del parrafo que presenta la arquitectura V3 LayerMix y antes de la tabla de arquitectura e hiperparametros.",
        "",
        *figure_lines,
        table_block(2, "Caracteristicas de las representaciones extraidas con HuBERT", ["Parametro", "Valor"], t2_rows),
        table_block(3, "Arquitectura e hiperparametros del modelo V3 LayerMix", ["Parametro", "Valor"], t3_rows),
        table_block(4, "Modelos evaluados", ["Modelo", "Descripcion"], t4_rows),
        table_block(5, "Rendimiento del modelo en validacion cruzada a nivel de muestra", ["Metrica", "V3 Control", "V3 LayerMix", "Diferencia"], t5_rows),
        table_block(6, "Rendimiento del modelo en validacion cruzada a nivel de sujeto", ["Metrica", "V3 Control", "V3 LayerMix", "Diferencia"], t6_rows),
        table_block(7, "Estabilidad del modelo entre folds", ["Metrica", "V3 Control", "V3 LayerMix"], t7_rows),
        table_block(8, "Comparacion del modelo propuesto con modelos del estado del arte", ["Autor / Estudio", "Enfoque", "Metrica principal"], t8_rows),
    ]
    return "\n".join(blocks)


def main() -> None:
    ensure_dirs()
    bundle = load_bundle()
    render_architecture_figure(bundle)
    cm_values = render_confusion_figure(bundle)
    render_roc_figure(bundle)
    render_metric_comparison_figures(bundle)
    markdown = build_tables_markdown(bundle, cm_values)
    out_md = REPORTS_DIR / "capitulo_4_tablas_y_figuras.md"
    out_md.write_text(markdown, encoding="utf-8")

    generated = [
        REPORTS_DIR / "figura_1_arquitectura_v3_layermix.png",
        REPORTS_DIR / "figura_2_matriz_confusion_v3_layermix.png",
        REPORTS_DIR / "figura_3_curva_roc_v3_layermix.png",
        REPORTS_DIR / "figura_4_metricas_sample_level.png",
        REPORTS_DIR / "figura_5_metricas_subject_level.png",
        out_md,
    ]
    for path in generated:
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
