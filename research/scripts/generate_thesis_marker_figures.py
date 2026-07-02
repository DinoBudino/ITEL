from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
CSV_DIR = RESEARCH_DIR / "research_results" / "CSV"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"


TASK_LABELS = {
    "consonant": "Consonantes",
    "one_syllable_items": "Items monosilabicos",
    "word_2syll": "Palabras de 2 silabas",
    "word_3syll": "Palabras de 3 silabas",
    "word_4syll": "Palabras de 4 silabas",
    "word_difficult": "Palabras dificiles",
}

PROFILE_LABELS = {
    "natural_multilingual_light": "Natural + multilingual_light",
    "natural_anti_confound": "Natural + anti_confound",
    "fixed_2s_multilingual_light": "Fijo 2 s + multilingual_light",
}


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 360) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def _load_task_frame() -> pd.DataFrame:
    frame = pd.read_csv(CSV_DIR / "preprocessing_by_task_comparison.csv")
    frame = frame[
        (frame["profile"] == "fixed_2s_multilingual_light")
        & (frame["group_col"] == "syllable_group")
    ].copy()
    frame["task_label"] = frame["group"].map(TASK_LABELS).fillna(frame["group"])
    return frame


def _load_profile_frame() -> pd.DataFrame:
    frame = pd.read_csv(CSV_DIR / "preprocessing_comparison.csv")
    frame["profile_label"] = frame["profile"].map(PROFILE_LABELS).fillna(frame["profile"])
    return frame


def render_marker_summary() -> None:
    task_frame = _load_task_frame()
    best_task = task_frame.sort_values("balanced_accuracy", ascending=False).iloc[0]
    worst_task = task_frame.sort_values("balanced_accuracy", ascending=True).iloc[0]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.05, 0.96, "Figura 6.", fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(0.05, 0.925, "Marcadores acustico-temporales inferidos por el modelo", fontsize=19, fontweight="bold", ha="left", va="top")
    fig.text(0.05, 0.892, "Resumen esquematico de los patrones que aportan mayor evidencia discriminativa en el corpus checo.", fontsize=11.5, ha="left", va="top", color="#444444")

    boxes = [
        (0.07, 0.60, "Marcador 1", "Ventanas\nbreves", "La evidencia no se distribuye\nen todo el audio por igual."),
        (0.40, 0.60, "Marcador 2", "Mayor carga\narticulatoria", "Las palabras con mas\nestructura secuencial aportan\nmas informacion."),
        (0.73, 0.60, "Marcador 3", "Menor valor de\nunidades aisladas", "Las consonantes y tareas\npoco estructuradas son las\nmas dificiles de separar."),
    ]

    for x, y, title, subtitle, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 0.20, 0.20,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#f4f4f4",
                edgecolor="#6e6e6e",
                linewidth=1.5,
            )
        )
        ax.text(x + 0.10, y + 0.16, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + 0.10, y + 0.11, subtitle, ha="center", va="center", fontsize=12)
        ax.text(x + 0.10, y + 0.05, body, ha="center", va="center", fontsize=10)

    ax.add_patch(FancyArrowPatch((0.27, 0.70), (0.40, 0.70), arrowstyle="->", mutation_scale=16, linewidth=1.8, color="#5a5a5a"))
    ax.add_patch(FancyArrowPatch((0.60, 0.70), (0.73, 0.70), arrowstyle="->", mutation_scale=16, linewidth=1.8, color="#5a5a5a"))

    ax.text(0.07, 0.40, "Evidencia cuantitativa", fontsize=14, fontweight="bold", ha="left")
    evidence = [
        f"• Mejor grupo: {best_task['task_label']} (Balanced Accuracy = {best_task['balanced_accuracy']:.3f})",
        f"• Grupo mas dificil: {worst_task['task_label']} (Balanced Accuracy = {worst_task['balanced_accuracy']:.3f})",
        "• El modelo mantiene sensibilidad alta, pero los falsos positivos se concentran en sujetos sanos atipicos.",
    ]
    for idx, line in enumerate(evidence):
        ax.text(0.08, 0.34 - idx * 0.07, line, fontsize=11.5, ha="left")

    fig.text(0.05, 0.03, "Nota. La figura resume patrones inferidos a partir del comportamiento del modelo y de las comparaciones por tipo de pronunciacion. Elaboracion propia.", fontsize=10, ha="left")
    savefig(REPORTS_DIR / "figura_6_marcadores_inferidos.png")


def render_task_bars() -> None:
    frame = _load_task_frame().sort_values("balanced_accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("white")

    labels = frame["task_label"].tolist()
    values = frame["balanced_accuracy"].to_numpy()
    y = np.arange(len(labels))

    ax.barh(y, values, color="#5a5a5a", edgecolor="black", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0.55, 0.83)
    ax.set_xlabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Figura 7.\nValor discriminativo por tipo de pronunciacion", fontsize=17, fontweight="bold", pad=18)
    ax.grid(axis="x", alpha=0.18)

    for yi, value in zip(y, values):
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=11)

    fig.text(0.10, 0.04, "Nota. Perfil evaluado: fijo 2 s + multilingual_light. Las tareas con mayor estructura fonetica muestran mejor separacion entre Healthy y Patients. Elaboracion propia.", fontsize=10, ha="left")
    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
    savefig(REPORTS_DIR / "figura_7_pronunciacion_marcadores.png")


def render_preprocessing_comparison() -> None:
    frame = _load_profile_frame().copy()
    frame = frame.sort_values("subject_balanced_accuracy", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=200)
    fig.patch.set_facecolor("white")

    x = np.arange(len(frame))
    labels = frame["profile_label"].tolist()
    subj = frame["subject_balanced_accuracy"].to_numpy()
    auc = frame["subject_auc"].to_numpy()

    axes[0].bar(x, subj, color="#4d4d4d", edgecolor="black", width=0.6)
    axes[0].set_ylim(0.75, 0.93)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right", fontsize=11)
    axes[0].set_ylabel("Balanced Accuracy", fontsize=12)
    axes[0].set_title("Rendimiento por sujeto", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.18)
    for xi, value in zip(x, subj):
        axes[0].text(xi, value + 0.004, f"{value:.3f}", ha="center", fontsize=10.5)

    axes[1].bar(x, auc, color="#b8b8b8", edgecolor="black", width=0.6)
    axes[1].set_ylim(0.90, 0.98)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right", fontsize=11)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_title("Capacidad de separacion", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.18)
    for xi, value in zip(x, auc):
        axes[1].text(xi, value + 0.0025, f"{value:.3f}", ha="center", fontsize=10.5)

    fig.suptitle("Figura 8.\nComparacion simple de perfiles de preprocesamiento", fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.07, 0.03, "Nota. En el corpus checo, los tres perfiles muestran sensibilidad completa a nivel sujeto. El mejor compromiso general se observa en la configuracion fija de 2 s + multilingual_light. Elaboracion propia.", fontsize=10, ha="left")
    plt.tight_layout(rect=(0, 0.08, 1, 0.90))
    savefig(REPORTS_DIR / "figura_8_preprocesamiento_comparacion.png")


def build_notes_markdown() -> str:
    lines = [
        "# Figuras simples de marcadores - Notas sugeridas",
        "",
        "## Figura 6",
        "Marcadores acustico-temporales inferidos por el modelo.",
        "",
        "Nota. La figura resume los patrones que aportan mayor evidencia discriminativa en el corpus checo: ventanas temporales breves, mayor estructura articulatoria y menor utilidad de unidades aisladas. Elaboracion propia.",
        "",
        "## Figura 7",
        "Valor discriminativo por tipo de pronunciacion.",
        "",
        "Nota. La figura muestra el rendimiento por grupo de pronunciacion para el perfil checo seleccionado. Las palabras de mayor estructura secuencial muestran mejor capacidad discriminativa que las consonantes aisladas. Elaboracion propia.",
        "",
        "## Figura 8",
        "Comparacion simple de perfiles de preprocesamiento.",
        "",
        "Nota. La figura compara los perfiles evaluados a nivel de sujeto mediante Balanced Accuracy y AUC. Elaboracion propia.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    render_marker_summary()
    render_task_bars()
    render_preprocessing_comparison()
    notes_path = REPORTS_DIR / "figuras_marcadores_notas.md"
    notes_path.write_text(build_notes_markdown(), encoding="utf-8")
    print(f"Generated {REPORTS_DIR / 'figura_6_marcadores_inferidos.png'}")
    print(f"Generated {REPORTS_DIR / 'figura_7_pronunciacion_marcadores.png'}")
    print(f"Generated {REPORTS_DIR / 'figura_8_preprocesamiento_comparacion.png'}")
    print(f"Generated {notes_path}")


if __name__ == "__main__":
    main()
