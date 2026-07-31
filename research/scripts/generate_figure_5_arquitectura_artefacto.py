from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
RESULTS_DIR = RESEARCH_DIR / "research_results" / "CSV" / "multilingual_transfer"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"

FINAL_JSON = RESULTS_DIR / "multilingual_bidirectional_logreg_pca_centroid.json"
BENCHMARK_JSON = RESULTS_DIR / "multilingual_cz_en_heads_natural.json"
CZ_MANIFEST = RESEARCH_DIR / "processed_data" / "manifests" / "hubert_sequences_natural_multilingual_light.csv"
EN_MANIFEST = RESEARCH_DIR / "processed_data" / "manifests" / "hubert_sequences_en_test_v2.csv"

OUT_SVG = REPORTS_DIR / "figure_5_arquitectura_artefacto.svg"
OUT_PNG = REPORTS_DIR / "figure_5_arquitectura_artefacto.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources() -> None:
    required_paths = [FINAL_JSON, BENCHMARK_JSON, CZ_MANIFEST, EN_MANIFEST]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source artifacts: {missing}")

    final_report = load_json(FINAL_JSON)
    benchmark_report = load_json(BENCHMARK_JSON)

    final_config = final_report.get("config", {})
    if final_config.get("model_type") != "logreg_pca":
        raise ValueError("Final report is not logreg_pca.")
    if final_config.get("alignment_method") != "centroid_shift":
        raise ValueError("Final report is not centroid_shift.")
    if final_config.get("layer_policy") != "last":
        raise ValueError("Final report does not use last layer.")

    leaderboard = benchmark_report.get("leaderboard", [])
    if not leaderboard:
        raise ValueError("Benchmark leaderboard is empty.")
    top = leaderboard[0]
    if top.get("model_type") != "logreg_pca" or top.get("alignment_method") != "centroid_shift":
        raise ValueError("Benchmark top configuration is not logreg_pca + centroid_shift.")


def add_box(ax, x: float, y: float, w: float, h: float, lines: list[str], fontsize: int = 18) -> None:
    rect = Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=1.8)
    ax.add_patch(rect)
    total_lines = len(lines)
    line_spacing = h / (total_lines + 1)
    for idx, line in enumerate(lines, start=1):
        ax.text(
            x + w / 2,
            y + h - idx * line_spacing,
            line,
            ha="center",
            va="center",
            fontsize=fontsize,
            family="DejaVu Sans",
            color="black",
        )


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8, shrinkA=0, shrinkB=0),
    )


def draw_figure() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=400, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor("white")

    # Top parallel inputs
    add_box(ax, 8, 91, 30, 6.5, ["LANNA — Checo"], fontsize=18)
    add_box(ax, 62, 91, 30, 6.5, ["ENNI — Inglés"], fontsize=18)

    add_arrow(ax, 23, 91, 23, 86.5)
    add_arrow(ax, 77, 91, 77, 86.5)
    add_arrow(ax, 23, 86.5, 46, 86.5)
    add_arrow(ax, 77, 86.5, 54, 86.5)

    add_box(ax, 25, 80, 50, 7, ["Preprocesamiento del audio"], fontsize=18)
    add_arrow(ax, 50, 80, 50, 74.5)

    add_box(ax, 31, 68, 38, 9, ["HuBERT Base", "Última capa"], fontsize=18)
    add_arrow(ax, 50, 68, 50, 62.5)

    add_box(ax, 27, 56, 46, 9, ["Estadísticas globales", "y temporales"], fontsize=18)
    add_arrow(ax, 50, 56, 50, 50.5)

    add_box(ax, 24, 44, 52, 7, ["Agregación a nivel de sujeto"], fontsize=18)
    add_arrow(ax, 50, 44, 50, 38.5)

    add_box(ax, 24, 31, 52, 10, ["Adaptación entre dominios", "centroid_shift"], fontsize=18)
    add_arrow(ax, 50, 31, 50, 26.5)

    add_box(ax, 33, 21, 34, 6.5, ["StandardScaler"], fontsize=18)
    add_arrow(ax, 50, 21, 50, 16.5)

    add_box(ax, 40, 12, 20, 6.5, ["PCA"], fontsize=18)
    add_arrow(ax, 50, 12, 50, 8.0)

    add_box(ax, 23, 1.2, 54, 9.2, ["Regresión logística", "Ponderación balanceada"], fontsize=18)

    # Lower chain slightly shifted to fit page
    add_arrow(ax, 50, 1.2, 50, -2.0)

    # Since the page ends here, add bottom boxes in figure coordinates above the last one
    add_box(ax, 17, -11.0, 66, 7.5, ["Predicción a nivel de participante"], fontsize=18)
    add_arrow(ax, 50, -11.0, 50, -15.5)
    add_box(ax, 10, -24.0, 80, 9.5, ["Evaluación cross-lingual bidireccional", "Checo ↔ Inglés"], fontsize=18)

    # Extend visible limits to include lower boxes cleanly
    ax.set_ylim(-28, 100)

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    verify_sources()
    draw_figure()
    print(f"SVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
