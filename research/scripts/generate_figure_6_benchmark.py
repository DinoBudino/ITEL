from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
MULTI_DIR = RESEARCH_DIR / "research_results" / "CSV" / "multilingual_transfer"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"

BENCHMARK_JSON = MULTI_DIR / "multilingual_cz_en_heads_natural.json"

ROWS = [
    ("raw", "logreg"),
    ("raw", "linear_svm"),
    ("raw", "logreg_pca"),
    ("raw", "mlp_small"),
    ("raw", "mlp_medium"),
    ("centroid_shift", "logreg"),
    ("centroid_shift", "linear_svm"),
    ("centroid_shift", "logreg_pca"),
    ("centroid_shift", "mlp_small"),
    ("centroid_shift", "mlp_medium"),
]

BIDIRECTIONAL_JSON_BY_CONFIG = {
    ("raw", "logreg"): MULTI_DIR / "multilingual_bidirectional_logreg_raw_48.json",
    ("raw", "linear_svm"): MULTI_DIR / "multilingual_bidirectional_linear_svm_raw_48.json",
    ("raw", "logreg_pca"): MULTI_DIR / "multilingual_bidirectional_logreg_pca_raw.json",
    ("raw", "mlp_small"): MULTI_DIR / "multilingual_bidirectional_mlp_small_raw_48.json",
    ("raw", "mlp_medium"): MULTI_DIR / "multilingual_bidirectional_mlp_medium_raw_48.json",
    ("centroid_shift", "logreg"): MULTI_DIR / "multilingual_bidirectional_logreg_centroid_48.json",
    ("centroid_shift", "linear_svm"): MULTI_DIR / "multilingual_bidirectional_linear_svm_centroid_48.json",
    ("centroid_shift", "logreg_pca"): MULTI_DIR / "multilingual_bidirectional_logreg_pca_centroid.json",
    ("centroid_shift", "mlp_small"): MULTI_DIR / "multilingual_bidirectional_mlp_small_centroid_48.json",
    ("centroid_shift", "mlp_medium"): MULTI_DIR / "multilingual_bidirectional_mlp_medium_centroid_48.json",
}

FINAL_CONFIG = ("centroid_shift", "logreg_pca")

OUT_SVG = REPORTS_DIR / "figura_6_benchmark_corregida.svg"
OUT_PNG = REPORTS_DIR / "figura_6_benchmark_corregida.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources() -> tuple[dict, dict[tuple[str, str], dict]]:
    if not BENCHMARK_JSON.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {BENCHMARK_JSON}")

    benchmark = load_json(BENCHMARK_JSON)
    config = benchmark.get("config", {})
    if config.get("alignment_methods") != ["raw", "centroid_shift"]:
        raise ValueError("Benchmark alignment methods do not match the current official benchmark.")
    if config.get("model_types") != ["logreg", "linear_svm", "logreg_pca", "mlp_small", "mlp_medium"]:
        raise ValueError("Benchmark model types do not match the current official benchmark.")

    bidirectional: dict[tuple[str, str], dict] = {}
    for key, path in BIDIRECTIONAL_JSON_BY_CONFIG.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing bidirectional JSON for {key}: {path}")
        payload = load_json(path)
        datasets = payload.get("datasets", {})
        if datasets.get("en_subjects") != 48:
            raise ValueError(f"Bidirectional JSON for {key} is not from the current 48-subject English benchmark: {path}")
        bidirectional[key] = payload

    final_payload = bidirectional[FINAL_CONFIG]
    en = final_payload["target_english"]
    cz = final_payload["target_czech"]
    expected = (0.8542, 0.9323, 0.9175, 0.9710)
    observed = (
        round(float(en["balanced_accuracy"]), 4),
        round(float(en["auc"]), 4),
        round(float(cz["balanced_accuracy"]), 4),
        round(float(cz["auc"]), 4),
    )
    if observed != expected:
        raise ValueError(f"Final config metrics do not match expected values. Observed={observed} Expected={expected}")

    return benchmark, bidirectional


def build_table(benchmark: dict, bidirectional: dict[tuple[str, str], dict]) -> tuple[list[str], np.ndarray]:
    results = benchmark["results"]
    row_labels: list[str] = []
    values: list[list[float]] = []

    for alignment_method, model_type in ROWS:
        en_metrics = results[model_type][alignment_method]
        bi_metrics = bidirectional[(alignment_method, model_type)]
        cz_metrics = bi_metrics["target_czech"]

        row_labels.append(f"{alignment_method} · {model_type}")
        values.append(
            [
                float(en_metrics["balanced_accuracy"]),
                float(en_metrics["auc"]),
                float(cz_metrics["balanced_accuracy"]),
                float(cz_metrics["auc"]),
            ]
        )

    return row_labels, np.array(values, dtype=float)


def draw_figure(row_labels: list[str], matrix: np.ndarray) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    col_labels = [
        "Inglés\nBalanced Accuracy",
        "Inglés\nAUC-ROC",
        "Checo\nBalanced Accuracy",
        "Checo\nAUC-ROC",
    ]

    n_rows, n_cols = matrix.shape
    fig_w = 12.4
    fig_h = 8.1
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=400, facecolor="white")
    ax = fig.add_axes([0.06, 0.08, 0.90, 0.86])
    ax.set_facecolor("white")
    ax.axis("off")

    left_label_w = 3.95
    cell_w = 1.78
    cell_h = 0.80
    x0 = 0.0
    y0 = 0.0

    vmin = float(matrix.min())
    vmax = float(matrix.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Greys")

    # headers
    for j, col in enumerate(col_labels):
        x = left_label_w + j * cell_w
        header = Rectangle((x, y0 + n_rows * cell_h), cell_w, cell_h, facecolor="white", edgecolor="black", linewidth=1.2)
        ax.add_patch(header)
        ax.text(x + cell_w / 2, y0 + n_rows * cell_h + cell_h / 2, col, ha="center", va="center", fontsize=13.5, family="DejaVu Sans")

    label_header = Rectangle((x0, y0 + n_rows * cell_h), left_label_w, cell_h, facecolor="white", edgecolor="black", linewidth=1.2)
    ax.add_patch(label_header)

    final_idx = ROWS.index(FINAL_CONFIG)

    for i, row_label in enumerate(row_labels):
        y = y0 + (n_rows - 1 - i) * cell_h

        label_rect = Rectangle((x0, y), left_label_w, cell_h, facecolor="white", edgecolor="black", linewidth=1.0)
        ax.add_patch(label_rect)
        ax.text(x0 + 0.12, y + cell_h / 2, row_label, ha="left", va="center", fontsize=12.4, family="DejaVu Sans")

        for j in range(n_cols):
            x = left_label_w + j * cell_w
            face = cmap(norm(matrix[i, j]))
            rect = Rectangle((x, y), cell_w, cell_h, facecolor=face, edgecolor="black", linewidth=1.0)
            ax.add_patch(rect)
            text_color = "white" if norm(matrix[i, j]) > 0.63 else "black"
            display_value = f"{matrix[i, j]:.4f}".replace(".", ",")
            ax.text(x + cell_w / 2, y + cell_h / 2, display_value, ha="center", va="center", fontsize=11.9, family="DejaVu Sans", color=text_color)

        if i == final_idx:
            highlight = Rectangle((x0, y), left_label_w + n_cols * cell_w, cell_h, facecolor="none", edgecolor="black", linewidth=1.8)
            ax.add_patch(highlight)

    ax.set_xlim(-0.05, left_label_w + n_cols * cell_w + 0.05)
    ax.set_ylim(-0.05, y0 + (n_rows + 1) * cell_h + 0.05)

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    benchmark, bidirectional = verify_sources()
    row_labels, matrix = build_table(benchmark, bidirectional)
    draw_figure(row_labels, matrix)
    print(f"SVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
