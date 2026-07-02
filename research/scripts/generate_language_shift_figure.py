from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from research.src.config import CONFIG


REPORT_JSON = CONFIG.paths.reports_dir / "CSV" / "language_shift" / "language_shift_transfer_report.json"
OUT_DIR = CONFIG.paths.reports_dir / "attention_viz" / "language_shift"


def _row(report: dict, section: str, classifier: str, alignment: str, metric: str) -> float:
    return float(report[section][classifier][alignment][metric])


def main() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classifier = "logreg"
    aligns = report["alignment_methods"]
    labels = [a.replace("_", " ") for a in aligns]

    transfer_bal = np.array([_row(report, "czech_to_english_transfer", classifier, a, "balanced_accuracy") for a in aligns])
    transfer_auc = np.array([_row(report, "czech_to_english_transfer", classifier, a, "auc") for a in aligns])
    loso_bal = np.array([_row(report, "english_only_loso", classifier, a, "balanced_accuracy") for a in aligns])
    loso_auc = np.array([_row(report, "english_only_loso", classifier, a, "auc") for a in aligns])

    x = np.arange(len(aligns))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=220)
    fig.patch.set_facecolor("white")

    axes[0].bar(x - width / 2, transfer_bal, width, color="#4d4d4d", edgecolor="black", label="BalAcc")
    axes[0].bar(x + width / 2, transfer_auc, width, color="#b3b3b3", edgecolor="black", label="AUC")
    axes[0].set_title("Transferencia CZ -> EN", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10, ha="right")
    axes[0].set_ylim(0.45, 0.85)
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    for xi, b, a in zip(x, transfer_bal, transfer_auc):
        axes[0].text(xi - width / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=10)
        axes[0].text(xi + width / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=10)

    axes[1].bar(x - width / 2, loso_bal, width, color="#4d4d4d", edgecolor="black", label="BalAcc")
    axes[1].bar(x + width / 2, loso_auc, width, color="#b3b3b3", edgecolor="black", label="AUC")
    axes[1].set_title("Entrenamiento solo en ingles (LOSO)", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10, ha="right")
    axes[1].set_ylim(0.45, 0.85)
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False)
    for xi, b, a in zip(x, loso_bal, loso_auc):
        axes[1].text(xi - width / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=10)
        axes[1].text(xi + width / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=10)

    fig.suptitle("Figura 12. Comparacion de alineaciones para transferencia multilingue", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.04, 0.03, "Nota. La figura compara el desempeno del clasificador logistico con tres esquemas de alineacion: sin ajuste, desplazamiento por centroides y alineacion de varianzas (coral_shift). Elaboracion propia.", fontsize=10, ha="left")
    plt.tight_layout(rect=(0, 0.07, 1, 0.92))
    out = OUT_DIR / "figura_12_language_shift_comparison.png"
    plt.savefig(out, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated {out}")


if __name__ == "__main__":
    main()
