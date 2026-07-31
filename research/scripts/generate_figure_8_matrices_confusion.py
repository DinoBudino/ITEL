from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
MULTI_DIR = RESEARCH_DIR / "research_results" / "CSV" / "multilingual_transfer"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"

PRED_CSV = MULTI_DIR / "multilingual_bidirectional_logreg_pca_centroid_predictions.csv"
REPORT_JSON = MULTI_DIR / "multilingual_bidirectional_logreg_pca_centroid.json"

OUT_SVG = REPORTS_DIR / "figura_8_matrices_confusion.svg"
OUT_PNG = REPORTS_DIR / "figura_8_matrices_confusion.png"


LABEL_MAP = {"Healthy": 0, "Patients": 1}
CLASS_NAMES = ["Sano", "Paciente"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources() -> tuple[pd.DataFrame, dict]:
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {PRED_CSV}")
    if not REPORT_JSON.exists():
        raise FileNotFoundError(f"Missing report JSON: {REPORT_JSON}")

    pred = pd.read_csv(PRED_CSV)
    report = load_json(REPORT_JSON)

    required_cols = {"subject_id", "label", "pred_label", "target_domain"}
    missing = required_cols.difference(pred.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return pred, report


def compute_and_validate(pred: pd.DataFrame, report: dict) -> dict[str, dict[str, object]]:
    official = {
        "english": {
            "n_rows": int(report["target_english"]["n_rows"]),
            "accuracy": float(report["target_english"]["accuracy"]),
            "balanced_accuracy": float(report["target_english"]["balanced_accuracy"]),
        },
        "czech": {
            "n_rows": int(report["target_czech"]["n_rows"]),
            "accuracy": float(report["target_czech"]["accuracy"]),
            "balanced_accuracy": float(report["target_czech"]["balanced_accuracy"]),
        },
    }

    outputs: dict[str, dict[str, object]] = {}

    for domain_key, domain_label in [("english", "english"), ("czech", "czech")]:
        sub = pred.loc[pred["target_domain"] == domain_label].copy()
        y_true = sub["label"].astype(str).map(LABEL_MAP).to_numpy()
        y_pred = sub["pred_label"].astype(str).map(LABEL_MAP).to_numpy()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        n = int(len(sub))
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / n
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0

        off = official[domain_key]
        if n != off["n_rows"]:
            raise ValueError(f"Row count mismatch for {domain_key}: {n} vs {off['n_rows']}")
        if abs(acc - off["accuracy"]) > 1e-6:
            raise ValueError(f"Accuracy mismatch for {domain_key}: {acc} vs {off['accuracy']}")
        if abs(bal_acc - off["balanced_accuracy"]) > 1e-6:
            raise ValueError(f"BalAcc mismatch for {domain_key}: {bal_acc} vs {off['balanced_accuracy']}")

        outputs[domain_key] = {
            "cm": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_rows": n,
        }

    return outputs


def draw_figure(matrices: dict[str, dict[str, object]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), facecolor="white")
    domains = [("english", "Inglés objetivo"), ("czech", "Checo objetivo")]

    for ax, (key, panel_title) in zip(axes, domains):
        ax.set_facecolor("white")
        m = matrices[key]
        cm = m["cm"]
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]

        n = m["n_rows"]
        pct_matrix = cm / n * 100

        gray_max = max(cm.max(), 1)
        img = np.array([[tn, fp], [fn, tp]], dtype=float) / gray_max
        img = 1.0 - img * 0.7

        ax.matshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal")

        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = pct_matrix[i, j]
                text_color = "white" if img[i, j] < 0.45 else "black"
                ax.text(
                    j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=16, color=text_color, fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, fontsize=13)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASS_NAMES, fontsize=13)

        ax.set_xlabel("Predicción", fontsize=13, labelpad=10)
        ax.set_ylabel("Real", fontsize=13, labelpad=10)

        ax.tick_params(axis="both", which="both", length=0)

        ax.set_title(panel_title, fontsize=15, pad=12, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.tight_layout(pad=2.0)

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    pred, report = verify_sources()
    matrices = compute_and_validate(pred, report)
    draw_figure(matrices)
    print(f"SVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
