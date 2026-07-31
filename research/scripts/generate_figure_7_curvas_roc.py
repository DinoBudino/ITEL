from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
MULTI_DIR = RESEARCH_DIR / "research_results" / "CSV" / "multilingual_transfer"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"

PRED_CSV = MULTI_DIR / "multilingual_bidirectional_logreg_pca_centroid_predictions.csv"
REPORT_JSON = MULTI_DIR / "multilingual_bidirectional_logreg_pca_centroid.json"

OUT_SVG = REPORTS_DIR / "figura_7_curvas_roc.svg"
OUT_PNG = REPORTS_DIR / "figura_7_curvas_roc.png"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sources() -> tuple[pd.DataFrame, dict]:
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {PRED_CSV}")
    if not REPORT_JSON.exists():
        raise FileNotFoundError(f"Missing report JSON: {REPORT_JSON}")

    pred = pd.read_csv(PRED_CSV)
    report = load_json(REPORT_JSON)

    required_cols = {
        "subject_id",
        "label",
        "score_patients",
        "pred_label",
        "threshold_fold",
        "model_type",
        "alignment_method",
        "target_domain",
    }
    missing = required_cols.difference(pred.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing)}")

    model_types = set(pred["model_type"].astype(str).unique().tolist())
    alignments = set(pred["alignment_method"].astype(str).unique().tolist())
    domains = set(pred["target_domain"].astype(str).unique().tolist())

    if model_types != {"logreg_pca"}:
        raise ValueError(f"Unexpected model_type values: {model_types}")
    if alignments != {"centroid_shift"}:
        raise ValueError(f"Unexpected alignment_method values: {alignments}")
    if domains != {"english", "czech"}:
        raise ValueError(f"Unexpected target_domain values: {domains}")

    return pred, report


def compute_curves(pred: pd.DataFrame, report: dict) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    official = {
        "english": float(report["target_english"]["auc"]),
        "czech": float(report["target_czech"]["auc"]),
    }

    for domain in ["english", "czech"]:
        sub = pred.loc[pred["target_domain"] == domain].copy()
        y_true = (sub["label"].astype(str) == "Patients").astype(int).to_numpy()
        y_score = sub["score_patients"].astype(float).to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = float(roc_auc_score(y_true, y_score))
        if abs(auc - official[domain]) > 1e-10:
            raise ValueError(
                f"AUC mismatch for {domain}: computed={auc} official={official[domain]}"
            )

        outputs[domain] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "n_rows": int(len(sub)),
        }

    return outputs


def draw_figure(curves: dict[str, dict[str, object]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.4, 6.4), dpi=400, facecolor="white")
    ax = fig.add_axes([0.12, 0.12, 0.82, 0.80])
    ax.set_facecolor("white")

    ax.plot(
        curves["english"]["fpr"],
        curves["english"]["tpr"],
        color="black",
        linewidth=2.2,
        linestyle="-",
        label=f"Inglés objetivo (AUC = {curves['english']['auc']:.4f})",
    )
    ax.plot(
        curves["czech"]["fpr"],
        curves["czech"]["tpr"],
        color="#5c5c5c",
        linewidth=2.2,
        linestyle="--",
        label=f"Checo objetivo (AUC = {curves['czech']['auc']:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#9a9a9a", linewidth=1.4, linestyle=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Tasa de falsos positivos", fontsize=14, family="DejaVu Sans")
    ax.set_ylabel("Tasa de verdaderos positivos", fontsize=14, family="DejaVu Sans")
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.legend(frameon=False, fontsize=12, loc="lower right")

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    pred, report = verify_sources()
    curves = compute_curves(pred, report)
    draw_figure(curves)
    print(f"SVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
