from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.src.config import CONFIG
from research.src.multilingual_features import STATISTICAL_BLOCK_NAMES, load_statistical_sequence_embedding


REPORT_DIR = CONFIG.paths.reports_dir / "CSV" / "multilingual_transfer"
REPORTS_FIG_DIR = ROOT_DIR / "research" / "reports" / "thesis_chapter4"

CZ_MANIFEST = CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural_multilingual_light.csv"
EN_MANIFEST = CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test_v2.csv"

BLOCK_NAMES = list(STATISTICAL_BLOCK_NAMES)
BLOCK_SIZE = 768


def load_subject_embeddings(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    utt_rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        utt_rows.append({
            "subject_id": str(getattr(row, "subject_id")),
            "label": str(getattr(row, "label")),
            "embedding": load_statistical_sequence_embedding(getattr(row, "sequence_path"), layer_policy="last"),
        })
    utt = pd.DataFrame(utt_rows)
    subj_rows: list[dict] = []
    for (sid, lbl), grp in utt.groupby(["subject_id", "label"], dropna=False):
        subj_rows.append({
            "subject_id": str(sid),
            "label": str(lbl),
            "embedding": np.vstack(grp["embedding"].to_list()).mean(axis=0),
        })
    return pd.DataFrame(subj_rows)


def healthy_centroid(frame: pd.DataFrame) -> np.ndarray:
    healthy = frame.loc[frame["label"] == "Healthy", "embedding"]
    return np.vstack(healthy.to_list()).mean(axis=0)


def centroid_shift(source: pd.DataFrame, target_train: pd.DataFrame) -> pd.DataFrame:
    shift_vec = healthy_centroid(source) - healthy_centroid(target_train)
    shifted = source.copy()
    shifted["embedding"] = shifted["embedding"].apply(lambda z: z - shift_vec)
    return shifted


def fit_and_extract(X_train: np.ndarray, y_train: pd.Series) -> dict:
    n_components = int(min(64, X_train.shape[0] - 1, X_train.shape[1]))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000, solver="lbfgs")),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_train, y_train)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    pca: PCA = pipe.named_steps["pca"]
    lr: LogisticRegression = pipe.named_steps["clf"]

    classes = lr.classes_.tolist()
    pos_idx = classes.index("Patients") if "Patients" in classes else 1
    w_pca = lr.coef_[0]  # shape [n_components]

    # Reproject through PCA: w_std = components_.T @ w_pca  (shape [4608])
    w_std = pca.components_.T @ w_pca

    # Account for StandardScaler: w_orig[j] = w_std[j] / scale_[j]
    scale = scaler.scale_.astype(np.float64)
    w_orig = w_std / scale

    # Per-block L1 contribution
    block_sums = []
    for b in range(6):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        block_sums.append(float(np.sum(np.abs(w_orig[start:end]))))

    total = sum(block_sums)
    block_rel = [s / total if total > 0 else 0.0 for s in block_sums]

    # Also store individual weights for optional finer analysis
    return {
        "w_orig": w_orig,
        "block_l1": block_sums,
        "block_rel": block_rel,
        "n_components": n_components,
    }


def run_loso(source: pd.DataFrame, target: pd.DataFrame) -> list[dict]:
    target_ids = sorted(set(target["subject_id"].astype(str).tolist()))
    all_fold: list[dict] = []
    for test_sub in target_ids:
        tgt_train = target.loc[target["subject_id"].astype(str) != test_sub].copy()
        src_aligned = centroid_shift(source, tgt_train)
        train = pd.concat([src_aligned, tgt_train], ignore_index=True)
        X = np.vstack(train["embedding"].to_list())
        y = train["label"].reset_index(drop=True)
        result = fit_and_extract(X, y)
        result["test_subject"] = test_sub
        all_fold.append(result)
    return all_fold


def aggregate_blocks(folds: list[dict]) -> dict:
    n_folds = len(folds)
    block_rel = np.array([f["block_rel"] for f in folds])  # [n_folds, 6]
    means = block_rel.mean(axis=0) * 100
    stds = block_rel.std(axis=0) * 100
    return {"means": means.tolist(), "stds": stds.tolist(), "n_folds": n_folds}


def compare_domains(en_agg: dict, cz_agg: dict) -> dict:
    en_means = np.asarray(en_agg["means"], dtype=float)
    cz_means = np.asarray(cz_agg["means"], dtype=float)
    en_order = np.argsort(-en_means)
    cz_order = np.argsort(-cz_means)
    differences = en_means - cz_means
    return {
        "same_importance_order": bool(np.array_equal(en_order, cz_order)),
        "en_target_order": [BLOCK_NAMES[index] for index in en_order],
        "cz_target_order": [BLOCK_NAMES[index] for index in cz_order],
        "en_minus_cz_percentage_points": differences.tolist(),
        "max_absolute_difference_percentage_points": float(np.max(np.abs(differences))),
    }


def main() -> None:
    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading subject-level embeddings ...")
    cz_sub = load_subject_embeddings(CZ_MANIFEST)
    en_sub = load_subject_embeddings(EN_MANIFEST)
    print(f"  CZ: {len(cz_sub)} subjects")
    print(f"  EN: {len(en_sub)} subjects")

    # Evaluate in both directions
    print("\nEN-target LOSO (CZ -> EN, centroid_shift) ...")
    en_folds = run_loso(cz_sub, en_sub)
    print(f"  {len(en_folds)} folds complete")

    print("CZ-target LOSO (EN -> CZ, centroid_shift) ...")
    cz_folds = run_loso(en_sub, cz_sub)
    print(f"  {len(cz_folds)} folds complete")

    # Aggregate per domain
    en_agg = aggregate_blocks(en_folds)
    cz_agg = aggregate_blocks(cz_folds)
    domain_comparison = compare_domains(en_agg, cz_agg)

    # Combined
    all_folds = en_folds + cz_folds
    all_agg = aggregate_blocks(all_folds)

    print("\n===== BLOCK CONTRIBUTION RESULTS (relative %) =====")
    print(f"{'Block':<35} {'All folds':>10} {'EN-target':>10} {'CZ-target':>10}")
    print("-" * 67)
    for i, name in enumerate(BLOCK_NAMES):
        print(f"{name:<35} "
              f"{all_agg['means'][i]:>8.2f} ± {all_agg['stds'][i]:.2f} "
              f"{en_agg['means'][i]:>8.2f} ± {en_agg['stds'][i]:.2f} "
              f"{cz_agg['means'][i]:>8.2f} ± {cz_agg['stds'][i]:.2f}")
    print("-" * 67)
    print(f"{'N folds':<35} {all_agg['n_folds']:>10} {en_agg['n_folds']:>10} {cz_agg['n_folds']:>10}")

    print("\n===== AGGREGATION ===== (all folds combined =====")
    for i, name in enumerate(BLOCK_NAMES):
        print(f"{name}: {all_agg['means'][i]:.2f}% ± {all_agg['stds'][i]:.2f}")

    # Save numeric results
    results = {
        "block_names": BLOCK_NAMES,
        "all_folds": all_agg,
        "en_target": en_agg,
        "cz_target": cz_agg,
        "domain_comparison": domain_comparison,
        "figure_scope": "all_folds_combined",
    }
    out_json = REPORT_DIR / "block_contributions.json"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {out_json}")

    # Generate figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="white")
    ax.set_facecolor("white")

    means = all_agg["means"]
    stds = all_agg["stds"]
    y_pos = np.arange(len(BLOCK_NAMES))

    # Only black and grey
    bar_colors = ["#333333", "#555555", "#777777", "#777777", "#555555", "#333333"]

    ax.barh(
        y_pos, means, xerr=stds,
        color=bar_colors, edgecolor="black", linewidth=0.8,
        capsize=3, error_kw={"linewidth": 1.2, "ecolor": "black"},
        height=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(BLOCK_NAMES, fontsize=12)
    ax.set_xlabel("Contribución relativa media (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color("black")
        ax.spines[spine_name].set_linewidth(1.0)

    # Annotate bar values on the right
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.3, i, f"{m:.1f}%", va="center", fontsize=10)

    plt.tight_layout(pad=1.0)

    out_svg = REPORTS_FIG_DIR / "figura_10_contribucion_bloques.svg"
    out_png = REPORTS_FIG_DIR / "figura_10_contribucion_bloques.png"
    fig.savefig(out_svg, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_png, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"\nSVG: {out_svg}")
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
