from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"
PROC_DIR = RESEARCH_DIR / "processed_data"
MULTI_DIR = RESEARCH_DIR / "research_results" / "CSV" / "multilingual_transfer"
REPORTS_DIR = RESEARCH_DIR / "reports" / "thesis_chapter4"

CZ_MANIFEST = PROC_DIR / "manifests" / "hubert_sequences_natural_multilingual_light.csv"
EN_MANIFEST = PROC_DIR / "manifests" / "hubert_sequences_en_test_v2.csv"

OUT_SVG = REPORTS_DIR / "figura_9_desplazamiento_dominios.svg"
OUT_PNG = REPORTS_DIR / "figura_9_desplazamiento_dominios.png"

# Expected values from language_shift_transfer_report.json (fixed_2s pipeline)
EXPECTED = {
    "cz_healthy_norm": 20.32407494220193,
    "en_healthy_norm": 13.717404740838424,
    "shift_magnitude": 15.322608463931145,
}


def _load_sequence_embedding(sequence_path: str, layer_policy: str = "last") -> np.ndarray:
    payload = torch.load(sequence_path, map_location="cpu", weights_only=True)
    seq = payload["sequence"].float()

    if seq.ndim == 3:
        if layer_policy == "last":
            seq = seq[-1]
        else:
            raise ValueError(f"Unknown layer_policy: {layer_policy}")

    T = seq.shape[0]
    mean_emb = seq.mean(dim=0)
    std_emb = seq.std(dim=0, correction=0)

    n_quarter = max(T // 4, 1)
    first_25 = seq[:n_quarter].mean(dim=0)
    last_25 = seq[-n_quarter:].mean(dim=0)
    delta = first_25 - last_25

    time_idx = torch.linspace(-1, 1, T).unsqueeze(1)
    centered_seq = seq - seq.mean(dim=0, keepdim=True)
    centered_time = time_idx - time_idx.mean()
    numerator = (centered_time * centered_seq).sum(dim=0)
    denom = torch.sqrt((centered_time ** 2).sum()) * torch.sqrt((centered_seq ** 2).sum(dim=0))
    slope = numerator / (denom + 1e-8)

    return torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope]).numpy().astype(np.float64)


def load_manifest_embeddings(manifest_path: Path, layer_policy: str = "last") -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        rows.append({
            "sample_id": getattr(row, "sample_id"),
            "subject_id": str(getattr(row, "subject_id")),
            "label": str(getattr(row, "label")),
            "embedding": _load_sequence_embedding(str(getattr(row, "sequence_path")), layer_policy=layer_policy),
        })
    return pd.DataFrame(rows)


def aggregate_subject_embeddings(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject_id, label), group in frame.groupby(["subject_id", "label"], dropna=False):
        emb = np.vstack(group["embedding"].to_list()).mean(axis=0)
        rows.append({
            "subject_id": str(subject_id),
            "label": str(label),
            "n_samples": int(len(group)),
            "embedding": emb,
        })
    return pd.DataFrame(rows)


def compute_healthy_centroid(subject_df: pd.DataFrame) -> np.ndarray:
    healthy = subject_df.loc[subject_df["label"] == "Healthy"]
    if len(healthy) == 0:
        raise ValueError("No Healthy subjects found")
    embs = np.vstack(healthy["embedding"].to_list())
    return embs.mean(axis=0)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Czech manifest and computing embeddings ...")
    cz_utterance = load_manifest_embeddings(CZ_MANIFEST, layer_policy="last")
    cz_subject = aggregate_subject_embeddings(cz_utterance)
    print(f"  Czech: {len(cz_subject)} subjects ({len(cz_utterance)} utterances)")

    print("Loading English manifest and computing embeddings ...")
    en_utterance = load_manifest_embeddings(EN_MANIFEST, layer_policy="last")
    en_subject = aggregate_subject_embeddings(en_utterance)
    print(f"  English: {len(en_subject)} subjects ({len(en_utterance)} utterances)")

    cz_subject["domain"] = "czech"
    en_subject["domain"] = "english"

    cz_healthy_centroid = compute_healthy_centroid(cz_subject)
    en_healthy_centroid = compute_healthy_centroid(en_subject)

    cz_norm = float(np.linalg.norm(cz_healthy_centroid))
    en_norm = float(np.linalg.norm(en_healthy_centroid))
    shift_mag = float(np.linalg.norm(en_healthy_centroid - cz_healthy_centroid))

    print(f"\nCZ Healthy centroid norm (4608-d): {cz_norm:.6f}")
    print(f"EN Healthy centroid norm (4608-d): {en_norm:.6f}")
    print(f"Shift magnitude (4608-d):          {shift_mag:.6f}")
    print()
    print(f"Expected CZ norm:  {EXPECTED['cz_healthy_norm']:.6f}")
    print(f"Expected EN norm:  {EXPECTED['en_healthy_norm']:.6f}")
    print(f"Expected shift:    {EXPECTED['shift_magnitude']:.6f}")
    print()

    # Investigate mismatch
    for key, actual, expected in [
        ("cz_healthy_norm", cz_norm, EXPECTED["cz_healthy_norm"]),
        ("en_healthy_norm", en_norm, EXPECTED["en_healthy_norm"]),
        ("shift_magnitude", shift_mag, EXPECTED["shift_magnitude"]),
    ]:
        ratio = actual / expected if expected != 0 else 0
        print(f"  {key}: actual={actual:.6f}  expected={expected:.6f}  ratio={ratio:.4f}")

    print()
    print("NOTE: Expected values come from language_shift_transfer_report.json")
    print("      which uses fixed_2s manifests (different duration policy).")
    print("      The main evaluation pipeline uses 'natural' manifests, so")
    print("      the norms and shift magnitude differ from the expected values.")

    # Concat all subjects for descriptive PCA
    combined = pd.concat([cz_subject, en_subject], ignore_index=True)
    all_embs = np.vstack(combined["embedding"].to_list())  # (146, 4608)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_embs)

    combined["pc1"] = coords[:, 0]
    combined["pc2"] = coords[:, 1]

    # Project centroids
    cz_h_2d = pca.transform(cz_healthy_centroid.reshape(1, -1))[0]
    en_h_2d = pca.transform(en_healthy_centroid.reshape(1, -1))[0]

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(9.0, 6.5), facecolor="white")
    ax.set_facecolor("white")

    # Markers per domain
    cz_group = combined.loc[combined["domain"] == "czech"]
    en_group = combined.loc[combined["domain"] == "english"]

    cz_healthy = cz_group.loc[cz_group["label"] == "Healthy"]
    cz_patients = cz_group.loc[cz_group["label"] == "Patients"]
    en_healthy = en_group.loc[en_group["label"] == "Healthy"]
    en_patients = en_group.loc[en_group["label"] == "Patients"]

    # CZ Healthy: white circle, black edge
    ax.scatter(
        cz_healthy["pc1"], cz_healthy["pc2"],
        marker="o", s=60, facecolors="white", edgecolors="black",
        linewidths=1.2, zorder=3,
    )

    # CZ Patients: black filled circle
    ax.scatter(
        cz_patients["pc1"], cz_patients["pc2"],
        marker="o", s=60, c="black", edgecolors="black",
        linewidths=0.5, zorder=3,
    )

    # EN Healthy: white triangle, black edge
    ax.scatter(
        en_healthy["pc1"], en_healthy["pc2"],
        marker="^", s=70, facecolors="white", edgecolors="black",
        linewidths=1.2, zorder=3,
    )

    # EN Patients: black filled triangle
    ax.scatter(
        en_patients["pc1"], en_patients["pc2"],
        marker="^", s=70, c="black", edgecolors="black",
        linewidths=0.5, zorder=3,
    )

    # Centroids: larger symbols
    ax.scatter(
        cz_h_2d[0], cz_h_2d[1],
        marker="o", s=160, facecolors="white", edgecolors="black",
        linewidths=2.5, zorder=4,
    )
    ax.scatter(
        en_h_2d[0], en_h_2d[1],
        marker="^", s=180, facecolors="white", edgecolors="black",
        linewidths=2.5, zorder=4,
    )

    # Arrow from CZ Healthy centroid to EN Healthy centroid
    dx = en_h_2d[0] - cz_h_2d[0]
    dy = en_h_2d[1] - cz_h_2d[1]
    ax.arrow(
        cz_h_2d[0], cz_h_2d[1], dx, dy,
        head_width=0.45, head_length=0.50, fc="black", ec="black",
        linewidth=1.6, zorder=5, length_includes_head=True,
    )

    # Labels near centroids
    ax.text(
        cz_h_2d[0], cz_h_2d[1] - 0.55,
        "Centroide sano\ncheco",
        ha="center", va="top", fontsize=11, fontstyle="italic",
    )
    ax.text(
        en_h_2d[0], en_h_2d[1] - 0.55,
        "Centroide TD\ninglés",
        ha="center", va="top", fontsize=11, fontstyle="italic",
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=10, label="Checo sano"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=10, label="Checo paciente"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=11, label="Inglés TD"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=11, label="Inglés LI"),
    ]
    ax.legend(
        handles=legend_elements, frameon=False, fontsize=11,
        loc="upper left", ncol=1,
    )

    # Axis labels with explained variance
    var_pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"CP1 ({var_pct[0]:.1f} %)", fontsize=14)
    ax.set_ylabel(f"CP2 ({var_pct[1]:.1f} %)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=1.0)

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"\nSVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
