from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT_DIR / "research" / "processed_data"
REPORTS_DIR = ROOT_DIR / "research" / "reports" / "thesis_chapter4"

PROJ_CSV = PROC_DIR / "pca_projection_fig9.csv"
CENT_CSV = PROC_DIR / "pca_centroids_fig9.csv"

OUT_SVG = REPORTS_DIR / "figura_9_desplazamiento_dominios_corregida.svg"
OUT_PNG = REPORTS_DIR / "figura_9_desplazamiento_dominios_corregida.png"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(str(PROJ_CSV))
    cents = pd.read_csv(str(CENT_CSV))

    cz_h = pts.loc[(pts["domain"] == "czech") & (pts["label"] == "Healthy")]
    cz_p = pts.loc[(pts["domain"] == "czech") & (pts["label"] == "Patients")]
    en_h = pts.loc[(pts["domain"] == "english") & (pts["label"] == "Healthy")]
    en_p = pts.loc[(pts["domain"] == "english") & (pts["label"] == "Patients")]

    cz_cent = cents.loc[cents["centroid"] == "CZ_Healthy"].iloc[0]
    en_cent = cents.loc[cents["centroid"] == "EN_Healthy"].iloc[0]
    cx, cy = float(cz_cent["pc1"]), float(cz_cent["pc2"])
    ex, ey = float(en_cent["pc1"]), float(en_cent["pc2"])

    var_cp1 = 63.3
    var_cp2 = 6.3

    fig, ax = plt.subplots(figsize=(9.0, 6.5), facecolor="white")
    ax.set_facecolor("white")

    # Individual markers with transparency
    ax.scatter(
        cz_h["pc1"], cz_h["pc2"],
        marker="o", s=45, facecolors="white", edgecolors="black",
        linewidths=1.0, alpha=0.80, zorder=2,
    )
    ax.scatter(
        cz_p["pc1"], cz_p["pc2"],
        marker="o", s=45, c="black", edgecolors="black",
        linewidths=0.4, alpha=0.80, zorder=2,
    )
    ax.scatter(
        en_h["pc1"], en_h["pc2"],
        marker="^", s=52, facecolors="white", edgecolors="black",
        linewidths=1.0, alpha=0.80, zorder=2,
    )
    ax.scatter(
        en_p["pc1"], en_p["pc2"],
        marker="^", s=52, c="black", edgecolors="black",
        linewidths=0.4, alpha=0.80, zorder=2,
    )

    # Centroids: no transparency, larger, higher zorder
    ax.scatter(
        cx, cy,
        marker="o", s=140, facecolors="white", edgecolors="black",
        linewidths=2.5, zorder=4,
    )
    ax.scatter(
        ex, ey,
        marker="^", s=160, facecolors="white", edgecolors="black",
        linewidths=2.5, zorder=4,
    )

    # Arrow between centroids
    dx = ex - cx
    dy = ey - cy
    ax.arrow(
        cx, cy, dx, dy,
        head_width=0.35, head_length=0.40, fc="black", ec="black",
        linewidth=1.1, zorder=5, length_includes_head=True,
    )

    # Arrow midpoint for label placement
    mx = (cx + ex) / 2
    my = (cy + ey) / 2
    arr_angle = np.arctan2(dy, dx)
    perp_angle = arr_angle + np.pi / 2
    offset_dist = 0.65
    lx = mx + offset_dist * np.cos(perp_angle)
    ly = my + offset_dist * np.sin(perp_angle)
    ax.text(
        lx, ly,
        "Dirección del\ndesplazamiento\nproyectado",
        ha="center", va="center", fontsize=8.5, color="#444444",
        style="italic",
    )

    # Centroid labels with leader lines
    # CZ centroid label: place below and slightly left
    ax.annotate(
        "Centroide Healthy checo",
        xy=(cx, cy), xytext=(cx - 1.5, cy - 3.2),
        ha="center", va="top", fontsize=11, fontstyle="italic",
        arrowprops=dict(
            arrowstyle="-", color="#777777", linewidth=0.8,
        ),
        zorder=6,
    )
    # EN centroid label: place above and slightly right
    ax.annotate(
        "Centroide TD ingl\u00e9s",
        xy=(ex, ey), xytext=(ex + 2.0, ey + 2.8),
        ha="center", va="bottom", fontsize=11, fontstyle="italic",
        arrowprops=dict(
            arrowstyle="-", color="#777777", linewidth=0.8,
        ),
        zorder=6,
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=8.5, label="Checo \u2014 Healthy"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=8.5, label="Checo \u2014 Patients"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=9.5, label="Ingl\u00e9s \u2014 TD"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
                   markeredgecolor="black", markersize=9.5, label="Ingl\u00e9s \u2014 LI"),
    ]
    leg = ax.legend(
        handles=legend_elements, frameon=False, fontsize=10.5,
        loc="upper left", ncol=1,
    )
    leg.set_zorder(10)

    # Axis labels
    ax.set_xlabel(f"CP1 ({var_cp1:.1f} %)", fontsize=14)
    ax.set_ylabel(f"CP2 ({var_cp2:.1f} %)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=1.0)

    fig.savefig(OUT_SVG, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, format="png", dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"SVG: {OUT_SVG}")
    print(f"PNG: {OUT_PNG}")
    print()
    print("--- Verification ---")
    print(f"Data file: {PROJ_CSV}")
    print(f"Centroid file: {CENT_CSV}")
    n_cz_h = len(cz_h)
    n_cz_p = len(cz_p)
    n_en_h = len(en_h)
    n_en_p = len(en_p)
    print(f"Participants: CZ Healthy={n_cz_h}, CZ Patients={n_cz_p}, EN TD={n_en_h}, EN LI={n_en_p}")
    print(f"Total: {n_cz_h + n_cz_p + n_en_h + n_en_p}")
    print(f"CP1: {var_cp1:.1f}%  CP2: {var_cp2:.1f}%")
    print("PCA is purely descriptive (2 components, visualization only).")
    print("Arrow shows a 2D projection of the domain shift, not the exact centroid_shift vector.")


if __name__ == "__main__":
    main()
