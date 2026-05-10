from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from research.src.config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real duration distribution of LANNA Czech audio files.")
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
    )
    parser.add_argument(
        "--output-plot",
        default=str(CONFIG.paths.reports_dir / "CSV" / "duration_distribution_cz.png"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(CONFIG.paths.reports_dir / "CSV" / "duration_stats_cz.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.sample_manifest)

    durations: list[float] = []
    labels: list[str] = []
    errors = 0

    for row in tqdm(manifest.to_dict("records"), desc="reading durations"):
        try:
            dur = librosa.get_duration(path=row["raw_audio_path"])
            durations.append(dur)
            labels.append(row["label"])
        except Exception:
            errors += 1

    dur_arr = np.array(durations)
    print(f"\nTotal files: {len(manifest)}, read: {len(dur_arr)}, errors: {errors}")
    print(f"Min:    {dur_arr.min():.3f}s")
    print(f"Max:    {dur_arr.max():.3f}s")
    print(f"Mean:   {dur_arr.mean():.3f}s")
    print(f"Median: {np.median(dur_arr):.3f}s")
    print(f"Std:    {dur_arr.std():.3f}s")
    print()

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, float("inf")]
    for t in thresholds:
        if t == float("inf"):
            print(f"  >= {thresholds[-2]:.1f}s: {(dur_arr >= thresholds[-2]).sum():>5}")
        else:
            thresh_prev = thresholds[thresholds.index(t) - 1] if thresholds.index(t) > 0 else 0
            mask = (dur_arr >= thresh_prev) & (dur_arr < t)
            print(f"  [{thresh_prev:4.1f}, {t:4.1f})s: {mask.sum():>5}")

    healthy_dur = np.array([d for d, l in zip(durations, labels) if l == "Healthy"])
    patients_dur = np.array([d for d, l in zip(durations, labels) if l == "Patients"])
    print(f"\nHealthy:  mean={healthy_dur.mean():.3f}s median={np.median(healthy_dur):.3f}s")
    print(f"Patients: mean={patients_dur.mean():.3f}s median={np.median(patients_dur):.3f}s")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = 60
    ax = axes[0]
    ax.hist(dur_arr, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="current 2.0s crop/pad")
    ax.axvline(x=np.median(dur_arr), color="orange", linestyle="--", linewidth=1.5, label=f"median ({np.median(dur_arr):.2f}s)")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("LANNA Czech Audio Duration Distribution")
    ax.legend()

    ax2 = axes[1]
    bins2 = 40
    ax2.hist(healthy_dur, bins=bins2, alpha=0.6, label=f"Healthy (n={len(healthy_dur)})", color="green")
    ax2.hist(patients_dur, bins=bins2, alpha=0.6, label=f"Patients (n={len(patients_dur)})", color="darkorange")
    ax2.axvline(x=2.0, color="red", linestyle="--", linewidth=2, label="2.0s boundary")
    ax2.set_xlabel("Duration (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Duration by Label")
    ax2.legend()

    plt.tight_layout()
    Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {args.output_plot}")

    stats_df = pd.DataFrame({
        "percentile": [5, 10, 25, 50, 75, 90, 95],
        "duration_s": [np.percentile(dur_arr, p) for p in [5, 10, 25, 50, 75, 90, 95]],
    })
    stats_df.loc[len(stats_df)] = ["<0.5s", (dur_arr < 0.5).sum()]
    stats_df.loc[len(stats_df)] = ["0.5-1.0s", ((dur_arr >= 0.5) & (dur_arr < 1.0)).sum()]
    stats_df.loc[len(stats_df)] = ["1.0-1.5s", ((dur_arr >= 1.0) & (dur_arr < 1.5)).sum()]
    stats_df.loc[len(stats_df)] = ["1.5-2.0s", ((dur_arr >= 1.5) & (dur_arr < 2.0)).sum()]
    stats_df.loc[len(stats_df)] = [">=2.0s", (dur_arr >= 2.0).sum()]
    stats_df.loc[len(stats_df)] = ["mean", dur_arr.mean()]
    stats_df.loc[len(stats_df)] = ["median", np.median(dur_arr)]
    stats_df.loc[len(stats_df)] = ["std", dur_arr.std()]
    stats_df.to_csv(args.output_csv, index=False)
    print(f"CSV saved: {args.output_csv}")


if __name__ == "__main__":
    main()
