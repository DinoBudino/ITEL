from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.src.config import CONFIG


ATTN_DIR = CONFIG.paths.reports_dir / "attention_viz"
THESIS_DIR = CONFIG.paths.reports_dir.parent / "reports" / "thesis_chapter4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera espectrogramas tipo mapa de calor con regiones marcadas.")
    parser.add_argument(
        "--predictions",
        default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_natural.csv"),
        help="Predicciones usadas para reidentificar los casos A/B/C.",
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Manifest de muestras checas.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(THESIS_DIR),
        help="Directorio de salida.",
    )
    return parser.parse_args()


def _find_case(df: pd.DataFrame, label_str: str, pred_label_val: str, task_type: str, duration_range: tuple[float, float], subject_include: list[str] | None = None) -> pd.Series | None:
    mask = (
        (df["label_str"] == label_str)
        & (df["pred_label"] == pred_label_val)
        & (df["task_type"] == task_type)
        & (df["duration_ms"] >= duration_range[0])
        & (df["duration_ms"] < duration_range[1])
    )
    if subject_include:
        mask &= df["subject_id"].isin(subject_include)
    candidates = df[mask].sort_values("score_patients", ascending=False)
    return candidates.iloc[0] if len(candidates) else None


def _load_case_rows(predictions_path: str, sample_manifest_path: str) -> dict[str, pd.Series]:
    predictions = pd.read_csv(predictions_path)
    samples = pd.read_csv(sample_manifest_path)
    df = predictions.merge(samples[["sample_id", "subject_id", "task_type", "raw_audio_path"]], on=["sample_id", "subject_id"], how="left")
    df["label_str"] = df["label"].apply(lambda x: "Patients" if float(x) == 1.0 else "Healthy")
    df["duration_ms"] = [librosa.get_duration(path=path) * 1000 for path in df["raw_audio_path"]]

    critical = [s for s in ["H27", "H29", "H28", "H26"] if s in df["subject_id"].values]
    cases = {
        "A_TP_word": _find_case(df, "Patients", "Patients", "controlled_word", (600, 5000)),
        "B_FP_H27_H29_word": _find_case(df, "Healthy", "Patients", "controlled_word", (300, 2500), subject_include=critical[:2]),
        "C_FP_vowel": _find_case(df, "Healthy", "Patients", "controlled_vowel", (100, 2000)),
    }
    return {k: v for k, v in cases.items() if v is not None}


def _load_regions(case_tag: str) -> pd.DataFrame:
    path = ATTN_DIR / f"key_regions_{case_tag}.csv"
    return pd.read_csv(path)


def _plot_heatmap(audio_path: str, case_tag: str, title: str, subtitle: str, output_path: Path) -> None:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, win_length=400, n_mels=64, fmin=50, fmax=7600)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    duration_ms = len(y) / sr * 1000.0
    regions = _load_regions(case_tag)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    fig.patch.set_facecolor("white")
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, ha="left", va="bottom", color="#333333")
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.tick_params(labelsize=10)

    for idx, region in regions.iterrows():
        start_s = float(region["start_ms"]) / 1000.0
        width_s = float(region["duration_ms"]) / 1000.0
        ax.axvspan(start_s, start_s + width_s, color="#ffd166", alpha=0.30, ec="#ffffff", lw=2.0)
        label_x = start_s + width_s / 2
        if idx == 0:
            ax.text(label_x, 0.93, "Region de mayor atencion", color="white", fontsize=10, ha="center", va="center", fontweight="bold", transform=ax.get_xaxis_transform())

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Energia", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    note = f"Duracion total: {duration_ms:.0f} ms  |  Regiones marcadas: {len(regions)}"
    fig.text(0.01, 0.01, note, fontsize=9.5, ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_consonant_hard_case(predictions_path: str, sample_manifest_path: str, output_dir: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    samples = pd.read_csv(sample_manifest_path)
    df = predictions.merge(samples[["sample_id", "subject_id", "task_type", "raw_audio_path"]], on=["sample_id", "subject_id"], how="left")
    df = df[(df["task_type"] == "controlled_consonant") & (df["label"].astype(float) == 0.0) & (df["pred_label"] == "Patients")]
    if df.empty:
        return

    row = df.sort_values("score_patients", ascending=False).iloc[0]
    y, sr = librosa.load(str(row["raw_audio_path"]), sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, win_length=400, n_mels=64, fmin=50, fmax=7600)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    duration_s = len(y) / sr

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    fig.patch.set_facecolor("white")
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    ax.set_title("Figura 11. Espectrograma tipo mapa de calor en una consonante de alta dificultad", fontsize=15, fontweight="bold", pad=14)
    ax.text(0.0, 1.02, "Las consonantes aisladas contienen poca estructura temporal y suelen aportar menos evidencia discriminativa.", transform=ax.transAxes, fontsize=10.5, ha="left", va="bottom", color="#333333")
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.axvspan(0, duration_s, color="#9ecae1", alpha=0.10)
    ax.text(duration_s * 0.50, 0.93, "Estructura temporal breve", color="white", fontsize=10, ha="center", va="center", fontweight="bold", transform=ax.get_xaxis_transform())
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Energia", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.text(0.01, 0.01, f"Duracion total: {duration_s*1000:.0f} ms  |  Caso: {row['sample_id']}", fontsize=9.5, ha="left")
    output_path = output_dir / "figura_11_heatmap_consonante_dificil.png"
    plt.savefig(output_path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_case_rows(args.predictions, args.sample_manifest)

    titles = {
        "A_TP_word": (
            "Figura 9. Espectrograma tipo mapa de calor en un caso Patient bien clasificado",
            "Las regiones resaltadas concentran la evidencia temporal mas relevante para la decision del modelo.",
            output_dir / "figura_9_heatmap_tp_patient.png",
        ),
        "B_FP_H27_H29_word": (
            "Figura 10. Espectrograma tipo mapa de calor en un falso positivo de sujeto sano",
            "El modelo focaliza regiones breves que pueden inducir una decision errada hacia Patients.",
            output_dir / "figura_10_heatmap_fp_healthy.png",
        ),
        "C_FP_vowel": (
            "Figura 11. Espectrograma tipo mapa de calor en vocal aislada con decision inestable",
            "Las vocales aisladas contienen menos estructura temporal y ofrecen evidencia menos robusta.",
            output_dir / "figura_11_heatmap_fp_vowel.png",
        ),
    }

    for case_tag, row in cases.items():
        title, subtitle, out_path = titles[case_tag]
        _plot_heatmap(str(row["raw_audio_path"]), case_tag, title, subtitle, out_path)
        print(f"Generated {out_path}")

    _plot_consonant_hard_case(args.predictions, args.sample_manifest, output_dir)

    notes = output_dir / "figuras_heatmap_notas.md"
    notes.write_text(
        "# Notas sugeridas para figuras heatmap\n\n"
        "## Figura 9\n"
        "Espectrograma tipo mapa de calor en un caso Patient bien clasificado.\n\n"
        "Nota. La figura muestra un ejemplo representativo en el que el modelo concentra su atencion en regiones temporales breves del espectrograma. Las zonas resaltadas corresponden a ventanas de mayor peso de decision. Elaboracion propia.\n\n"
        "## Figura 10\n"
        "Espectrograma tipo mapa de calor en un falso positivo de sujeto sano.\n\n"
        "Nota. La figura ilustra un caso en el que el modelo asigna alto peso a una region breve del audio, generando una clasificacion erronea hacia Patients. Elaboracion propia.\n\n"
        "## Figura 11\n"
        "Espectrograma tipo mapa de calor en una consonante de alta dificultad.\n\n"
        "Nota. La figura ilustra un caso representativo del grupo de consonantes, que presenta menor estructura temporal y, por tanto, una señal mas dificil de discriminar para el modelo. Elaboracion propia.\n",
        encoding="utf-8",
    )
    print(f"Generated {notes}")


if __name__ == "__main__":
    main()
