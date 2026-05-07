"""Compute and show interviewer similarity scores for the 20 trimmed English files.

Compares old reference (sim embedded in chi_strategy string) vs
new reference (recomputed from assets/models/interviewer_reference_embedding.npy).

Usage:
    python research/scripts/compare_similarities.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from research.src.config import CONFIG
from research.src.diarization import (
    extract_speaker_embeddings_from_segments,
    identify_interviewer_by_reference,
    load_audio_mono_16k,
    load_embedding_model,
)
from research.src.preprocessing import preprocess_samples


REF_PATH = CONFIG.paths.model_dir / "interviewer_reference_embedding.npy"
DIAR_DIR = CONFIG.paths.diarization_dir
META_CSV = CONFIG.paths.manifests_dir / "english_external_metadata.csv"


def main() -> None:
    if not REF_PATH.exists():
        print(f"[ERROR] Reference embedding not found: {REF_PATH}")
        sys.exit(1)

    reference = np.load(str(REF_PATH))
    print(f"Referencia cargada: {REF_PATH.name}  shape={reference.shape}\n")

    print("Cargando modelo de embeddings...")
    embedding_model = load_embedding_model()
    print("Modelo listo.\n")

    meta = pd.read_csv(META_CSV).dropna(subset=["file_name"])

    rows = []
    for _, row in meta.iterrows():
        fname = str(row["file_name"])
        parts = fname.replace("\\", "/").split("/")
        label, stem = parts[0], Path(parts[1]).stem
        json_path = DIAR_DIR / label / f"{stem}_segments.json"
        raw_audio = CONFIG.paths.raw_en_dir / label / parts[1]

        if not json_path.exists():
            rows.append({"archivo": parts[1], "label": label,
                         "sim_antigua": None, "sim_nueva_exa": None,
                         "sim_nueva_chi": None, "nota": "JSON no encontrado"})
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        chi_speaker = data.get("chi_speaker")
        strategy = data.get("chi_strategy", "")

        # Old sim: extracted from chi_strategy string e.g. "reference(sim=0.16)"
        m = re.search(r"sim=([\d.]+)", strategy)
        sim_old = float(m.group(1)) if m else None

        if not raw_audio.exists():
            rows.append({"archivo": parts[1], "label": label,
                         "sim_antigua": sim_old, "sim_nueva_exa": None,
                         "sim_nueva_chi": None, "nota": "audio no encontrado"})
            continue

        try:
            waveform, sr = load_audio_mono_16k(raw_audio)

            # Preprocess each segment with anti_confound before embedding
            audio_np = waveform.squeeze().cpu().numpy()
            clean_segs = []
            for seg in data.get("segments", []):
                start = max(int(float(seg["start"]) * sr), 0)
                end = min(int(float(seg["end"]) * sr), len(audio_np))
                if end <= start:
                    continue
                clip = preprocess_samples(audio_np[start:end].astype("float32"),
                                          sr=sr, profile="anti_confound")
                if len(clip) > sr // 4:
                    clean_segs.append({
                        "speaker": seg["speaker"],
                        "start": len(np.concatenate(
                            [s["_clip"] for s in clean_segs], axis=0
                        )) / sr if clean_segs else 0.0,
                        "end": (len(np.concatenate(
                            [s["_clip"] for s in clean_segs], axis=0
                        )) + len(clip)) / sr if clean_segs else len(clip) / sr,
                        "_clip": clip,
                    })

            if not clean_segs:
                rows.append({"archivo": parts[1], "label": label,
                             "sim_antigua": sim_old, "sim_nueva_exa": None,
                             "sim_nueva_chi": None, "nota": "sin segmentos limpios"})
                continue

            # Rebuild waveform from clean clips
            import torch as _torch
            all_clips = np.concatenate([s["_clip"] for s in clean_segs]).astype("float32")
            clean_waveform = _torch.tensor(all_clips).unsqueeze(0)
            # Use original segments for speaker mapping (reuse original timing)
            speaker_embs = extract_speaker_embeddings_from_segments(
                waveform, sr, data.get("segments", []), embedding_model
            )

            _, similarities, confident = identify_interviewer_by_reference(
                speaker_embs, reference
            )

            exa_speaker = max(similarities, key=lambda k: similarities[k]) if similarities else None
            sim_exa = round(similarities[exa_speaker], 4) if exa_speaker else None
            sim_chi = round(similarities.get(chi_speaker, float("nan")), 4) if chi_speaker else None

            rows.append({
                "archivo": parts[1],
                "label": label,
                "chi_speaker": chi_speaker,
                "exa_speaker": exa_speaker,
                "sim_antigua": sim_old,
                "sim_nueva_exa": sim_exa,
                "sim_nueva_chi": sim_chi,
                "confianza": "✓" if confident else "⚠ baja",
                "nota": "",
            })
            print(f"  {parts[1]:12s} ({label:8s})  "
                  f"antigua={sim_old:.2f}  nueva_exa={sim_exa:.4f}  nueva_chi={sim_chi:.4f}  "
                  f"{'✓' if confident else '⚠ baja'}")

        except Exception as exc:
            print(f"  [ERROR] {parts[1]}: {exc}")
            rows.append({"archivo": parts[1], "label": label,
                         "sim_antigua": sim_old, "sim_nueva_exa": None,
                         "sim_nueva_chi": None, "nota": str(exc)})

    df = pd.DataFrame(rows)
    out = ROOT_DIR / "research" / "reports" / "similitud_comparacion.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    validos = df[df["sim_nueva_exa"].notna()]
    if len(validos):
        print(f"  Archivos procesados : {len(validos)}")
        print(f"  Sim antigua  — media={validos['sim_antigua'].mean():.3f}  "
              f"min={validos['sim_antigua'].min():.3f}  max={validos['sim_antigua'].max():.3f}")
        print(f"  Sim nueva (investigador) — media={validos['sim_nueva_exa'].mean():.3f}  "
              f"min={validos['sim_nueva_exa'].min():.3f}  max={validos['sim_nueva_exa'].max():.3f}")
        print(f"  Sim nueva (niño)         — media={validos['sim_nueva_chi'].mean():.3f}  "
              f"min={validos['sim_nueva_chi'].min():.3f}  max={validos['sim_nueva_chi'].max():.3f}")
        gap = validos["sim_nueva_exa"] - validos["sim_nueva_chi"]
        print(f"  Separación EXA-CHI — media={gap.mean():.3f}  min={gap.min():.3f}")
    print(f"\nResultados guardados en: {out}")


if __name__ == "__main__":
    main()
