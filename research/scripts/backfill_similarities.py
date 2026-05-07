"""Backfill speaker_similarities into existing diarization JSONs and summary CSV.

For files that were already diarized (before speaker_similarities was added),
this script re-runs ONLY the embedding step — it reads the existing segment JSON,
loads the raw audio, computes per-speaker embeddings, and saves cosine similarities
against the interviewer reference.

No diarization is re-run. Runtime: ~10-20s per file on CPU.

Usage:
    python research/scripts/backfill_similarities.py \
        --reference-embedding assets/models/interviewer_reference_embedding.npy

Optional:
    --device cpu    Override torch device
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill speaker_similarities into existing diarization results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference-embedding",
        required=True,
        help="Path to the interviewer reference embedding .npy",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(CONFIG.paths.diarization_dir / "english_external_diarization_summary.csv"),
        help="Diarization summary CSV to update.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-compute similarities even for files that already have them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_path = Path(args.reference_embedding)
    if not ref_path.is_absolute():
        ref_path = ROOT_DIR / ref_path
    if not ref_path.exists():
        print(f"[ERROR] Reference embedding not found: {ref_path}")
        sys.exit(1)

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"[ERROR] Summary CSV not found: {summary_path}")
        sys.exit(1)

    reference_embedding = np.load(str(ref_path))
    print(f"Loaded reference embedding: {ref_path.name}  shape={reference_embedding.shape}")

    print("Loading embedding model…")
    embedding_model = load_embedding_model()
    print("Model ready.\n")

    df = pd.read_csv(summary_path)

    # Ensure columns exist
    for col in ["speaker_similarities", "interviewer_speaker", "interviewer_sim", "chi_sim"]:
        if col not in df.columns:
            df[col] = None

    updated = 0
    skipped = 0
    failed = 0

    for idx, row in df.iterrows():
        file_name = row["file_name"]
        json_path = Path(str(row.get("segments_json_path", "") or ""))

        # Skip if already filled (unless --force)
        if not args.force and pd.notna(row.get("speaker_similarities")) and str(row["speaker_similarities"]) not in ("", "None", "nan"):
            skipped += 1
            continue

        if not json_path.exists():
            print(f"  [skip] {file_name}: segment JSON not found at {json_path}")
            failed += 1
            continue

        # Load segment data
        with json_path.open(encoding="utf-8") as f:
            seg_data = json.load(f)

        segments = seg_data.get("segments", [])
        if not segments:
            print(f"  [skip] {file_name}: no segments in JSON")
            failed += 1
            continue

        # Load raw audio
        source_path = Path(str(seg_data.get("source_audio_path", "")))
        if not source_path.exists():
            # Try resolving relative to project root
            source_path = ROOT_DIR / CONFIG.paths.raw_en_dir / file_name
        if not source_path.exists():
            print(f"  [skip] {file_name}: source audio not found")
            failed += 1
            continue

        try:
            waveform, sr = load_audio_mono_16k(source_path)
            speaker_embeddings = extract_speaker_embeddings_from_segments(
                waveform, sr, segments, embedding_model
            )
            _, similarities, confident = identify_interviewer_by_reference(
                speaker_embeddings, reference_embedding
            )

            interviewer_speaker = max(similarities, key=lambda k: similarities[k]) if similarities else None
            interviewer_sim = round(similarities[interviewer_speaker], 4) if interviewer_speaker else None
            chi_speaker = str(row.get("chi_speaker") or "")
            chi_sim = round(similarities.get(chi_speaker, float("nan")), 4) if (similarities and chi_speaker) else None

            # Update JSON with similarities
            seg_data["speaker_similarities"] = {k: round(v, 4) for k, v in similarities.items()}
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(seg_data, f, indent=2)

            # Update dataframe row
            df.at[idx, "speaker_similarities"] = str({k: round(v, 4) for k, v in similarities.items()})
            df.at[idx, "interviewer_speaker"] = interviewer_speaker
            df.at[idx, "interviewer_sim"] = interviewer_sim
            df.at[idx, "chi_sim"] = chi_sim

            flag = "✓" if confident else "⚠ low-conf"
            print(f"  [{flag}] {file_name}: {similarities}  CHI={chi_speaker}(sim={chi_sim})  EXA={interviewer_speaker}(sim={interviewer_sim})")
            updated += 1

        except Exception as exc:
            print(f"  [ERROR] {file_name}: {exc}")
            failed += 1

    df.to_csv(summary_path, index=False)
    print(f"\nDone. Updated: {updated}  Skipped (already filled): {skipped}  Failed: {failed}")
    print(f"Summary saved to: {summary_path}")

    # Print a quick analysis table
    if updated > 0 or skipped > 0:
        print("\n── Similarity distribution ──────────────────────────────")
        filled = df[df["interviewer_sim"].notna()]
        print(f"  Files with similarity data: {len(filled)}")
        if len(filled) > 0:
            print(f"  Interviewer sim  — mean={filled['interviewer_sim'].mean():.3f}  "
                  f"min={filled['interviewer_sim'].min():.3f}  max={filled['interviewer_sim'].max():.3f}")
            low_conf = filled[filled["interviewer_sim"] < 0.12]
            print(f"  Low-confidence (<0.12, used fallback): {len(low_conf)} files")
            if len(low_conf):
                print(low_conf[["file_name", "label", "interviewer_sim", "chi_sim", "chi_strategy"]].to_string(index=False))


if __name__ == "__main__":
    main()
