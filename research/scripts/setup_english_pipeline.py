"""One-shot setup script for the English diarization pipeline.

Steps executed automatically:
  1. Select 40 TD + 40 SLI files from Datos_Extra (balanced across A/B subgroups)
  2. Copy them to English_Data/Healthy/ and English_Data/Patients/
  3. Bootstrap the metadata CSV (register new files)
  4. Run diarization using the interviewer reference embedding
  5. Build the English sample manifest
  6. Extract XLSR embeddings
  7. Evaluate with the trained model

Usage:
    python research/scripts/setup_english_pipeline.py \
        --reference-embedding assets/models/interviewer_reference_embedding.npy

Optional flags:
    --n-per-label       Number of files per label (default: 40)
    --source-dir        Root of the extra English audio (default: research/Datos_Extra/INGLES)
    --skip-copy         Skip step 1-2 if files are already in English_Data
    --skip-diarization  Skip diarization (if already done)
    --skip-embeddings   Skip embedding extraction
    --eval-mode         Embedding eval mode: seg_2s | seg_5s | full_audio (default: seg_2s)
    --device            Torch device override (e.g. cpu, cuda)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "TD": ("Healthy", ["A - TD", "B - TD"]),
    "SLI": ("Patients", ["A - SLI", "B - SLI"]),
}


def select_files(source_dir: Path, subfolders: list[str], n: int) -> list[Path]:
    """Pick *n* files balanced across *subfolders* (round-robin)."""
    pools: list[list[Path]] = []
    for sf in subfolders:
        folder = source_dir / sf
        if folder.exists():
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp3")
            pools.append(files)
        else:
            print(f"  [warn] subfolder not found: {folder}")

    if not pools:
        return []

    selected: list[Path] = []
    idx = [0] * len(pools)
    pool_idx = 0
    while len(selected) < n:
        pool = pools[pool_idx % len(pools)]
        i = idx[pool_idx % len(pools)]
        if i < len(pool):
            selected.append(pool[i])
            idx[pool_idx % len(pools)] += 1
        pool_idx += 1
        # Break if all pools exhausted
        if all(idx[j] >= len(pools[j]) for j in range(len(pools))):
            break

    return selected[:n]


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"▶  {desc}")
    print(f"   {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up and run the full English diarization pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference-embedding",
        required=True,
        help="Path to the interviewer reference embedding .npy (from extract_interviewer_reference.py).",
    )
    parser.add_argument(
        "--n-per-label",
        type=int,
        default=40,
        help="Number of audio files to use per label (TD/SLI). Default: 40.",
    )
    parser.add_argument(
        "--source-dir",
        default=str(ROOT_DIR / "research" / "Datos_Extra" / "INGLES"),
        help="Root folder containing TD/ and SLI/ subgroups.",
    )
    parser.add_argument(
        "--diarization-only",
        action="store_true",
        help="Run only file copy + diarization. Stop after CHI wav files are generated.",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip file copy — assume files are already in English_Data/.",
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip diarization step.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding extraction step.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["seg_2s", "seg_5s", "full_audio"],
        default="seg_2s",
        help="Embedding eval mode (default: seg_2s).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    ref_embedding = Path(args.reference_embedding)

    if not ref_embedding.exists():
        print(
            f"[ERROR] Reference embedding not found: {ref_embedding}\n"
            "Run extract_interviewer_reference.py --interviewer-speaker SPEAKER_01 first."
        )
        sys.exit(1)

    python = sys.executable

    # ------------------------------------------------------------------
    # STEP 1-2: Copy files to English_Data/
    # ------------------------------------------------------------------
    if not args.skip_copy:
        print(f"\n{'=' * 60}")
        print(f"STEP 1/6  Selecting and copying {args.n_per_label} files per label")
        print(f"{'=' * 60}")

        copied_total = 0
        for group_key, (label, subfolders) in LABEL_MAP.items():
            group_src = source_dir / group_key
            dest_dir = CONFIG.paths.raw_en_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            files = select_files(group_src, subfolders, args.n_per_label)
            if not files:
                print(f"  [warn] No files found for {group_key} in {group_src}")
                continue

            print(f"\n  {group_key} → English_Data/{label}/ ({len(files)} files)")
            for src in files:
                dst = dest_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"    copied  {src.name}")
                else:
                    print(f"    exists  {src.name}")
            copied_total += len(files)

        print(f"\n  Total: {copied_total} files ready in English_Data/")
    else:
        print("\nSTEP 1-2 skipped (--skip-copy)")

    # ------------------------------------------------------------------
    # STEP 3: Bootstrap metadata
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 2/6  Bootstrapping English metadata")
    print(f"{'=' * 60}")
    run([python, "research/scripts/bootstrap_en_metadata.py"], "bootstrap_en_metadata")

    # ------------------------------------------------------------------
    # STEP 4: Diarization with reference embedding
    # ------------------------------------------------------------------
    if not args.skip_diarization:
        print(f"\n{'=' * 60}")
        print("STEP 3/6  Running diarization (reference strategy)")
        print(f"{'=' * 60}")
        diar_cmd = [
            python, "research/scripts/run_diarization.py",
            "--all",
            "--skip-completed",
            "--chi-strategy", "reference",
            "--reference-embedding", str(ref_embedding),
        ]
        if args.device:
            diar_cmd += ["--device", args.device]
        run(diar_cmd, "run_diarization")
    else:
        print("\nSTEP 3 skipped (--skip-diarization)")

    if args.diarization_only:
        print(f"\n{'=' * 60}")
        print("✓ Diarization complete.")
        print(f"  CHI audio files saved to:")
        print(f"  research/processed_data/audio/english_external/child_only/")
        print(f"{'=' * 60}\n")
        return

    # ------------------------------------------------------------------
    # STEP 5: Build English sample manifest
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 4/6  Building English sample manifest")
    print(f"{'=' * 60}")
    run(
        [python, "research/scripts/build_en_manifest.py", "--completed-only"],
        "build_en_manifest",
    )

    # ------------------------------------------------------------------
    # STEP 6: Extract embeddings
    # ------------------------------------------------------------------
    if not args.skip_embeddings:
        print(f"\n{'=' * 60}")
        print(f"STEP 5/6  Extracting embeddings (eval-mode: {args.eval_mode})")
        print(f"{'=' * 60}")
        run(
            [
                python, "research/scripts/build_embeddings.py",
                "--dataset", "english_external",
                "--eval-mode", args.eval_mode,
            ],
            f"build_embeddings --eval-mode {args.eval_mode}",
        )
    else:
        print("\nSTEP 5 skipped (--skip-embeddings)")

    # ------------------------------------------------------------------
    # STEP 7: Evaluate
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 6/6  Evaluating with trained model")
    print(f"{'=' * 60}")
    run([python, "research/scripts/evaluate_external.py"], "evaluate_external")

    print(f"\n{'=' * 60}")
    print("✓ Pipeline complete. Check research/reports/ for results.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
