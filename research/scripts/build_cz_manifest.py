from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.data_manifest import SAMPLE_MANIFEST_COLUMNS, SampleRecord, records_to_frame, save_manifest


TASK_TYPE_MAP = {
    "01samoht": "controlled_vowel",
    "01samohl": "controlled_vowel",
    "02souhl": "controlled_consonant",
    "03_1sl": "controlled_word",
    "04_2sl": "controlled_word",
    "05_3sl": "controlled_word",
    "06_4sl": "controlled_word",
    "07_vsl": "controlled_word",
}

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def _build_processed_path(subject_id: str, file_name: str) -> Path:
    return CONFIG.paths.processed_dir / "audio" / CONFIG.datasets.cz_dataset_id / subject_id / Path(file_name).with_suffix(".wav")


def _task_type_from_folder(folder_name: str) -> str:
    normalized = folder_name.strip().lower()
    return TASK_TYPE_MAP.get(normalized, "")


def _iter_audio_files(raw_root: Path) -> list[Path]:
    return sorted(path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def build_manifest(raw_root: Path, output_csv: Path) -> Path:
    records: list[SampleRecord] = []
    for raw_path in _iter_audio_files(raw_root):
        if len(raw_path.parts) < len(raw_root.parts) + 4:
            continue

        relative = raw_path.relative_to(raw_root)
        label = relative.parts[0]
        subject_id = relative.parts[1]
        task_folder = relative.parts[2]
        file_name = raw_path.name
        stem = Path(file_name).stem.lower()
        task_type = _task_type_from_folder(task_folder)
        if not task_type:
            continue
        records.append(
            SampleRecord(
                sample_id=f"{CONFIG.datasets.cz_dataset_id}_{subject_id.lower()}_{stem}",
                dataset_id=CONFIG.datasets.cz_dataset_id,
                language="cs",
                subject_id=subject_id,
                session_id=None,
                label=label,
                task_type=task_type,
                speaker_role="CHI",
                split_role="train",
                eval_mode="base_manifest",
                raw_audio_path=str(raw_path),
                processed_audio_path=str(_build_processed_path(subject_id, file_name)),
                duration_s=None,
                sr_original=None,
                sr_final=CONFIG.audio.target_sr,
                source_format=raw_path.suffix.lstrip(".").lower(),
                diarization_status="not_required",
                qc_flag=None,
                notes=task_folder,
            )
        )

    frame = records_to_frame(records)
    if frame.empty:
        import pandas as pd

        frame = pd.DataFrame(columns=SAMPLE_MANIFEST_COLUMNS)
    return save_manifest(frame, output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cz_main sample manifest by scanning LANNA_Czech_Data.")
    parser.add_argument(
        "--input-dir",
        default=str(CONFIG.paths.raw_cz_dir),
        help="Root directory of the LANNA Czech corpus.",
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
        help="Output path for the new Czech sample manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_manifest(Path(args.input_dir), Path(args.output))
    print(f"Saved Czech manifest to: {output}")


if __name__ == "__main__":
    main()
