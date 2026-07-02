from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.data_manifest import SAMPLE_MANIFEST_COLUMNS, SampleRecord, records_to_frame, save_manifest
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_params


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

ARTICULATORY_GROUP_MAP = {
    "02SOUHL": "consonant",
    "03_1SL": "one_syllable_items",
    "04_2SL": "two_syllable_words",
    "05_3SL": "three_syllable_words",
    "06_4SL": "four_syllable_words",
    "07_VSL": "difficult_words",
}

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def _build_processed_path(subject_id: str, file_name: str) -> Path:
    return CONFIG.paths.processed_dir / "audio" / CONFIG.datasets.cz_dataset_id / subject_id / Path(file_name).with_suffix(".wav")


def _task_type_from_folder(folder_name: str) -> str:
    normalized = folder_name.strip().lower()
    return TASK_TYPE_MAP.get(normalized, "")


def _articulatory_group_from_folder(folder_name: str) -> str | None:
    normalized = folder_name.strip().upper()
    return ARTICULATORY_GROUP_MAP.get(normalized)


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
                articulatory_group=_articulatory_group_from_folder(task_folder),
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
    parser.add_argument(
        "--mlflow-experiment",
        default="itel-data-prep",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to a local file store under research_results/mlruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        output = build_manifest(Path(args.input_dir), Path(args.output))
        frame = records_to_frame([])
        try:
            import pandas as pd

            frame = pd.read_csv(output)
        except Exception:
            pass

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "build_cz_manifest.py",
                "dataset_id": CONFIG.datasets.cz_dataset_id,
            }
        )
        log_params(
            {
                "input_dir": args.input_dir,
                "output": str(output),
                "tracking_uri": tracking_uri,
            }
        )
        log_dict(
            "metadata/build_cz_manifest_summary.json",
            {
                "n_rows": int(len(frame)),
                "n_subjects": 0 if frame.empty else int(frame["subject_id"].nunique()),
                "labels": {} if frame.empty else {str(k): int(v) for k, v in frame["label"].value_counts().to_dict().items()},
                "task_types": {} if frame.empty else {str(k): int(v) for k, v in frame["task_type"].value_counts().to_dict().items()},
            },
        )
        log_artifact(output, artifact_path="manifests")

        print(f"Saved Czech manifest to: {output}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
