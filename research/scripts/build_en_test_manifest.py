from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.src.config import CONFIG
from research.src.data_manifest import SAMPLE_MANIFEST_COLUMNS, SampleRecord, records_to_frame, save_manifest
from research.src.tracking import configure_tracking, log_artifact, log_dict, log_params


VALID_LABELS = ("Healthy", "Patients")
DEFAULT_EXCLUDE_PATTERNS = ("CHIDESC", "CHI_DESC")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def _include_file(path: Path, exclude_patterns: tuple[str, ...]) -> bool:
    upper_name = path.name.upper()
    return not any(pattern.upper() in upper_name for pattern in exclude_patterns)


def build_manifest(input_dir: Path, output_csv: Path, exclude_patterns: tuple[str, ...]) -> Path:
    records: list[SampleRecord] = []
    for label in VALID_LABELS:
        label_dir = input_dir / label
        if not label_dir.exists():
            continue
        for raw_path in sorted(label_dir.rglob("*")):
            if not raw_path.is_file() or raw_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue
            if not _include_file(raw_path, exclude_patterns):
                continue
            stem = raw_path.stem
            subject_id = stem.split("_")[0]
            records.append(
                SampleRecord(
                    sample_id=f"{CONFIG.datasets.en_dataset_id}_{subject_id.lower()}_{stem.lower()}",
                    dataset_id=CONFIG.datasets.en_dataset_id,
                    language="en",
                    subject_id=subject_id,
                    session_id="test",
                    label=label,
                    task_type="picture_description",
                    speaker_role="CHI",
                    split_role="eval_only",
                    eval_mode="base_manifest",
                    raw_audio_path=str(raw_path),
                    processed_audio_path=str(raw_path),
                    interviewer_audio_path=None,
                    duration_s=None,
                    sr_original=None,
                    sr_final=CONFIG.audio.target_sr,
                    source_format=raw_path.suffix.lstrip(".").lower(),
                    diarization_status="not_required",
                    review_status="approved",
                    review_reason=None,
                    auto_qc_flag=None,
                    chi_duration_s=None,
                    total_speech_s=None,
                    chi_speech_share=None,
                    n_detected_speakers=None,
                    qc_flag=None,
                    notes="english_test_child_only",
                )
            )

    frame = records_to_frame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=SAMPLE_MANIFEST_COLUMNS)
    return save_manifest(frame, output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an English test manifest from label folders, excluding CHIDESC-style files.")
    parser.add_argument(
        "--input-dir",
        default=str(CONFIG.paths.raw_en_dir / "test"),
        help="Directory containing English test subfolders Healthy/ and Patients/.",
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.manifests_dir / "english_test_samples.csv"),
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--exclude-patterns",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_PATTERNS),
        help="Filename substrings to exclude.",
    )
    parser.add_argument("--mlflow-experiment", default="itel-data-prep", help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. Defaults to a local file store under research_results/mlruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.mlflow_tracking_uri, args.mlflow_experiment)
    exclude_patterns = tuple(args.exclude_patterns)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        output = build_manifest(Path(args.input_dir), Path(args.output), exclude_patterns)
        frame = pd.read_csv(output)

        mlflow.set_tags(
            {
                "stage": "research",
                "script": "build_en_test_manifest.py",
                "dataset_id": CONFIG.datasets.en_dataset_id,
                "split": "test",
            }
        )
        log_params(
            {
                "input_dir": args.input_dir,
                "output": str(output),
                "exclude_patterns": list(exclude_patterns),
                "tracking_uri": tracking_uri,
            }
        )
        log_dict(
            "metadata/build_en_test_manifest_summary.json",
            {
                "n_rows": int(len(frame)),
                "n_subjects": 0 if frame.empty else int(frame["subject_id"].nunique()),
                "labels": {} if frame.empty else {str(k): int(v) for k, v in frame["label"].value_counts().to_dict().items()},
            },
        )
        log_artifact(output, artifact_path="manifests")

        print(f"Saved English test manifest to: {output}")
        print(f"MLflow experiment: {args.mlflow_experiment}")
        print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
