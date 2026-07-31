from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ignored, anonymized catalog for the temporary thesis demo.")
    parser.add_argument("--czech-root", default=str(ROOT_DIR / "research" / "LANNA_Czech_Data"))
    parser.add_argument("--english-root", default=str(ROOT_DIR / "research" / "English_Data"))
    parser.add_argument("--czech-manifest", default=str(ROOT_DIR / "research" / "manifests" / "cz_main_cleaned.csv"))
    parser.add_argument("--english-manifest", default=str(ROOT_DIR / "research" / "manifests" / "english_test_samples.csv"))
    parser.add_argument("--output", default=str(ROOT_DIR / "artifacts" / "demo_catalog" / ".sample_audio_catalog.json"))
    return parser.parse_args()


def build_catalog(czech_root: Path, english_root: Path, czech_manifest: Path, english_manifest: Path, output: Path) -> Path:
    specs = [
        ("cs", "CS", "czech", czech_root.resolve(), pd.read_csv(czech_manifest)),
        ("en", "EN", "english", english_root.resolve(), pd.read_csv(english_manifest)),
    ]
    subjects: list[dict] = []
    for language, prefix, path_prefix, dataset_root, frame in specs:
        required = {"subject_id", "label", "raw_audio_path", "task_type"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{language} manifest is missing columns: {sorted(missing)}")
        labels_per_subject = frame.groupby("subject_id")["label"].nunique()
        if (labels_per_subject != 1).any():
            raise ValueError(f"{language} manifest contains subjects with multiple labels.")

        for subject_index, (subject_id, rows) in enumerate(sorted(frame.groupby("subject_id"), key=lambda item: str(item[0])), start=1):
            audios: list[dict] = []
            total_duration = 0.0
            for audio_index, row in enumerate(rows.sort_values(["task_type", "sample_id"]).itertuples(index=False), start=1):
                audio_path = Path(row.raw_audio_path).resolve()
                try:
                    relative_path = audio_path.relative_to(dataset_root)
                except ValueError as exc:
                    raise ValueError(f"Audio path is outside the authorized {language} root: {audio_path}") from exc
                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                duration = float(row.duration_s) if hasattr(row, "duration_s") and pd.notna(row.duration_s) else float(sf.info(audio_path).duration)
                total_duration += duration
                audios.append({
                    "token": secrets.token_urlsafe(18),
                    "display_name": f"Grabacion {audio_index:02d}",
                    "task_type": str(row.task_type),
                    "duration_seconds": round(duration, 3),
                    "relative_path": f"{path_prefix}/{relative_path.as_posix()}",
                })
            subjects.append({
                "token": secrets.token_urlsafe(18),
                "display_id": f"{prefix}-DEMO-{subject_index:03d}",
                "language": language,
                "reference_label": str(rows.iloc[0]["label"]),
                "audio_count": len(audios),
                "task_types": sorted(rows["task_type"].astype(str).unique().tolist()),
                "total_duration_seconds": round(total_duration, 3),
                "audios": audios,
            })

    audio_count = sum(subject["audio_count"] for subject in subjects)
    if len(subjects) != 146 or audio_count != 3023:
        raise ValueError(f"Expected 146 subjects and 3023 eligible audios, got {len(subjects)} and {audio_count}.")
    payload = {"schema_version": 1, "subject_count": len(subjects), "audio_count": audio_count, "subjects": subjects}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    output = build_catalog(
        Path(args.czech_root), Path(args.english_root), Path(args.czech_manifest), Path(args.english_manifest), Path(args.output)
    )
    print(f"Demo catalog: {output}")


if __name__ == "__main__":
    main()
