from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
VALID_LABELS = {"Healthy", "Patients"}
VALID_DIARIZATION_STATUSES = {"pending", "completed", "failed", "not_required"}
METADATA_COLUMNS = [
    "file_name",
    "subject_id",
    "session_id",
    "label",
    "speaker_role",
    "chi_speaker_override",
    "diarization_status",
    "review_status",
    "review_reason",
    "auto_qc_flag",
    "chi_duration_s",
    "total_speech_s",
    "chi_speech_share",
    "n_detected_speakers",
    "qc_flag",
    "notes",
]


def infer_label(audio_path: Path) -> str:
    for part in audio_path.parts:
        if part in VALID_LABELS:
            return part
    return ""


def infer_subject_id(audio_path: Path) -> str:
    label = infer_label(audio_path)
    if label and audio_path.parent.name not in VALID_LABELS:
        return audio_path.parent.name
    return audio_path.stem


def relative_file_name(audio_dir: Path, audio_path: Path) -> str:
    return str(audio_path.relative_to(audio_dir))


def _load_existing_metadata(output_csv: Path) -> dict[str, dict[str, str]]:
    if not output_csv.exists():
        return {}
    frame = pd.read_csv(output_csv)
    if "file_name" not in frame.columns:
        return {}
    records = {}
    for row in frame.to_dict(orient="records"):
        file_name = str(row.get("file_name", "") or "").strip()
        if file_name:
            def clean(value: object) -> str:
                text = str(value or "").strip()
                return "" if text.lower() == "nan" else text

            override = clean(row.get("chi_speaker_override", ""))
            status = clean(row.get("diarization_status", ""))
            qc_flag = clean(row.get("qc_flag", ""))
            review_status = clean(row.get("review_status", ""))
            review_reason = clean(row.get("review_reason", ""))
            auto_qc_flag = clean(row.get("auto_qc_flag", ""))
            chi_duration_s = clean(row.get("chi_duration_s", ""))
            total_speech_s = clean(row.get("total_speech_s", ""))
            chi_speech_share = clean(row.get("chi_speech_share", ""))
            n_detected_speakers = clean(row.get("n_detected_speakers", ""))
            notes = clean(row.get("notes", ""))
            row["session_id"] = clean(row.get("session_id", ""))
            row["subject_id"] = clean(row.get("subject_id", ""))
            row["speaker_role"] = clean(row.get("speaker_role", ""))
            row["label"] = clean(row.get("label", ""))

            # Repair rows created before chi_speaker_override existed, where
            # the old diarization_status value shifted into the new column.
            if override in VALID_DIARIZATION_STATUSES and not status:
                row["chi_speaker_override"] = ""
                row["diarization_status"] = override
                row["review_status"] = review_status
                row["review_reason"] = review_reason
                row["auto_qc_flag"] = auto_qc_flag
                row["chi_duration_s"] = chi_duration_s
                row["total_speech_s"] = total_speech_s
                row["chi_speech_share"] = chi_speech_share
                row["n_detected_speakers"] = n_detected_speakers
                row["qc_flag"] = qc_flag if qc_flag != "auto-generated" else ""
                row["notes"] = notes or (qc_flag if qc_flag == "auto-generated" else "auto-generated")
            else:
                row["chi_speaker_override"] = override
                row["diarization_status"] = status
                row["review_status"] = review_status
                row["review_reason"] = review_reason
                row["auto_qc_flag"] = auto_qc_flag
                row["chi_duration_s"] = chi_duration_s
                row["total_speech_s"] = total_speech_s
                row["chi_speech_share"] = chi_speech_share
                row["n_detected_speakers"] = n_detected_speakers
                row["qc_flag"] = qc_flag
                row["notes"] = notes
            records[file_name] = row
    return records


def bootstrap_metadata(audio_dir: Path, output_csv: Path, default_label: str | None = None) -> Path:
    if default_label is not None and default_label not in VALID_LABELS:
        raise ValueError(f"default_label must be one of {sorted(VALID_LABELS)}, got: {default_label}")

    existing = _load_existing_metadata(output_csv)
    audio_files = sorted(
        path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    rows = []
    for audio_path in audio_files:
        file_name = relative_file_name(audio_dir, audio_path)
        previous = existing.get(file_name, {})
        inferred_label = infer_label(audio_path) or (default_label or "")
        preserved_label = str(previous.get("label", "") or "").strip()
        label = preserved_label or inferred_label
        preserved_override = str(previous.get("chi_speaker_override", "") or "").strip()
        preserved_qc = str(previous.get("qc_flag", "") or "").strip()
        preserved_review_status = str(previous.get("review_status", "") or "").strip()
        preserved_review_reason = str(previous.get("review_reason", "") or "").strip()
        preserved_auto_qc_flag = str(previous.get("auto_qc_flag", "") or "").strip()
        preserved_chi_duration_s = str(previous.get("chi_duration_s", "") or "").strip()
        preserved_total_speech_s = str(previous.get("total_speech_s", "") or "").strip()
        preserved_chi_speech_share = str(previous.get("chi_speech_share", "") or "").strip()
        preserved_n_detected_speakers = str(previous.get("n_detected_speakers", "") or "").strip()
        preserved_notes = str(previous.get("notes", "") or "").strip()
        preserved_session = str(previous.get("session_id", "") or "").strip()
        preserved_speaker_role = str(previous.get("speaker_role", "") or "").strip() or "CHI"
        preserved_status = str(previous.get("diarization_status", "") or "").strip() or "pending"
        rows.append(
            {
                "file_name": file_name,
                "subject_id": str(previous.get("subject_id", "") or "").strip() or infer_subject_id(audio_path),
                "session_id": preserved_session,
                "label": label,
                "speaker_role": preserved_speaker_role,
                "chi_speaker_override": preserved_override,
                "diarization_status": preserved_status,
                "review_status": preserved_review_status or "pending",
                "review_reason": preserved_review_reason,
                "auto_qc_flag": preserved_auto_qc_flag,
                "chi_duration_s": preserved_chi_duration_s,
                "total_speech_s": preserved_total_speech_s,
                "chi_speech_share": preserved_chi_speech_share,
                "n_detected_speakers": preserved_n_detected_speakers,
                "qc_flag": preserved_qc or ("needs_label" if not label else ""),
                "notes": preserved_notes or "auto-generated",
            }
        )

    frame = pd.DataFrame(rows).reindex(columns=METADATA_COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-generate english_external metadata from audio files.")
    parser.add_argument(
        "--audio-dir",
        default=str(CONFIG.paths.raw_en_dir),
        help="Directory containing the English external audio corpus.",
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.manifests_dir / "english_external_metadata.csv"),
        help="Where to write the generated metadata template.",
    )
    parser.add_argument(
        "--default-label",
        choices=sorted(VALID_LABELS),
        default=None,
        help="Optional label to assign when labels cannot be inferred from folder names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = bootstrap_metadata(Path(args.audio_dir), Path(args.output), default_label=args.default_label)
    print(f"Saved English metadata template to: {output}")


if __name__ == "__main__":
    main()
