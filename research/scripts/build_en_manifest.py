from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG
from research.src.data_manifest import SAMPLE_MANIFEST_COLUMNS, SampleRecord, records_to_frame, save_manifest


VALID_LABELS = {"Healthy", "Patients"}

SUMMARY_FALLBACK_COLUMNS = [
    "diarization_status",
    "auto_qc_flag",
    "chi_duration_s",
    "total_speech_s",
    "chi_speech_share",
    "n_detected_speakers",
]


def infer_label_from_path(audio_path: Path) -> str | None:
    for part in audio_path.parts:
        if part in VALID_LABELS:
            return part
    return None


def _build_processed_path(subject_id: str, file_name: str) -> Path:
    path = Path(file_name)
    return CONFIG.paths.english_child_dir / path.parent / f"{path.stem}_CHI.wav"


def _build_interviewer_path(file_name: str) -> Path:
    path = Path(file_name)
    return CONFIG.paths.english_interviewer_dir / path.parent / f"{path.stem}_EXA.wav"


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_int(value: object) -> int | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _merge_diarization_summary(metadata: pd.DataFrame, summary_csv: Path | None) -> pd.DataFrame:
    if summary_csv is None or not summary_csv.exists():
        return metadata
    summary = pd.read_csv(summary_csv)
    if "file_name" not in summary.columns:
        return metadata

    if "chi_speech_share" not in summary.columns and {"speaker_totals", "chi_speaker"}.issubset(summary.columns):
        chi_durations = []
        total_speeches = []
        chi_shares = []
        n_speakers = []
        auto_qc_flags = []
        suspect_reasons = []
        qc = CONFIG.diarization_qc

        for row in summary.to_dict(orient="records"):
            totals_raw = row.get("speaker_totals")
            try:
                totals = ast.literal_eval(totals_raw) if isinstance(totals_raw, str) and totals_raw else {}
            except (SyntaxError, ValueError):
                totals = {}
            chi_speaker = str(row.get("chi_speaker") or "").strip()
            chi_duration_s = float(totals.get(chi_speaker, 0.0)) if chi_speaker else 0.0
            total_speech_s = float(sum(float(value) for value in totals.values()))
            chi_share = chi_duration_s / total_speech_s if total_speech_s > 0 else 0.0
            speakers = len(totals)
            reasons = []
            if chi_duration_s < qc.auto_exclude_child_duration_s:
                reasons.append("child_duration_too_short")
            elif chi_duration_s < qc.review_child_duration_s:
                reasons.append("child_duration_low")
            if chi_share < qc.auto_exclude_child_share:
                reasons.append("child_share_too_low")
            elif chi_share < qc.review_child_share:
                reasons.append("child_share_low")
            if speakers >= qc.review_many_speakers:
                reasons.append("multi_speaker_session")
            strategy = str(row.get("chi_strategy") or "")
            if strategy.endswith("->loudest"):
                reasons.append("speaker_selection_used_loudness_fallback")
            auto_qc_flag = ""
            if "child_duration_too_short" in reasons or "child_share_too_low" in reasons:
                auto_qc_flag = "exclude_suspect_child_assignment"
            elif "child_duration_low" in reasons or "child_share_low" in reasons:
                auto_qc_flag = "review_child_assignment"

            chi_durations.append(chi_duration_s)
            total_speeches.append(total_speech_s)
            chi_shares.append(chi_share)
            n_speakers.append(speakers)
            auto_qc_flags.append(auto_qc_flag)
            suspect_reasons.append("; ".join(reasons))

        summary["chi_duration_s"] = chi_durations
        summary["total_speech_s"] = total_speeches
        summary["chi_speech_share"] = chi_shares
        summary["n_detected_speakers"] = n_speakers
        summary["auto_qc_flag"] = auto_qc_flags
        summary["suspect_reasons"] = suspect_reasons

    merged = metadata.merge(summary, on="file_name", how="left", suffixes=("", "_summary"))
    for column in SUMMARY_FALLBACK_COLUMNS:
        summary_column = f"{column}_summary"
        if summary_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = None
        if column in {"diarization_status", "auto_qc_flag"}:
            merged[column] = merged[column].astype("object")
        base_values = merged[column] if column in merged.columns else pd.Series([None] * len(merged))
        if column == "diarization_status":
            base_empty = base_values.fillna("").astype(str).str.strip().isin(["", "pending"])
        else:
            base_empty = base_values.fillna("").astype(str).str.strip() == ""
        merged.loc[base_empty, column] = merged.loc[base_empty, summary_column]

    if "review_reason" in merged.columns and "suspect_reasons" in merged.columns:
        merged["review_reason"] = merged["review_reason"].astype("object")
        review_empty = merged["review_reason"].fillna("").astype(str).str.strip() == ""
        merged.loc[review_empty, "review_reason"] = merged.loc[review_empty, "suspect_reasons"]

    drop_columns = [column for column in merged.columns if column.endswith("_summary") or column == "suspect_reasons"]
    return merged.drop(columns=drop_columns, errors="ignore")


def build_manifest(audio_dir: Path, metadata_csv: Path, output_csv: Path, diarization_summary_csv: Path | None = None) -> Path:
    metadata = pd.read_csv(metadata_csv)
    metadata = _merge_diarization_summary(metadata, diarization_summary_csv)
    records: list[SampleRecord] = []
    for row in metadata.to_dict(orient="records"):
        file_name = str(row["file_name"])
        raw_path = audio_dir / file_name
        if not raw_path.exists():
            raise FileNotFoundError(f"Audio file listed in metadata does not exist: {raw_path}")

        subject_id = str(row["subject_id"])
        stem = raw_path.stem.lower()
        raw_label = str(row.get("label", "") or "").strip()
        label = raw_label or infer_label_from_path(raw_path)
        if label not in VALID_LABELS:
            raise ValueError(
                f"Missing or invalid label for '{file_name}'. Expected one of {sorted(VALID_LABELS)}. "
                f"Update {metadata_csv.name} or place the file inside a Healthy/Patients subfolder."
            )
        speaker_role = str(row.get("speaker_role", "CHI") or "CHI")
        diarization_status = str(row.get("diarization_status", "pending") or "pending")
        review_status = _optional_text(row.get("review_status")) or "pending"
        review_reason = _optional_text(row.get("review_reason"))
        auto_qc_flag = _optional_text(row.get("auto_qc_flag"))
        chi_duration_s = _optional_float(row.get("chi_duration_s"))
        total_speech_s = _optional_float(row.get("total_speech_s"))
        chi_speech_share = _optional_float(row.get("chi_speech_share"))
        n_detected_speakers = _optional_int(row.get("n_detected_speakers"))
        processed_audio_path = _build_processed_path(subject_id, file_name)
        interviewer_audio_path = _build_interviewer_path(file_name)
        if processed_audio_path.exists():
            diarization_status = "completed"
        records.append(
            SampleRecord(
                sample_id=f"{CONFIG.datasets.en_dataset_id}_{subject_id.lower()}_{stem}",
                dataset_id=CONFIG.datasets.en_dataset_id,
                language="en",
                subject_id=subject_id,
                session_id=_optional_text(row.get("session_id")),
                label=label,
                task_type="picture_description",
                speaker_role=speaker_role,
                split_role="eval_only",
                eval_mode="base_manifest",
                raw_audio_path=str(raw_path),
                processed_audio_path=str(processed_audio_path),
                interviewer_audio_path=str(interviewer_audio_path),
                duration_s=None,
                sr_original=None,
                sr_final=CONFIG.audio.target_sr,
                source_format=raw_path.suffix.lstrip(".").lower(),
                diarization_status=diarization_status,
                review_status=review_status,
                review_reason=review_reason,
                auto_qc_flag=auto_qc_flag,
                chi_duration_s=chi_duration_s,
                total_speech_s=total_speech_s,
                chi_speech_share=chi_speech_share,
                n_detected_speakers=n_detected_speakers,
                qc_flag=_optional_text(row.get("qc_flag")),
                notes=_optional_text(row.get("notes")),
            )
        )

    frame = records_to_frame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=SAMPLE_MANIFEST_COLUMNS)
    return save_manifest(frame, output_csv)


def filter_manifest_rows(
    frame: pd.DataFrame,
    *,
    completed_only: bool = False,
    approved_only: bool = False,
    include_excluded: bool = False,
) -> pd.DataFrame:
    filtered = frame.copy()
    if completed_only:
        filtered = filtered[filtered["diarization_status"] == "completed"]
    if approved_only and "review_status" in filtered.columns:
        review_values = filtered["review_status"].fillna("").astype(str).str.strip().str.lower()
        filtered = filtered.loc[review_values == "approved"].copy()
    if not include_excluded and "qc_flag" in filtered.columns:
        qc_values = filtered["qc_flag"].fillna("").astype(str).str.strip().str.lower()
        exclude_mask = qc_values.str.startswith("exclude_")
        filtered = filtered.loc[~exclude_mask].copy()
    if not include_excluded and "auto_qc_flag" in filtered.columns:
        auto_qc_values = filtered["auto_qc_flag"].fillna("").astype(str).str.strip().str.lower()
        exclude_mask = auto_qc_values.str.startswith("exclude_")
        filtered = filtered.loc[~exclude_mask].copy()
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the english_external sample manifest from metadata.")
    parser.add_argument(
        "--audio-dir",
        default=str(CONFIG.paths.raw_en_dir),
        help="Directory containing the raw English external audio files.",
    )
    parser.add_argument(
        "--metadata",
        default=str(CONFIG.paths.manifests_dir / "english_external_metadata.csv"),
        help="Metadata CSV with file_name, subject_id, label, and optional review fields.",
    )
    parser.add_argument(
        "--output",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.en_manifest_name),
        help="Output path for the English external sample manifest.",
    )
    parser.add_argument(
        "--diarization-summary",
        default=str(CONFIG.paths.diarization_dir / "english_external_diarization_summary.csv"),
        help="Optional diarization summary CSV used to backfill English QC metrics into the manifest.",
    )
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Keep only English rows whose CHI processed audio already exists.",
    )
    parser.add_argument(
        "--approved-only",
        action="store_true",
        help="Keep only English rows manually marked with review_status=approved.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep rows marked with qc_flag/auto_qc_flag starting with exclude_. By default they are dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_manifest(
        Path(args.audio_dir),
        Path(args.metadata),
        Path(args.output),
        Path(args.diarization_summary) if args.diarization_summary else None,
    )
    frame = pd.read_csv(output)
    frame = filter_manifest_rows(
        frame,
        completed_only=args.completed_only,
        approved_only=args.approved_only,
        include_excluded=args.include_excluded,
    )
    frame.to_csv(output, index=False)
    print(f"Saved English manifest to: {output}")


if __name__ == "__main__":
    main()
