from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from research.src.config import CONFIG
from research.src.diarization import build_diarization_pipeline, diarize_audio, load_embedding_model


REQUIRED_METADATA_COLUMNS = [
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


def _load_metadata(metadata_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_csv)
    if "file_name" not in frame.columns or "label" not in frame.columns:
        raise ValueError("english_external_metadata.csv must contain at least file_name and label columns.")
    for column in REQUIRED_METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _select_pilot_subset(frame: pd.DataFrame, per_label: int) -> pd.DataFrame:
    subsets = []
    for label in ["Healthy", "Patients"]:
        label_frame = frame[frame["label"] == label].head(per_label)
        subsets.append(label_frame)
    return pd.concat(subsets, ignore_index=True)


def _filter_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    if "chi_speaker_override" not in frame.columns:
        return frame.iloc[0:0]
    mask = frame["chi_speaker_override"].fillna("").astype(str).str.strip() != ""
    return frame.loc[mask].copy()


def _filter_by_file_names(frame: pd.DataFrame, file_names: list[str]) -> pd.DataFrame:
    wanted = [name.strip() for name in file_names if name.strip()]
    return frame.loc[frame["file_name"].isin(wanted)].copy()


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _default_review_status(auto_qc_flag: str, diarization_status: str) -> str:
    if diarization_status == "failed":
        return "needs_review"
    if auto_qc_flag.startswith("exclude_") or auto_qc_flag.startswith("review_"):
        return "needs_review"
    return "pending"


def _default_review_reason(result: dict[str, object]) -> str:
    reasons = result.get("suspect_reasons") or []
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(reason) for reason in reasons)
    error_message = _clean_text(result.get("error_message"))
    if error_message:
        return error_message
    return ""


def _update_metadata(metadata_csv: Path, metadata: pd.DataFrame, results: list[dict[str, object]]) -> None:
    result_by_file = {str(row["file_name"]): row for row in results}
    updated_rows = []
    for row in metadata.to_dict(orient="records"):
        file_name = str(row["file_name"])
        result = result_by_file.get(file_name)
        if result is None:
            updated_rows.append(row)
            continue

        diarization_status = _clean_text(result.get("diarization_status")) or _clean_text(row.get("diarization_status")) or "pending"
        auto_qc_flag = _clean_text(result.get("auto_qc_flag"))
        review_status = _clean_text(row.get("review_status")) or _default_review_status(auto_qc_flag, diarization_status)
        review_reason = _clean_text(row.get("review_reason")) or _default_review_reason(result)

        row["diarization_status"] = diarization_status
        row["auto_qc_flag"] = auto_qc_flag
        row["review_status"] = review_status
        row["review_reason"] = review_reason
        row["chi_duration_s"] = result.get("chi_duration_s", "")
        row["total_speech_s"] = result.get("total_speech_s", "")
        row["chi_speech_share"] = result.get("chi_speech_share", "")
        row["n_detected_speakers"] = result.get("n_detected_speakers", "")
        updated_rows.append(row)

    updated = pd.DataFrame(updated_rows)
    updated = updated.reindex(columns=REQUIRED_METADATA_COLUMNS)
    updated.to_csv(metadata_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pyannote diarization and export CHI/EXA audio.")
    parser.add_argument(
        "--metadata",
        default=str(CONFIG.paths.manifests_dir / "english_external_metadata.csv"),
        help="English metadata CSV with file_name and label columns.",
    )
    parser.add_argument(
        "--pilot-per-label",
        type=int,
        default=10,
        help="Number of Healthy and Patients files to process for the pilot subset.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process the full English corpus instead of the pilot subset.",
    )
    parser.add_argument(
        "--only-overrides",
        action="store_true",
        help="Process only rows that have chi_speaker_override filled in metadata.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip files that already have diarization_status=completed in metadata.",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        default=None,
        help="Process only the specified relative English file name. Can be repeated.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--chi-strategy",
        choices=["longest", "shortest", "loudest", "highest_pitch", "child_voice", "duration_x_rms", "reference"],
        default="longest",
        help="Automatic rule to assign the child speaker when no manual override is provided.",
    )
    parser.add_argument(
        "--reference-embedding",
        default=None,
        help=(
            "Path to a .npy file with the interviewer reference embedding. "
            "Required when --chi-strategy reference is used. "
            "Generate it with extract_interviewer_reference.py."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=str(CONFIG.paths.diarization_dir / "english_external_diarization_summary.csv"),
        help="Where to save the diarization summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = _load_metadata(Path(args.metadata))
    subset = metadata if args.all else _select_pilot_subset(metadata, args.pilot_per_label)
    if args.file_name:
        subset = _filter_by_file_names(subset, args.file_name)
    if args.only_overrides:
        subset = _filter_overrides(subset)
    if args.skip_completed and "diarization_status" in subset.columns:
        before = len(subset)
        subset = subset[subset["diarization_status"].fillna("").astype(str).str.strip() != "completed"].copy()
        skipped = before - len(subset)
        if skipped:
            print(f"Skipping {skipped} already-completed files.")

    if subset.empty:
        print("All files already diarized. Nothing to do.")
        return

    # Load reference embedding and embedding model when using reference strategy
    reference_embedding = None
    embedding_model = None
    if args.chi_strategy == "reference":
        if not args.reference_embedding:
            raise ValueError(
                "--reference-embedding is required when --chi-strategy reference is used. "
                "Run extract_interviewer_reference.py first."
            )
        import numpy as np
        reference_embedding = np.load(args.reference_embedding)
        print(f"Loaded interviewer reference embedding from: {args.reference_embedding}")
        embedding_model = load_embedding_model()
        print("Loaded ECAPA-TDNN embedding model.")

    pipeline, device = build_diarization_pipeline(device=args.device)
    results = []
    for row in subset.to_dict(orient="records"):
        raw_audio_path = CONFIG.paths.raw_en_dir / str(row["file_name"])
        override = row.get("chi_speaker_override")
        if override is None:
            override = None
        else:
            try:
                if bool(pd.isna(override)):
                    override = None
                else:
                    override = str(override).strip() or None
            except TypeError:
                override = str(override).strip() or None
        result = diarize_audio(
            audio_path=raw_audio_path,
            pipeline=pipeline,
            device=str(device),
            chi_strategy=args.chi_strategy,
            chi_speaker_override=override,
            reference_embedding=reference_embedding,
            embedding_model=embedding_model,
        )
        results.append(
            {
                "file_name": row["file_name"],
                "label": row["label"],
                "source_audio_path": result.source_audio_path,
                "child_audio_path": result.child_audio_path,
                "interviewer_audio_path": result.interviewer_audio_path,
                "segments_json_path": result.segments_json_path,
                "diarization_status": result.diarization_status,
                "chi_speaker": result.chi_speaker,
                "chi_strategy": result.chi_strategy,
                "n_segments": result.n_segments,
                "speaker_totals": result.speaker_totals,
                "speaker_rms": result.speaker_rms,
                "speaker_pitch_hz": result.speaker_pitch_hz,
                "chi_duration_s": result.chi_duration_s,
                "total_speech_s": result.total_speech_s,
                "chi_speech_share": result.chi_speech_share,
                "n_detected_speakers": result.n_detected_speakers,
                "suspect_reasons": result.suspect_reasons,
                "auto_qc_flag": result.auto_qc_flag,
                "error_message": result.error_message,
            }
        )
        if result.error_message:
            print(f"[{result.diarization_status}] {row['file_name']} -> CHI={result.chi_speaker} ({result.chi_strategy}) | {result.error_message}")
        else:
            print(f"[{result.diarization_status}] {row['file_name']} -> CHI={result.chi_speaker} ({result.chi_strategy})")

    summary = pd.DataFrame(results)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    _update_metadata(Path(args.metadata), metadata, results)
    print(f"Saved diarization summary to: {summary_path}")
    print(f"Updated metadata with diarization QC: {args.metadata}")
    print(f"Processed {len(summary)} English files on device={device}")


if __name__ == "__main__":
    main()
