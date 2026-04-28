from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio

from .config import CONFIG


@dataclass(frozen=True)
class DiarizationResult:
    source_audio_path: str
    child_audio_path: str | None
    interviewer_audio_path: str | None
    segments_json_path: str
    diarization_status: str
    chi_speaker: str | None
    chi_strategy: str
    speaker_totals: dict[str, float]
    speaker_rms: dict[str, float]
    speaker_pitch_hz: dict[str, float]
    chi_duration_s: float
    total_speech_s: float
    chi_speech_share: float
    n_detected_speakers: int
    suspect_reasons: list[str]
    auto_qc_flag: str | None
    n_segments: int
    error_message: str | None = None


def get_hf_token() -> str:
    for env_name in CONFIG.pyannote_token_env_vars:
        token = os.getenv(env_name)
        if token:
            return token
    envs = ", ".join(CONFIG.pyannote_token_env_vars)
    raise EnvironmentError(f"Missing Hugging Face token. Set one of: {envs}")


def build_diarization_pipeline(device: str | None = None):
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Specifications

    # Torch 2.6+ tightened torch.load defaults and pyannote 3.3.x still
    # loads checkpoints that reference custom objects. We trust the upstream
    # Hugging Face checkpoints used here, so we allow those globals and force
    # weights_only=False while pyannote loads the pipeline.
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem])
    except Exception:
        pass

    original_torch_load = torch.load

    @wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    token = get_hf_token()
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        pipeline = Pipeline.from_pretrained(
            CONFIG.pyannote_model_id,
            use_auth_token=token,
            cache_dir=str(CONFIG.paths.model_dir),
        )
    finally:
        torch.load = original_torch_load
    if pipeline is None:
        raise RuntimeError(
            "pyannote returned no pipeline. Make sure the Hugging Face token is valid and "
            "that you accepted the gated model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    return pipeline.to(target_device), target_device


def _get_tracks(diarization_output: Any):
    annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
    return list(annotation.itertracks(yield_label=True))


def load_audio_mono_16k(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != CONFIG.audio.target_sr:
        waveform = torchaudio.transforms.Resample(sr, CONFIG.audio.target_sr)(waveform)
        sr = CONFIG.audio.target_sr
    return waveform, sr


def diarize_segments(audio_path: str | Path, pipeline) -> tuple[list[dict[str, float | str]], torch.Tensor, int]:
    waveform, sr = load_audio_mono_16k(audio_path)
    with torch.inference_mode():
        diarization_output = pipeline({"waveform": waveform, "sample_rate": sr})

    tracks = _get_tracks(diarization_output)
    segments = [
        {"speaker": str(speaker), "start": float(segment.start), "end": float(segment.end)}
        for segment, _, speaker in tracks
    ]
    return segments, waveform, sr


def _speaker_rms_levels(
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict[str, float | str]],
) -> dict[str, float]:
    rms_values: dict[str, list[float]] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        start = max(int(float(segment["start"]) * sr), 0)
        end = min(int(float(segment["end"]) * sr), waveform.shape[-1])
        if end <= start:
            continue
        clip = waveform[:, start:end]
        rms = float(torch.sqrt(torch.mean(clip.pow(2))).item())
        rms_values.setdefault(speaker, []).append(rms)

    return {
        speaker: float(sum(values) / len(values))
        for speaker, values in rms_values.items()
        if values
    }


def _speaker_pitch_levels(
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict[str, float | str]],
) -> dict[str, float]:
    speaker_audio: dict[str, list[np.ndarray]] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        start = max(int(float(segment["start"]) * sr), 0)
        end = min(int(float(segment["end"]) * sr), waveform.shape[-1])
        if end <= start:
            continue
        clip = waveform[:, start:end].detach().cpu().numpy().squeeze()
        if clip.size == 0:
            continue
        speaker_audio.setdefault(speaker, []).append(clip.astype(np.float32))

    pitch_levels: dict[str, float] = {}
    for speaker, clips in speaker_audio.items():
        concatenated = np.concatenate(clips)
        if concatenated.size < sr // 4:
            continue
        try:
            f0 = librosa.yin(concatenated, fmin=80, fmax=500, sr=sr)
            voiced = f0[np.isfinite(f0)]
            if voiced.size:
                pitch_levels[speaker] = float(np.median(voiced))
        except Exception:
            continue
    return pitch_levels


def _max_key(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def _min_key(scores: dict[str, float]) -> str:
    return min(scores.items(), key=lambda item: item[1])[0]


def _fallback_child_speaker(
    totals: dict[str, float],
    rms_levels: dict[str, float],
    preferred_strategy: str,
) -> tuple[str, str]:
    if rms_levels:
        return _max_key(rms_levels), f"{preferred_strategy}->loudest"
    return _max_key(totals), f"{preferred_strategy}->longest"


def assess_diarization_quality(
    *,
    chi_speaker: str,
    totals: dict[str, float],
    resolved_strategy: str,
    pitch_levels: dict[str, float],
) -> tuple[float, float, float, int, list[str], str | None]:
    chi_duration_s = float(totals.get(chi_speaker, 0.0))
    total_speech_s = float(sum(totals.values()))
    chi_speech_share = chi_duration_s / total_speech_s if total_speech_s > 0 else 0.0
    n_detected_speakers = len(totals)
    suspect_reasons: list[str] = []

    qc = CONFIG.diarization_qc
    if chi_duration_s < qc.auto_exclude_child_duration_s:
        suspect_reasons.append("child_duration_too_short")
    elif chi_duration_s < qc.review_child_duration_s:
        suspect_reasons.append("child_duration_low")

    if chi_speech_share < qc.auto_exclude_child_share:
        suspect_reasons.append("child_share_too_low")
    elif chi_speech_share < qc.review_child_share:
        suspect_reasons.append("child_share_low")

    if resolved_strategy.endswith("->loudest"):
        suspect_reasons.append("speaker_selection_used_loudness_fallback")
    if not pitch_levels:
        suspect_reasons.append("pitch_unavailable")
    if n_detected_speakers >= qc.review_many_speakers:
        suspect_reasons.append("multi_speaker_session")

    severe_reasons = {"child_duration_too_short", "child_share_too_low"}
    review_reasons = {"child_duration_low", "child_share_low"}
    auto_qc_flag = None
    if any(reason in severe_reasons for reason in suspect_reasons):
        auto_qc_flag = "exclude_suspect_child_assignment"
    elif any(reason in review_reasons for reason in suspect_reasons):
        auto_qc_flag = "review_child_assignment"

    return chi_duration_s, total_speech_s, chi_speech_share, n_detected_speakers, suspect_reasons, auto_qc_flag


def identify_child_speaker(
    segments: list[dict[str, float | str]],
    waveform: torch.Tensor,
    sr: int,
    strategy: str = "longest",
    override_speaker: str | None = None,
) -> tuple[str | None, dict[str, float], dict[str, float], dict[str, float], str]:
    totals: dict[str, float] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        totals[speaker] = totals.get(speaker, 0.0) + float(segment["end"]) - float(segment["start"])
    rms_levels = _speaker_rms_levels(waveform, sr, segments)
    pitch_levels = _speaker_pitch_levels(waveform, sr, segments)
    if not totals:
        return None, {}, {}, {}, strategy

    if override_speaker:
        if override_speaker not in totals:
            raise ValueError(f"Manual chi_speaker_override '{override_speaker}' not found in diarization speakers {sorted(totals)}")
        return override_speaker, totals, rms_levels, pitch_levels, "manual"

    if strategy == "shortest":
        chi_speaker = _min_key(totals)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    if strategy == "longest":
        chi_speaker = _max_key(totals)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    if strategy == "loudest":
        if not rms_levels:
            raise ValueError("Could not compute per-speaker RMS levels for loudest strategy.")
        chi_speaker = _max_key(rms_levels)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    if strategy == "highest_pitch":
        if not pitch_levels:
            chi_speaker, resolved = _fallback_child_speaker(totals, rms_levels, strategy)
            return chi_speaker, totals, rms_levels, pitch_levels, resolved
        chi_speaker = _max_key(pitch_levels)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    if strategy == "child_voice":
        if not pitch_levels:
            chi_speaker, resolved = _fallback_child_speaker(totals, rms_levels, strategy)
            return chi_speaker, totals, rms_levels, pitch_levels, resolved
        scores: dict[str, float] = {}
        for speaker in totals:
            pitch_score = pitch_levels.get(speaker, 0.0)
            rms_score = rms_levels.get(speaker, 0.0)
            scores[speaker] = (0.7 * pitch_score) + (0.3 * (rms_score * 1000.0))
        chi_speaker = _max_key(scores)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    if strategy == "duration_x_rms":
        scores = {
            speaker: totals.get(speaker, 0.0) * rms_levels.get(speaker, 0.0)
            for speaker in totals
        }
        chi_speaker = _max_key(scores)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy

    raise ValueError(f"Unsupported CHI speaker strategy: {strategy}")


def _concat_speaker_audio(
    *,
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict[str, float | str]],
    speaker_id: str,
) -> torch.Tensor | None:
    clips: list[torch.Tensor] = []
    for segment in segments:
        if str(segment["speaker"]) != speaker_id:
            continue
        start = max(int(float(segment["start"]) * sr), 0)
        end = min(int(float(segment["end"]) * sr), waveform.shape[-1])
        if end > start:
            clips.append(waveform[:, start:end])
    if not clips:
        return None
    return torch.cat(clips, dim=-1)


def _relative_english_audio_path(audio_path: Path) -> Path:
    return audio_path.relative_to(CONFIG.paths.raw_en_dir)


def _output_paths(audio_path: Path) -> tuple[Path, Path, Path]:
    rel = _relative_english_audio_path(audio_path)
    child_path = CONFIG.paths.english_child_dir / rel.parent / f"{audio_path.stem}_CHI.wav"
    interviewer_path = CONFIG.paths.english_interviewer_dir / rel.parent / f"{audio_path.stem}_EXA.wav"
    json_path = CONFIG.paths.diarization_dir / rel.parent / f"{audio_path.stem}_segments.json"
    return child_path, interviewer_path, json_path


def diarize_audio(
    *,
    audio_path: str | Path,
    pipeline=None,
    device: str | None = None,
    chi_strategy: str = "longest",
    chi_speaker_override: str | None = None,
) -> DiarizationResult:
    audio_path = Path(audio_path)
    child_path, interviewer_path, json_path = _output_paths(audio_path)
    child_path.parent.mkdir(parents=True, exist_ok=True)
    interviewer_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if pipeline is None:
        pipeline, _ = build_diarization_pipeline(device=device)

    try:
        segments, waveform, sr = diarize_segments(audio_path, pipeline)
        chi_speaker, totals, rms_levels, pitch_levels, resolved_strategy = identify_child_speaker(
            segments,
            waveform,
            sr,
            strategy=chi_strategy,
            override_speaker=chi_speaker_override,
        )
        if chi_speaker is None:
            raise RuntimeError("No speakers detected during diarization.")
        chi_duration_s, total_speech_s, chi_speech_share, n_detected_speakers, suspect_reasons, auto_qc_flag = assess_diarization_quality(
            chi_speaker=chi_speaker,
            totals=totals,
            resolved_strategy=resolved_strategy,
            pitch_levels=pitch_levels,
        )

        child_audio = _concat_speaker_audio(waveform=waveform, sr=sr, segments=segments, speaker_id=chi_speaker)
        if child_audio is None:
            raise RuntimeError("Could not extract CHI audio from diarization output.")

        exa_speakers = [speaker for speaker in totals if speaker != chi_speaker]
        interviewer_audio = None
        if exa_speakers:
            exa_segments = [segment for segment in segments if str(segment["speaker"]) in exa_speakers]
            interviewer_audio = _concat_speaker_audio(
                waveform=waveform,
                sr=sr,
                segments=exa_segments,
                speaker_id=exa_speakers[0],
            )
            if len(exa_speakers) > 1:
                extra = [
                    _concat_speaker_audio(waveform=waveform, sr=sr, segments=segments, speaker_id=speaker)
                    for speaker in exa_speakers[1:]
                ]
                interviewer_audio = torch.cat([tensor for tensor in [interviewer_audio, *extra] if tensor is not None], dim=-1)

        torchaudio.save(str(child_path), child_audio.cpu(), sr)
        if interviewer_audio is not None:
            torchaudio.save(str(interviewer_path), interviewer_audio.cpu(), sr)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_audio_path": str(audio_path),
                    "chi_speaker": chi_speaker,
                    "chi_strategy": resolved_strategy,
                    "speaker_totals": totals,
                    "speaker_rms": rms_levels,
                    "speaker_pitch_hz": pitch_levels,
                    "chi_duration_s": chi_duration_s,
                    "total_speech_s": total_speech_s,
                    "chi_speech_share": chi_speech_share,
                    "n_detected_speakers": n_detected_speakers,
                    "suspect_reasons": suspect_reasons,
                    "auto_qc_flag": auto_qc_flag,
                    "segments": segments,
                },
                handle,
                indent=2,
            )

        return DiarizationResult(
            source_audio_path=str(audio_path),
            child_audio_path=str(child_path),
            interviewer_audio_path=str(interviewer_path) if interviewer_audio is not None else None,
            segments_json_path=str(json_path),
            diarization_status="completed",
            chi_speaker=chi_speaker,
            chi_strategy=resolved_strategy,
            speaker_totals=totals,
            speaker_rms=rms_levels,
            speaker_pitch_hz=pitch_levels,
            chi_duration_s=chi_duration_s,
            total_speech_s=total_speech_s,
            chi_speech_share=chi_speech_share,
            n_detected_speakers=n_detected_speakers,
            suspect_reasons=suspect_reasons,
            auto_qc_flag=auto_qc_flag,
            n_segments=len(segments),
        )
    except Exception as exc:
        return DiarizationResult(
            source_audio_path=str(audio_path),
            child_audio_path=None,
            interviewer_audio_path=None,
            segments_json_path=str(json_path),
            diarization_status="failed",
            chi_speaker=None,
            chi_strategy=chi_strategy,
            speaker_totals={},
            speaker_rms={},
            speaker_pitch_hz={},
            chi_duration_s=0.0,
            total_speech_s=0.0,
            chi_speech_share=0.0,
            n_detected_speakers=0,
            suspect_reasons=[],
            auto_qc_flag="exclude_diarization_failed",
            n_segments=0,
            error_message=str(exc),
        )
