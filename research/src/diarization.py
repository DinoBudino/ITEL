from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

# Suppress the noisy torchcodec "DLL not found" warning that torchaudio emits
# on Windows when the torchcodec backend is unavailable.  We use PyAV for
# audio loading so the missing backend has no effect on us.
warnings.filterwarnings("ignore", message=".*[Tt]orchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
logging.getLogger("torchaudio").setLevel(logging.ERROR)

import torchaudio  # noqa: E402

from .config import CONFIG


# ---------------------------------------------------------------------------
# Speaker embedding helpers (ECAPA-TDNN via SpeechBrain)
# ---------------------------------------------------------------------------

def load_embedding_model(savedir: str | Path | None = None):
    """Load pyannote wespeaker speaker embedding model (already cached from diarization).

    Uses the same pyannote/wespeaker-voxceleb-resnet34-LM model that the
    diarization pipeline already downloaded — no extra download needed and
    no symlink issues on Windows.
    """
    from pyannote.audio import Inference, Model

    token = get_hf_token()
    cache = str(savedir or CONFIG.paths.model_dir)
    model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=token,
        cache_dir=cache,
    )
    return Inference(model, window="whole")


def compute_speaker_embedding(audio_np: np.ndarray, sr: int, model) -> np.ndarray:
    """Return an L2-normalised speaker embedding for *audio_np* (mono float32).

    Uses pyannote wespeaker — volume-invariant, captures vocal tract characteristics.
    """
    if sr != CONFIG.audio.target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=CONFIG.audio.target_sr)
        sr = CONFIG.audio.target_sr

    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    embedding = model({"waveform": waveform, "sample_rate": sr})

    # pyannote Inference(window="whole") returns a (1, dim) numpy array
    vec = np.array(embedding).squeeze()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (range -1 to 1)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def extract_speaker_embeddings_from_segments(
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict[str, float | str]],
    model,
) -> dict[str, np.ndarray]:
    """Compute one ECAPA embedding per unique speaker by concatenating their segments."""
    speaker_clips: dict[str, list[np.ndarray]] = {}
    audio_np = waveform.squeeze().cpu().numpy()

    for segment in segments:
        speaker = str(segment["speaker"])
        start = max(int(float(segment["start"]) * sr), 0)
        end = min(int(float(segment["end"]) * sr), len(audio_np))
        if end > start:
            speaker_clips.setdefault(speaker, []).append(audio_np[start:end])

    embeddings: dict[str, np.ndarray] = {}
    for speaker, clips in speaker_clips.items():
        concatenated = np.concatenate(clips).astype(np.float32)
        if len(concatenated) < sr // 4:  # skip clips shorter than 0.25 s
            continue
        embeddings[speaker] = compute_speaker_embedding(concatenated, sr, model)

    return embeddings


def identify_interviewer_by_reference(
    speaker_embeddings: dict[str, np.ndarray],
    reference_embedding: np.ndarray,
    threshold: float = 0.12,
) -> tuple[str | None, dict[str, float], bool]:
    """Return the speaker label most similar to *reference_embedding*.

    Returns ``(interviewer_speaker_id, {speaker: similarity_score}, confident)``.

    *confident* is True when the best similarity exceeds *threshold*.
    When confidence is low the caller should apply a fallback strategy.
    """
    similarities = {
        speaker: cosine_similarity(embedding, reference_embedding)
        for speaker, embedding in speaker_embeddings.items()
    }
    if not similarities:
        return None, {}, False

    best_speaker = max(similarities, key=lambda k: similarities[k])
    confident = similarities[best_speaker] >= threshold
    return best_speaker, similarities, confident


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
    # Reference-strategy similarity scores: {speaker_id: cosine_similarity_to_interviewer_reference}
    # Empty dict when using non-reference strategies.
    speaker_similarities: dict[str, float] | None = None
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
            token=token,
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
    """Load audio as mono 16 kHz float32 tensor.

    Uses PyAV (``pip install av``) as the primary loader.  PyAV bundles its
    own FFmpeg codecs so MP3 files work on Windows without needing system
    FFmpeg DLLs or torchcodec.  Falls back to torchaudio for WAV/FLAC when
    PyAV is unavailable.
    """
    audio_path = Path(audio_path)

    # --- Primary: PyAV (self-contained FFmpeg, reliable on Windows for MP3) ---
    try:
        import av as _av  # noqa: PLC0415

        container = _av.open(str(audio_path))
        try:
            if not container.streams.audio:
                raise ValueError(f"No audio stream in {audio_path.name}")
            stream = container.streams.audio[0]
            sr_native: int = stream.codec_context.sample_rate
            samples: list[np.ndarray] = []

            for frame in container.decode(stream):
                arr = frame.to_ndarray()  # shape varies by format
                fmt = frame.format.name
                # Normalise to float32 in [-1, 1]
                if "s16" in fmt:
                    arr = arr.astype(np.float32) / 32_768.0
                elif "s32" in fmt:
                    arr = arr.astype(np.float32) / 2_147_483_648.0
                elif "u8" in fmt:
                    arr = (arr.astype(np.float32) - 128.0) / 128.0
                else:  # fltp, flt, dblp — already float-like
                    arr = arr.astype(np.float32)
                # Planar multi-channel: shape (channels, samples) → mono
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                samples.append(arr)
        finally:
            container.close()

        if not samples:
            raise ValueError(f"No audio decoded from {audio_path.name}")

        audio_np = np.concatenate(samples, axis=0)
        waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        sr = sr_native

    except ImportError:
        # PyAV not installed — fall back to torchaudio (works for WAV/FLAC;
        # MP3 on Windows requires system FFmpeg DLLs via torchcodec).
        print(
            f"  [warn] PyAV not installed — falling back to torchaudio for {audio_path.name}.\n"
            "         Install it with:  pip install av"
        )
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

    if sr != CONFIG.audio.target_sr:
        waveform = torchaudio.transforms.Resample(sr, CONFIG.audio.target_sr)(waveform)
        sr = CONFIG.audio.target_sr
    return waveform, sr


def diarize_segments(
    audio_path: str | Path,
    pipeline,
) -> tuple[list[dict[str, float | str]], torch.Tensor, int, dict[str, np.ndarray] | None]:
    """Run pyannote diarization and return segments, waveform, sample-rate, and speaker embeddings.

    The fourth return value is a dict ``{speaker_label: centroid_embedding}`` taken directly
    from pyannote's internal clustering step — the same wespeaker model that our reference
    embedding was extracted with.  Using these avoids a second embedding-model pass when
    applying the ``reference`` CHI-speaker strategy.  Returns ``None`` when the pipeline does
    not expose embeddings (e.g. legacy mode).
    """
    waveform, sr = load_audio_mono_16k(audio_path)
    with torch.inference_mode():
        diarization_output = pipeline({"waveform": waveform, "sample_rate": sr})

    tracks = _get_tracks(diarization_output)
    segments = [
        {"speaker": str(speaker), "start": float(segment.start), "end": float(segment.end)}
        for segment, _, speaker in tracks
    ]

    # Extract pyannote's internally computed per-speaker centroid embeddings.
    # These are (num_speakers, dim) numpy arrays in speaker_diarization.labels() order.
    speaker_embeddings: dict[str, np.ndarray] | None = None
    annotation = getattr(diarization_output, "speaker_diarization", None)
    raw_embeddings = getattr(diarization_output, "speaker_embeddings", None)
    if annotation is not None and raw_embeddings is not None:
        labels = annotation.labels()
        if len(raw_embeddings) == len(labels):
            speaker_embeddings = {}
            for i, label in enumerate(labels):
                vec = np.array(raw_embeddings[i], dtype=np.float32)
                norm = np.linalg.norm(vec)
                speaker_embeddings[label] = vec / norm if norm > 1e-8 else vec

    return segments, waveform, sr, speaker_embeddings


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

    # --- 1. CHI duration ---
    if chi_duration_s < qc.auto_exclude_child_duration_s:
        suspect_reasons.append("child_duration_too_short")
    elif chi_duration_s < qc.review_child_duration_s:
        suspect_reasons.append("child_duration_low")

    # --- 2. CHI speech share (fraction of total speech) ---
    if chi_speech_share < qc.auto_exclude_child_share:
        suspect_reasons.append("child_share_too_low")
    elif chi_speech_share < qc.review_child_share:
        suspect_reasons.append("child_share_low")
    elif chi_speech_share > qc.review_child_share_high:
        # Child speaking >90 % → likely the CHI/EXA labels are swapped
        suspect_reasons.append("child_share_suspiciously_high")

    # --- 3. Single speaker detected → cannot split CHI from EXA ---
    if qc.auto_exclude_single_speaker and n_detected_speakers <= 1:
        suspect_reasons.append("single_speaker_detected")

    # --- 4. Too many speakers → noisy environment ---
    if n_detected_speakers >= qc.review_many_speakers:
        suspect_reasons.append("multi_speaker_session")

    # --- 5. Strategy / audio quality signals ---
    if resolved_strategy.endswith("->loudest"):
        suspect_reasons.append("speaker_selection_used_loudness_fallback")
    if not pitch_levels:
        suspect_reasons.append("pitch_unavailable")

    severe_reasons = {
        "child_duration_too_short",
        "child_share_too_low",
        "single_speaker_detected",
    }
    review_reasons = {
        "child_duration_low",
        "child_share_low",
        "child_share_suspiciously_high",
        "multi_speaker_session",
    }
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
    reference_embedding: np.ndarray | None = None,
    embedding_model=None,
    precomputed_speaker_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[str | None, dict[str, float], dict[str, float], dict[str, float], str, dict[str, float]]:
    """Identify the child speaker in *segments*.

    Returns ``(chi_speaker, totals, rms_levels, pitch_levels, resolved_strategy, speaker_similarities)``.

    *speaker_similarities* maps each detected speaker to its cosine similarity against the
    interviewer reference embedding.  It is populated only when ``strategy="reference"``; for
    all other strategies it is an empty dict.

    When *precomputed_speaker_embeddings* is provided (extracted from pyannote's internal
    clustering step), it is used directly and no second embedding-model pass is performed.
    *embedding_model* is only used as a fallback when precomputed embeddings are unavailable.
    """
    totals: dict[str, float] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        totals[speaker] = totals.get(speaker, 0.0) + float(segment["end"]) - float(segment["start"])
    rms_levels = _speaker_rms_levels(waveform, sr, segments)
    pitch_levels = _speaker_pitch_levels(waveform, sr, segments)
    if not totals:
        return None, {}, {}, {}, strategy, {}

    if override_speaker:
        if override_speaker not in totals:
            raise ValueError(f"Manual chi_speaker_override '{override_speaker}' not found in diarization speakers {sorted(totals)}")
        return override_speaker, totals, rms_levels, pitch_levels, "manual", {}

    if strategy == "reference":
        if reference_embedding is None:
            raise ValueError(
                "strategy='reference' requires reference_embedding. "
                "Run extract_interviewer_reference.py first and pass --reference-embedding."
            )
        # Prefer pyannote's internally computed centroid embeddings (already done during
        # diarization — no extra model pass needed).  Fall back to computing them from
        # the raw waveform only when the pipeline didn't expose them.
        if precomputed_speaker_embeddings is not None:
            speaker_embeddings = precomputed_speaker_embeddings
            print(f"    [reference] using pyannote internal embeddings for {len(speaker_embeddings)} speakers")
        elif embedding_model is not None:
            speaker_embeddings = extract_speaker_embeddings_from_segments(waveform, sr, segments, embedding_model)
            print(f"    [reference] computed embeddings via fallback model for {len(speaker_embeddings)} speakers")
        else:
            raise ValueError(
                "strategy='reference' needs either precomputed_speaker_embeddings (from pyannote output) "
                "or embedding_model as fallback. Pass --reference-embedding."
            )
        interviewer_speaker, similarities, confident = identify_interviewer_by_reference(
            speaker_embeddings, reference_embedding
        )

        if interviewer_speaker is None or not confident:
            # Low confidence — fall back to longest (child always speaks more in clinical assessments)
            chi_speaker = _max_key(totals)
            best_sim = similarities.get(interviewer_speaker, 0.0) if similarities else 0.0
            print(
                f"    [low-confidence sim={best_sim:.3f}] falling back to 'longest' → CHI={chi_speaker}"
            )
            return chi_speaker, totals, rms_levels, pitch_levels, f"reference->longest_fallback(sim={best_sim:.2f})", similarities

        # child is whichever speaker is NOT the interviewer (lowest similarity)
        chi_speaker = min(similarities, key=lambda k: similarities[k])
        best_sim = similarities[interviewer_speaker]
        resolved = f"reference(sim={best_sim:.2f})"
        return chi_speaker, totals, rms_levels, pitch_levels, resolved, similarities

    if strategy == "shortest":
        chi_speaker = _min_key(totals)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

    if strategy == "longest":
        chi_speaker = _max_key(totals)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

    if strategy == "loudest":
        if not rms_levels:
            raise ValueError("Could not compute per-speaker RMS levels for loudest strategy.")
        chi_speaker = _max_key(rms_levels)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

    if strategy == "highest_pitch":
        if not pitch_levels:
            chi_speaker, resolved = _fallback_child_speaker(totals, rms_levels, strategy)
            return chi_speaker, totals, rms_levels, pitch_levels, resolved, {}
        chi_speaker = _max_key(pitch_levels)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

    if strategy == "child_voice":
        if not pitch_levels:
            chi_speaker, resolved = _fallback_child_speaker(totals, rms_levels, strategy)
            return chi_speaker, totals, rms_levels, pitch_levels, resolved, {}
        scores: dict[str, float] = {}
        for speaker in totals:
            pitch_score = pitch_levels.get(speaker, 0.0)
            rms_score = rms_levels.get(speaker, 0.0)
            scores[speaker] = (0.7 * pitch_score) + (0.3 * (rms_score * 1000.0))
        chi_speaker = _max_key(scores)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

    if strategy == "duration_x_rms":
        scores = {
            speaker: totals.get(speaker, 0.0) * rms_levels.get(speaker, 0.0)
            for speaker in totals
        }
        chi_speaker = _max_key(scores)
        return chi_speaker, totals, rms_levels, pitch_levels, strategy, {}

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


def _build_clean_chi_audio(
    *,
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict[str, float | str]],
    chi_speaker: str,
    trim_s: float = 0.15,
    min_seg_s: float = 0.5,
    reference_embedding: np.ndarray | None = None,
    embedding_model=None,
    sim_threshold: float = 0.5,
) -> torch.Tensor | None:
    """Concatenate CHI segments applying three quality filters:

    1. Boundary trim    — remove *trim_s* seconds from each segment edge to
                          avoid cross-talk bleed at speaker transitions.
    2. Min-duration     — discard segments shorter than *min_seg_s* after
                          trimming; these are typically diarization errors.
    3. Similarity check — if *reference_embedding* and *embedding_model* are
                          provided, compute per-segment wespeaker embeddings and
                          reject any CHI segment whose cosine similarity to the
                          interviewer reference exceeds *sim_threshold* (likely
                          a mis-labelled interviewer segment).
    """
    clips: list[torch.Tensor] = []
    n_short = 0
    n_similar = 0

    for seg in segments:
        if str(seg["speaker"]) != chi_speaker:
            continue

        # --- 1. Boundary trim ---
        start_s = float(seg["start"]) + trim_s
        end_s = float(seg["end"]) - trim_s

        # --- 2. Min-duration filter ---
        if end_s - start_s < min_seg_s:
            n_short += 1
            continue

        start_i = max(int(start_s * sr), 0)
        end_i = min(int(end_s * sr), waveform.shape[-1])
        if end_i <= start_i:
            n_short += 1
            continue

        clip = waveform[:, start_i:end_i]

        # --- 3. Per-segment similarity check ---
        if reference_embedding is not None and embedding_model is not None:
            clip_np = clip.squeeze().cpu().numpy()
            if len(clip_np) >= sr // 4:
                try:
                    seg_emb = compute_speaker_embedding(clip_np, sr, embedding_model)
                    sim = cosine_similarity(seg_emb, reference_embedding)
                    if sim > sim_threshold:
                        n_similar += 1
                        continue  # sounds too much like the interviewer — discard
                except Exception:
                    pass  # embedding failed; keep the clip to be safe

        clips.append(clip)

    rejected = n_short + n_similar
    if rejected:
        print(
            f"    [clean_chi] rechazados {rejected} segmentos "
            f"({n_short} cortos, {n_similar} similares al investigador)"
        )

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
    reference_embedding: np.ndarray | None = None,
    embedding_model=None,
) -> DiarizationResult:
    audio_path = Path(audio_path)
    child_path, interviewer_path, json_path = _output_paths(audio_path)
    child_path.parent.mkdir(parents=True, exist_ok=True)
    interviewer_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if pipeline is None:
        pipeline, _ = build_diarization_pipeline(device=device)

    try:
        segments, waveform, sr, precomputed_embeddings = diarize_segments(audio_path, pipeline)
        chi_speaker, totals, rms_levels, pitch_levels, resolved_strategy, speaker_similarities = identify_child_speaker(
            segments,
            waveform,
            sr,
            strategy=chi_strategy,
            override_speaker=chi_speaker_override,
            reference_embedding=reference_embedding,
            embedding_model=embedding_model,
            precomputed_speaker_embeddings=precomputed_embeddings,
        )
        if chi_speaker is None:
            raise RuntimeError("No speakers detected during diarization.")
        chi_duration_s, total_speech_s, chi_speech_share, n_detected_speakers, suspect_reasons, auto_qc_flag = assess_diarization_quality(
            chi_speaker=chi_speaker,
            totals=totals,
            resolved_strategy=resolved_strategy,
            pitch_levels=pitch_levels,
        )

        child_audio = _build_clean_chi_audio(
            waveform=waveform,
            sr=sr,
            segments=segments,
            chi_speaker=chi_speaker,
            reference_embedding=reference_embedding,
            embedding_model=embedding_model,
        )
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

        import soundfile as _sf
        _sf.write(str(child_path), child_audio.squeeze().cpu().numpy(), sr)
        if interviewer_audio is not None:
            _sf.write(str(interviewer_path), interviewer_audio.squeeze().cpu().numpy(), sr)

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_audio_path": str(audio_path),
                    "chi_speaker": chi_speaker,
                    "chi_strategy": resolved_strategy,
                    "speaker_totals": totals,
                    "speaker_rms": rms_levels,
                    "speaker_pitch_hz": pitch_levels,
                    "speaker_similarities": {k: round(v, 4) for k, v in speaker_similarities.items()},
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
            speaker_similarities=speaker_similarities,
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
