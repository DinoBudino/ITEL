from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"


@dataclass(frozen=True)
class AudioConfig:
    target_sr: int = 16000
    highpass_hz: int = 100
    train_segment_s: float = 2.0
    eval_segment_2s: float = 2.0
    eval_segment_5s: float = 5.0
    use_non_overlapping_segments: bool = True
    default_profile: str = "multilingual_light"
    trim_top_db: float = 30.0
    voiced_top_db: float = 35.0
    target_rms: float = 0.08
    max_peak_after_rms: float = 0.99
    min_segment_duration_ratio: float = 0.8
    max_segment_silence_ratio: float = 0.45
    min_segment_rms: float = 0.02
    relative_segment_rms_factor: float = 0.6


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str = "facebook/wav2vec2-xls-r-300m"
    pooling_recipe: str = "mean|std|first_quarter|last_quarter|delta|slope"
    hidden_size: int = 1024
    embedding_dim: int = 6144
    eval_modes: tuple[str, ...] = ("train_default", "seg_2s", "seg_5s", "full_audio")


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 64
    fmin: int = 50
    fmax: int = 7600
    top_db: float = 80.0
    target_frames_5s: int = 501
    normalize_per_sample: bool = True


@dataclass(frozen=True)
class DiarizationQCConfig:
    auto_exclude_child_duration_s: float = 8.0
    auto_exclude_child_share: float = 0.08
    review_child_duration_s: float = 20.0
    review_child_share: float = 0.20
    review_many_speakers: int = 3


@dataclass(frozen=True)
class PathConfig:
    manifests_dir: Path = RESEARCH_DIR / "manifests"
    processed_manifest_dir: Path = RESEARCH_DIR / "processed_data" / "manifests"
    src_dir: Path = RESEARCH_DIR / "src"
    scripts_dir: Path = RESEARCH_DIR / "scripts"
    raw_cz_dir: Path = RESEARCH_DIR / "LANNA_Czech_Data"
    raw_en_dir: Path = RESEARCH_DIR / "English_Data"
    processed_dir: Path = RESEARCH_DIR / "processed_data"
    processed_audio_dir: Path = RESEARCH_DIR / "processed_data" / "audio"
    english_child_dir: Path = RESEARCH_DIR / "processed_data" / "audio" / "english_external" / "child_only"
    english_interviewer_dir: Path = RESEARCH_DIR / "processed_data" / "audio" / "english_external" / "interviewer_only"
    diarization_dir: Path = RESEARCH_DIR / "processed_data" / "diarization"
    reports_dir: Path = RESEARCH_DIR / "research_results"
    logs_dir: Path = RESEARCH_DIR / "logs"
    model_dir: Path = ROOT_DIR / "assets" / "models"


@dataclass(frozen=True)
class DatasetConfig:
    cz_dataset_id: str = "cz_main"
    en_dataset_id: str = "english_external"
    cz_manifest_name: str = "cz_main_samples.csv"
    en_manifest_name: str = "english_external_samples.csv"


@dataclass(frozen=True)
class ResearchConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    diarization_qc: DiarizationQCConfig = field(default_factory=DiarizationQCConfig)
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    preproc_version: str = "v7_xlsr300m_multilingual_light"
    pyannote_model_id: str = "pyannote/speaker-diarization-3.1"
    pyannote_token_env_vars: tuple[str, ...] = ("PYANNOTE_HF_TOKEN", "HF_TOKEN")


CONFIG = ResearchConfig()
