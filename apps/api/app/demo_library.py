from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DemoAudio:
    token: str
    display_name: str
    task_type: str
    duration_seconds: float
    path: Path


@dataclass(frozen=True, slots=True)
class DemoSubject:
    token: str
    display_id: str
    language: str
    reference_label: str
    audio_count: int
    task_types: tuple[str, ...]
    total_duration_seconds: float
    audios: tuple[DemoAudio, ...]


class DemoLibrary:
    def __init__(self, catalog_path: Path, czech_root: Path, english_root: Path):
        self.catalog_path = catalog_path
        self.roots = {"czech": czech_root.resolve(), "english": english_root.resolve()}
        self.subjects: dict[str, DemoSubject] = {}
        self.audios: dict[str, DemoAudio] = {}
        self._load()

    def list_subjects(self, language: str) -> list[DemoSubject]:
        return sorted(
            (subject for subject in self.subjects.values() if subject.language == language),
            key=lambda subject: subject.display_id,
        )

    def get_subject(self, token: str) -> DemoSubject:
        try:
            return self.subjects[token]
        except KeyError as exc:
            raise KeyError("Unknown demo subject.") from exc

    def get_audio(self, token: str) -> DemoAudio:
        try:
            return self.audios[token]
        except KeyError as exc:
            raise KeyError("Unknown demo audio.") from exc

    def _load(self) -> None:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"Demo catalog not found: {self.catalog_path}")
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported demo catalog schema.")
        subjects: dict[str, DemoSubject] = {}
        audios: dict[str, DemoAudio] = {}
        display_ids: set[str] = set()
        for item in payload.get("subjects", []):
            token = str(item["token"])
            display_id = str(item["display_id"])
            language = str(item["language"])
            reference_label = str(item["reference_label"])
            if token in subjects or display_id in display_ids:
                raise ValueError("Duplicate subject token or display ID in demo catalog.")
            if language not in {"cs", "en"} or reference_label not in {"Healthy", "Patients"}:
                raise ValueError("Invalid language or reference label in demo catalog.")
            subject_audios: list[DemoAudio] = []
            for audio_item in item.get("audios", []):
                audio_token = str(audio_item["token"])
                relative = Path(str(audio_item["relative_path"]).replace("\\", "/"))
                if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
                    raise ValueError("Unsafe audio path in demo catalog.")
                root = self.roots.get(relative.parts[0])
                if root is None:
                    raise ValueError("Unknown dataset root in demo catalog.")
                path = (root / Path(*relative.parts[1:])).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise ValueError("Demo audio escapes its authorized root.") from exc
                if not path.is_file() or path.suffix.lower() != ".wav":
                    raise FileNotFoundError(path)
                if audio_token in audios:
                    raise ValueError("Duplicate audio token in demo catalog.")
                audio = DemoAudio(
                    token=audio_token,
                    display_name=str(audio_item["display_name"]),
                    task_type=str(audio_item["task_type"]),
                    duration_seconds=float(audio_item["duration_seconds"]),
                    path=path,
                )
                audios[audio_token] = audio
                subject_audios.append(audio)
            subject = DemoSubject(
                token=token,
                display_id=display_id,
                language=language,
                reference_label=reference_label,
                audio_count=len(subject_audios),
                task_types=tuple(str(value) for value in item["task_types"]),
                total_duration_seconds=float(item["total_duration_seconds"]),
                audios=tuple(subject_audios),
            )
            if subject.audio_count != int(item["audio_count"]):
                raise ValueError("Demo subject audio count does not match catalog.")
            subjects[token] = subject
            display_ids.add(display_id)
        if len(subjects) != int(payload.get("subject_count", -1)) or len(audios) != int(payload.get("audio_count", -1)):
            raise ValueError("Demo catalog totals do not match loaded entries.")
        self.subjects = subjects
        self.audios = audios
