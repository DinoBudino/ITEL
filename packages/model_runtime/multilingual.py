from __future__ import annotations

import time
import threading
from collections.abc import Callable

import numpy as np

from packages.audio_pipeline import AudioPreprocessor, AudioProfile, HubertFeatureExtractor, aggregate_embeddings, statistical_sequence_embedding

from .manifest import ModelManifest
from .feature_cache import FeatureCache
from .result import SubjectPrediction


ProgressCallback = Callable[[dict], None]


class MultilingualRuntime:
    def __init__(self, manifest: ModelManifest, extractor: HubertFeatureExtractor | None = None, feature_cache: FeatureCache | None = None):
        self.manifest = manifest
        preprocessing = manifest.payload["preprocessing"]
        self.preprocessor = AudioPreprocessor(AudioProfile(**preprocessing))
        self.extractor = extractor or HubertFeatureExtractor(manifest.payload["encoder"]["bundle"])
        self.feature_cache = feature_cache or FeatureCache()
        self._heads: dict[str, dict[str, np.ndarray]] = {}

    def warmup(self) -> None:
        for language in self.manifest.languages:
            self._load_head(language)

    def predict_subject(
        self,
        audio_files: list[bytes],
        language: str,
        progress_callback: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SubjectPrediction:
        if language not in self.manifest.languages:
            raise ValueError(f"Language '{language}' is not supported by {self.manifest.model_id}.")
        started = time.perf_counter()
        embeddings = []
        timings = {"cache_lookup": 0.0, "preprocessing": 0.0, "hubert": 0.0, "pooling": 0.0, "classification": 0.0}
        cache_hits = 0
        cache_misses = 0
        sources: list[str] = []
        contract_id = self.manifest.payload["feature_recipe"]["id"]
        for index, payload in enumerate(audio_files, start=1):
            self._check_cancel(cancel_event)
            self._emit(progress_callback, "cache_lookup", 10 + int(70 * (index - 1) / len(audio_files)), False, f"Buscando representación del archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            cached = self.feature_cache.get(self.manifest.model_id, self.manifest.version, contract_id, payload)
            timings["cache_lookup"] += (time.perf_counter() - step) * 1000
            if cached is not None:
                embeddings.append(cached.value)
                cache_hits += 1
                sources.append(cached.source)
                self._emit(progress_callback, "cache_hit", 10 + int(70 * index / len(audio_files)), False, f"Representación reutilizada para el archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
                continue

            cache_misses += 1
            sources.append("computed")
            self._emit(progress_callback, "preprocessing", 10 + int(70 * (index - 1) / len(audio_files)), False, f"Preprocesando archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            processed = self.preprocessor.process_bytes(payload)
            timings["preprocessing"] += (time.perf_counter() - step) * 1000
            self._check_cancel(cancel_event)
            self._emit(progress_callback, "hubert", None, True, f"Extrayendo HuBERT del archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            sequence = self.extractor.extract_last_layer(processed)
            timings["hubert"] += (time.perf_counter() - step) * 1000
            self._check_cancel(cancel_event)
            self._emit(progress_callback, "pooling", 10 + int(70 * (index - 0.25) / len(audio_files)), False, f"Resumiendo representación {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            embedding = statistical_sequence_embedding(sequence)
            timings["pooling"] += (time.perf_counter() - step) * 1000
            embeddings.append(embedding)
            self.feature_cache.put(self.manifest.model_id, self.manifest.version, contract_id, payload, embedding)
            self._emit(progress_callback, "file_complete", 10 + int(70 * index / len(audio_files)), False, f"Archivo {index}/{len(audio_files)} completado", index, len(audio_files), started, cache_hits, cache_misses)
        self._check_cancel(cancel_event)
        self._emit(progress_callback, "aggregation", 85, False, "Agregando representaciones del sujeto", len(audio_files), len(audio_files), started, cache_hits, cache_misses)
        subject_embedding = aggregate_embeddings(embeddings)
        self._emit(progress_callback, "classification", 93, False, "Aplicando scaler, PCA y clasificador", len(audio_files), len(audio_files), started, cache_hits, cache_misses)
        step = time.perf_counter()
        head = self._load_head(language)
        scaled = (subject_embedding - head["scaler_mean"]) / head["scaler_scale"]
        projected = (scaled - head["pca_mean"]) @ head["pca_components"].T
        logit = float(projected @ head["classifier_coef"].reshape(-1) + head["classifier_intercept"].reshape(-1)[0])
        patient_probability = float(
            1.0 / (1.0 + np.exp(-logit)) if logit >= 0 else np.exp(logit) / (1.0 + np.exp(logit))
        )
        threshold = float(head["threshold"].reshape(-1)[0])
        predicted_label = "Patients" if patient_probability >= threshold else "Healthy"
        probabilities = {"Healthy": 1.0 - patient_probability, "Patients": patient_probability}
        timings["classification"] = (time.perf_counter() - step) * 1000
        timings["total"] = (time.perf_counter() - started) * 1000
        feature_source = sources[0] if len(set(sources)) == 1 else "mixed"
        return SubjectPrediction(
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
            model_digest=self.manifest.digest,
            language=language,
            file_count=len(audio_files),
            predicted_label=predicted_label,
            confidence=probabilities[predicted_label],
            probabilities=probabilities,
            threshold=threshold,
            timings_ms={key: round(value, 2) for key, value in timings.items()},
            warnings=["Validated only for the Czech LANNA and English ENNI domains."],
            feature_source=feature_source,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            device=str(self.extractor.device or "not_loaded"),
        )

    def _load_head(self, language: str) -> dict[str, np.ndarray]:
        if language not in self._heads:
            file_name = self.manifest.payload["language_heads"][language]
            with np.load(self.manifest.artifact_dir / file_name) as payload:
                self._heads[language] = {key: payload[key] for key in payload.files}
        return self._heads[language]

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Prediction cancelled.")

    @staticmethod
    def _emit(callback, stage, progress, indeterminate, message, file_index, file_count, started, cache_hits, cache_misses):
        if callback is not None:
            callback({
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "indeterminate": indeterminate,
                "message": message,
                "file_index": file_index,
                "file_count": file_count,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            })
