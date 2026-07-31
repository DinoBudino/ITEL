from __future__ import annotations

import time
import threading
from collections.abc import Callable

import numpy as np

from packages.audio_pipeline import AudioPreprocessor, AudioProfile, HubertFeatureExtractor

from .manifest import ModelManifest
from .feature_cache import FeatureCache
from .result import SubjectPrediction


class CzechLayerMixRuntime:
    def __init__(self, manifest: ModelManifest, extractor: HubertFeatureExtractor | None = None, feature_cache: FeatureCache | None = None):
        self.manifest = manifest
        self.preprocessor = AudioPreprocessor(AudioProfile(**manifest.payload["preprocessing"]))
        self.extractor = extractor or HubertFeatureExtractor(manifest.payload["encoder"]["bundle"])
        self.feature_cache = feature_cache or FeatureCache()
        self._model = None

    def warmup(self) -> None:
        self._load_model()

    def predict_subject(self, audio_files: list[bytes], language: str, progress_callback: Callable[[dict], None] | None = None, cancel_event: threading.Event | None = None) -> SubjectPrediction:
        if language != "cs":
            raise ValueError("Czech LayerMix only supports language 'cs'.")
        import torch

        started = time.perf_counter()
        model = self._load_model()
        scores = []
        cache_hits = 0
        cache_misses = 0
        sources: list[str] = []
        timings = {"cache_lookup": 0.0, "preprocessing": 0.0, "hubert": 0.0, "classification": 0.0}
        contract_id = "hubert-base-all-12-layermix-score-v1"
        for index, payload in enumerate(audio_files, start=1):
            self._check_cancel(cancel_event)
            self._emit(progress_callback, "cache_lookup", 10 + int(70 * (index - 1) / len(audio_files)), False, f"Buscando resultado del archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            cached = self.feature_cache.get(self.manifest.model_id, self.manifest.version, contract_id, payload)
            timings["cache_lookup"] += (time.perf_counter() - step) * 1000
            if cached is not None:
                scores.append(float(cached.value.reshape(-1)[0]))
                cache_hits += 1
                sources.append(cached.source)
                self._emit(progress_callback, "cache_hit", 10 + int(70 * index / len(audio_files)), False, f"Resultado reutilizado para el archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
                continue
            cache_misses += 1
            sources.append("computed")
            self._emit(progress_callback, "preprocessing", 10 + int(70 * (index - 1) / len(audio_files)), False, f"Preprocesando archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            processed = self.preprocessor.process_bytes(payload)
            timings["preprocessing"] += (time.perf_counter() - step) * 1000
            self._check_cancel(cancel_event)
            self._emit(progress_callback, "hubert", None, True, f"Extrayendo las 12 capas HuBERT del archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            layers = torch.from_numpy(self.extractor.extract_all_layers(processed)).unsqueeze(0).to(model.device)
            timings["hubert"] += (time.perf_counter() - step) * 1000
            mask = torch.ones(1, layers.size(2), dtype=torch.bool, device=model.device)
            self._emit(progress_callback, "classification", 10 + int(70 * (index - 0.25) / len(audio_files)), False, f"Aplicando LayerMix al archivo {index}/{len(audio_files)}", index, len(audio_files), started, cache_hits, cache_misses)
            step = time.perf_counter()
            with torch.inference_mode():
                score = float(torch.sigmoid(model.module(layers, mask)).item())
            timings["classification"] += (time.perf_counter() - step) * 1000
            scores.append(score)
            self.feature_cache.put(self.manifest.model_id, self.manifest.version, contract_id, payload, np.asarray([score], dtype=np.float64))
            self._emit(progress_callback, "file_complete", 10 + int(70 * index / len(audio_files)), False, f"Archivo {index}/{len(audio_files)} completado", index, len(audio_files), started, cache_hits, cache_misses)
        self._check_cancel(cancel_event)
        self._emit(progress_callback, "aggregation", 90, False, "Agregando scores del sujeto", len(audio_files), len(audio_files), started, cache_hits, cache_misses)
        patient_probability = float(np.mean(scores))
        threshold = float(self.manifest.payload["threshold"])
        label = "Patients" if patient_probability >= threshold else "Healthy"
        probabilities = {"Healthy": 1.0 - patient_probability, "Patients": patient_probability}
        timings["total"] = (time.perf_counter() - started) * 1000
        feature_source = sources[0] if len(set(sources)) == 1 else "mixed"
        return SubjectPrediction(
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
            model_digest=self.manifest.digest,
            language=language,
            file_count=len(audio_files),
            predicted_label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            threshold=threshold,
            timings_ms={key: round(value, 2) for key, value in timings.items()},
            warnings=["Research support tool; not a clinical diagnosis."],
            feature_source=feature_source,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            device=str(model.device),
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch

        from .torch_models import AttentionBiLSTMClassifierV3, LoadedTorchModel

        checkpoint_file = self.manifest.payload["checkpoint_file"]
        checkpoint = torch.load(self.manifest.artifact_dir / checkpoint_file, map_location="cpu", weights_only=True)
        config = dict(self.manifest.payload["model_config"])
        module = AttentionBiLSTMClassifierV3(**config)
        module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.eval().to(device)
        self._model = LoadedTorchModel(module=module, device=device)
        return self._model

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Prediction cancelled.")

    @staticmethod
    def _emit(callback, stage, progress, indeterminate, message, file_index, file_count, started, cache_hits, cache_misses):
        if callback is not None:
            callback({
                "type": "progress", "stage": stage, "progress": progress,
                "indeterminate": indeterminate, "message": message,
                "file_index": file_index, "file_count": file_count,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "cache_hits": cache_hits, "cache_misses": cache_misses,
            })
