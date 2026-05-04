# Research pipeline

Clean structure for the multilingual TEL workflow.

## Folders

- `LANNA_Czech_Data/`: raw Czech LANNA corpus
- `English_Data/`: raw English corpus with `Patients/` and `Healthy/`
- `manifests/`: small versioned manifests and schemas
- `processed_data/`: generated embeddings and processed audio
- `research_results/`: active final metrics and predictions
- `research_history/`: older exploratory reports and per-run prediction dumps
- `scripts/`: command-line entrypoints for the new workflow
- `src/`: reusable Python modules

## Current baseline

- Final cross-lingual baseline: `assets/models/xlsr_logreg_speech_sounds_only.joblib`
- Source training subset: `controlled_vowel + controlled_consonant`
- Recommended English setting: `full cleaned corpus + seg_5s`

## Key outputs to keep visible

- `research/research_results/CSV/metrics_cz_internal_cv.json`
- `research/research_results/CSV/audit_cz_recording_confounds.json`
- `research/research_results/CSV/metrics_english_external.json`

## Archived outputs

- Exploratory comparisons, per-sample prediction dumps, and secondary model-family outputs are moved under `research/research_history/reports/CSV/`.
- These files are still useful for traceability, but they are no longer part of the main active surface of the project.

## Primary commands

```bash
python research/scripts/build_cz_manifest.py
python research/scripts/bootstrap_en_metadata.py
python research/scripts/build_en_manifest.py
python research/scripts/build_embeddings.py --dataset cz_main --eval-mode train_default
python research/scripts/evaluate_internal_cz.py
python research/scripts/audit_cz_recording_confounds.py
python research/scripts/compare_cz_task_variants.py --approved-only
python research/scripts/compare_cz_model_families.py --approved-only
python research/scripts/train_baseline.py
python research/scripts/train_baseline.py --task-variant speech_sounds_only --output-model assets/models/xlsr_logreg_speech_sounds_only.joblib
python research/scripts/train_baseline.py --task-variant speech_sounds_only --model-type logreg_tuned --output-model assets/models/xlsr_logreg_tuned_speech_sounds_only.joblib
python research/scripts/train_baseline.py --task-variant speech_sounds_only --model-type linear_svm --output-model assets/models/xlsr_linearsvm_speech_sounds_only.joblib
python research/scripts/train_baseline.py --task-variant speech_sounds_only --model-type xgboost --output-model assets/models/xlsr_xgboost_speech_sounds_only.joblib
python research/scripts/train_baseline.py --task-variant speech_sounds_only --model-type mlp_small --output-model assets/models/xlsr_mlp_small_speech_sounds_only.joblib
python research/scripts/build_embeddings.py --dataset english_external --eval-mode seg_2s
python research/scripts/build_embeddings.py --dataset english_external --eval-mode seg_5s
python research/scripts/build_embeddings.py --dataset english_external --eval-mode full_audio
python research/scripts/evaluate_external.py
```

## Notes

- Labels are inferred automatically from `research/English_Data/Healthy` and
  `research/English_Data/Patients`.
- The current English external flow consumes the cleaned full audio directly;
  diarization is no longer required for routine embedding generation.
- To exclude contaminated English files, set `qc_flag` to a
  value starting with `exclude_` (for example `exclude_interviewer_bleed`).
  `build_en_manifest.py` will drop them by default.
- Use `build_en_manifest.py --approved-only` after manual review if you want a
  stricter external-test subset.
- `evaluate_external.py` now uses `median` aggregation by default for `seg_2s`
  and `seg_5s`, while keeping `mean` for `full_audio`; use
  `--aggregation-method` to override and `--approved-only` to evaluate only the
  manually approved English subset.
- English segmented embedding manifests now store per-segment QC traces such as
  `segment_qc_status`, `segment_qc_reason`, `segment_rms_mean`, and
  `segment_silence_ratio`; segmented evaluation keeps only rows with
  `segment_qc_status=passed`.
- `evaluate_external.py` also reports a simple threshold sweep so you can see
  whether `score_patients` contains usable ranking signal even when the default
  `0.5` threshold still collapses to one class.
- `build_en_manifest.py --completed-only` is now only a legacy option for old
  diarized corpora and is not needed for the cleaned raw-audio path.
- For quicker experiments, `build_embeddings.py` supports `--limit` and skips existing `.npy` files by default.
- `evaluate_internal_cz.py` runs subject-disjoint internal cross-validation on Czech embeddings so you can separate baseline quality from cross-lingual transfer quality.
- `audit_cz_recording_confounds.py` measures how well simple low-level recording proxies alone can separate Czech controls vs patients, to detect possible recording-condition confounds in the source corpus.
- `compare_cz_task_variants.py` compares Czech training subsets by `task_type` against both internal Czech CV and the English external test, so you can see which source-task configuration transfers best.
- `compare_cz_model_families.py` compares classifier families on top of the same XLSR embeddings so you can see whether a more expressive head helps or hurts transfer.
- `train_baseline.py` supports `--task-variant` and `--model-type`, so you can export dedicated cross-lingual baselines such as `speech_sounds_only + linear_svm` or `speech_sounds_only + xgboost`.
- Czech embedding generation now uses an `anti_confound` preprocessing profile by default: silence trimming, voiced-region retention, and RMS normalization before XLSR encoding.

## Removed from the new flow

Legacy notebooks, old v5 reports, and old processed artifacts are no longer
part of the active pipeline.
