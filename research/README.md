# Research pipeline

Clean structure for the multilingual TEL workflow.

## Folders

- `LANNA_Czech_Data/`: raw Czech LANNA corpus
- `English_Data/`: raw English corpus with `Patients/` and `Healthy/`
- `manifests/`: small versioned manifests and schemas
- `processed_data/`: generated embeddings and processed audio
- `reports/`: generated metrics and predictions
- `scripts/`: command-line entrypoints for the new workflow
- `src/`: reusable Python modules

## Primary commands

```bash
python research/scripts/build_cz_manifest.py
python research/scripts/bootstrap_en_metadata.py
python research/scripts/sync_en_diarization_qc.py
set PYANNOTE_HF_TOKEN=hf_xxx
python research/scripts/run_diarization.py
python research/scripts/build_en_manifest.py --completed-only
python research/scripts/build_embeddings.py --dataset cz_main --eval-mode train_default
python research/scripts/evaluate_internal_cz.py
python research/scripts/audit_cz_recording_confounds.py
python research/scripts/compare_cz_task_variants.py --approved-only
python research/scripts/report_crosslingual_baselines.py
python research/scripts/train_baseline.py
python research/scripts/train_baseline.py --task-variant speech_sounds_only --output-model assets/models/xlsr_logreg_speech_sounds_only.joblib
python research/scripts/train_baseline.py --task-variant speech_sounds_only --model-type mlp --output-model assets/models/xlsr_mlp_speech_sounds_only.joblib
python research/scripts/build_embeddings.py --dataset english_external --eval-mode seg_2s
python research/scripts/build_embeddings.py --dataset english_external --eval-mode seg_5s
python research/scripts/build_embeddings.py --dataset english_external --eval-mode full_audio
python research/scripts/evaluate_external.py
```

## Notes

- English diarization reads `PYANNOTE_HF_TOKEN` first, then falls back to `HF_TOKEN`.
- `run_diarization.py` processes a pilot subset by default: `10 Healthy + 10 Patients`.
- If automatic CHI/EXA assignment is wrong, set `chi_speaker_override` in
  `research/manifests/english_external_metadata.csv` and rerun only those
  files with `python research/scripts/run_diarization.py --only-overrides`.
- `run_diarization.py` now writes English QC hints back into
  `research/manifests/english_external_metadata.csv`, including
  `auto_qc_flag`, `review_status`, `review_reason`, and CHI speech-share
  metrics for manual triage.
- If you already have an older diarization summary and do not want to rerun
  pyannote yet, use `python research/scripts/sync_en_diarization_qc.py` to
  backfill those QC hints into metadata.
- You can also target a single file with repeated `--file-name` arguments.
- If the child is consistently louder than the interviewer, try
  `--chi-strategy loudest` or `--chi-strategy duration_x_rms`.
- If the child voice is recognizably younger/higher, try
  `--chi-strategy highest_pitch` or `--chi-strategy child_voice`.
- Labels are inferred automatically from `research/English_Data/Healthy` and
  `research/English_Data/Patients`.
- To exclude contaminated English files from the pilot, set `qc_flag` to a
  value starting with `exclude_` (for example `exclude_interviewer_bleed`).
  `build_en_manifest.py` will drop them by default.
- Severe auto-QC failures from diarization are also dropped automatically when
  `auto_qc_flag` starts with `exclude_`.
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
- `build_en_manifest.py --completed-only` keeps only rows whose `CHI` audio already exists.
- For quicker experiments, `build_embeddings.py` supports `--limit` and skips existing `.npy` files by default.
- `evaluate_internal_cz.py` runs subject-disjoint internal cross-validation on Czech embeddings so you can separate baseline quality from cross-lingual transfer quality.
- `audit_cz_recording_confounds.py` measures how well simple low-level recording proxies alone can separate Czech controls vs patients, to detect possible recording-condition confounds in the source corpus.
- `compare_cz_task_variants.py` compares Czech training subsets by `task_type` against both internal Czech CV and the English external test, so you can see which source-task configuration transfers best.
- `train_baseline.py` supports `--task-variant`, so you can export a dedicated cross-lingual baseline such as `speech_sounds_only` after comparing variants.
- `report_crosslingual_baselines.py` generates figures that make the `all_tasks` vs `speech_sounds_only` comparison easier to read visually.
- Czech embedding generation now uses an `anti_confound` preprocessing profile by default: silence trimming, voiced-region retention, and RMS normalization before XLSR encoding.

## Removed from the new flow

Legacy notebooks, old v5 reports, and old processed artifacts are no longer
part of the active pipeline.
