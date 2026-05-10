# Research pipeline

This branch resets the research workspace toward a lightweight deep-learning path:

- `audio -> simple preprocessing -> frozen multilingual encoder -> small MLP`
- Czech (`research/LANNA_Czech_Data/`) is the first source dataset.
- English (`research/English_Data/`) stays for later transfer experiments after audio cleaning.

## Main directories

- `research/LANNA_Czech_Data/`: raw Czech audio kept as-is
- `research/English_Data/`: raw English audio kept as-is
- `research/src/`: reusable pipeline modules
- `research/scripts/`: small command-line entrypoints
- `research/manifests/`: rebuilt manifests will be written here
- `research/processed_data/`: rebuilt embeddings and derived artifacts will be written here
- `research/research_results/`: rebuilt metrics and reports will be written here

## Current model direction

- Encoder: `facebook/wav2vec2-xls-r-300m`
- Encoder usage: frozen embedding extractor
- Classifier head: compact `MLP`
- Default preprocessing: `multilingual_light`
- First evaluation target: internal Czech results only

## MLflow

- Primary experiment scripts now log to `MLflow` by default.
- Default local tracking store: `research/research_results/mlruns/`
- Main experiments:
  - `python research/scripts/build_cz_manifest.py --mlflow-experiment itel-data-prep`
  - `python research/scripts/build_embeddings.py --dataset cz_main --eval-mode train_default --mlflow-experiment itel-embeddings`
  - `python research/scripts/train_baseline.py --mlflow-experiment itel-baselines`
  - `python research/scripts/evaluate_internal_cz.py --mlflow-experiment itel-evaluations`
  - `python research/scripts/evaluate_external.py --mlflow-experiment itel-evaluations`
- You can override the tracking backend with `--mlflow-tracking-uri` in each script.

## Recommended first steps

```bash
python research/scripts/build_cz_manifest.py
python research/scripts/build_embeddings.py --dataset cz_main --eval-mode train_default
python research/scripts/train_baseline.py
python research/scripts/evaluate_internal_cz.py
```

## English minimal adaptation

```bash
python research/scripts/build_en_test_manifest.py
python research/scripts/build_embeddings.py --dataset english_external --manifest research/manifests/english_test_samples.csv --eval-mode seg_5s --output-dir research/processed_data/english_test/xlsr300m/seg_5s --output-manifest research/processed_data/english_test_embeddings_seg_5s.csv
python research/scripts/adapt_english_minimal.py
```

- `threshold_only`: keeps the Czech model fixed and tunes only the English decision threshold inside each train fold.
- `mixed_head_refit`: keeps the encoder frozen and refits only the classifier head using `Czech + English train folds`.

## Notes

- English audio is intentionally not part of the first evaluation cycle yet.
- Old generated artifacts were cleared so the Czech-first pipeline can be rebuilt cleanly.
- If you later prepare cleaned English child-only audio, reuse the frozen encoder + Bi-LSTM + attention setup for transfer checks.
- A previous CNN/logmel baseline was explored and later deprecated in favor of the HuBERT + Bi-LSTM + Attention line. Spectrogram support code (`build_spectrograms.py`, `research/src/spectrograms.py`) is kept as legacy only.
