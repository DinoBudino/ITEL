# Model lifecycle

The repository distinguishes three different deliverables:

1. An experimental configuration selected through cross-validation.
2. Out-of-fold evidence used to report research performance.
3. A frozen model bundle used by the API.

LOSO fold models are not deployment artifacts. A deployable model must be produced by a promotion command and pass manifest, checksum, shape, class-order and golden-vector checks.

## Multilingual model

`Unified Multilingual` is a supervised Czech-English domain-adaptation pipeline. It is not zero-shot and is not validated for languages or corpora other than LANNA Czech and ENNI English.

The deployed bundle contains two target-specific heads under one model ID:

- `target-cs.npz`
- `target-en.npz`

Each head contains the frozen scaler, PCA, logistic coefficients, threshold and healthy-domain centroids. Inference is subject-level and averages the six-block HuBERT representation across all submitted recordings.

Generate the bundle with:

```bash
python research/scripts/promote_multilingual_model.py
```

## Czech LayerMix

The Czech bundle contains one final non-fold checkpoint, its complete architecture configuration and a frozen subject-level threshold. Cross-validation fold checkpoints must not be promoted as if they were a final model. Train the final checkpoint with the selected hyperparameters and an out-of-fold threshold, then promote it:

```bash
python research/scripts/train_czech_final.py
python research/scripts/promote_czech_layermix.py --checkpoint artifacts/czech-layermix-final.pt
```

## Runtime states

`GET /live` reports process liveness without loading a model. `GET /ready` returns 503 until every registered bundle has all expected files and matching checksums. `GET /v1/models` reports availability per model without exposing filesystem paths.
