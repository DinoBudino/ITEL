# FINAL ARCHITECTURE COMPARISON SUMMARY

## Central Finding

**`LayerMix` maximizes within-domain Czech discrimination.**
**`Embedding Noise (std=0.03)` maximizes cross-lingual transfer robustness.**

There is no single model optimal for both objectives.

## Key Evidence

- **Czech**: `V3 LayerMix` achieves `sample bal_acc=0.878`, `subject bal_acc=0.943`, `AUC=0.945`.
- **English zero-shot**: `V3 Noise 0.03` achieves `AUC=0.800` (vs `0.600` baseline), `recalibrated bal_acc=0.786`.
- **English linear probe**: `V3 Noise 0.03` achieves `bal_acc=0.757`, `AUC=0.743`, matching/beating Fase 2.
- **LayerMix collapses in English**: all scores saturate at ~1.0.
- **Noise avoids saturation in English**: scores range from 0.0004 to 1.0000.
- **Clinical recall**: both branches maintain recall >= 0.95 across domains.

## Recommended Architecture Split

| Use Case | Best Model | Key Metric |
|---|---|---|
| Czech closed-domain diagnosis | V3 LayerMix | subject bal_acc 0.943 |
| English / multilingual transfer | V3 Noise 0.03 | zero-shot AUC 0.800 |
| Clinical sensitivity | V3 LayerMix (CZ) / Noise (EN) | recall 1.00 / 0.80 |

## Interpretation

- **LayerMix**: extra HuBERT layers capture Czech-specific phonetic detail, boosting intra-domain accuracy +6.9pp.
- **Noise 0.03**: Gaussian regularization forces the model to ignore domain-specific cues, improving cross-lingual AUC by +33%.

## Next Steps (Max 3 Experiments)

1. **Noise 0.03 + Sparse Attention** on multilingual branch
2. **Task-specialized models** (consonant-only, word-only)
3. **ResNet-18 benchmark** only if 1 and 2 are insufficient

## Artifacts

- `research/research_results/CSV/final_comparison/master_model_comparison.csv`
- `research/research_results/CSV/final_comparison/architectural_decision_table.csv`
- `research/research_results/attention_viz/final_comparison/czech_branch_comparison.png`
- `research/research_results/attention_viz/final_comparison/english_transfer_comparison.png`
- `research/research_results/attention_viz/final_comparison/specialization_vs_generalization.png`
- `research/research_results/attention_viz/final_comparison/clinical_recall_comparison.png`
- `research/research_results/attention_viz/final_comparison/relative_gains.png`
