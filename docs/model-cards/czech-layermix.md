# Czech V3 LayerMix

## Intended use

Subject-level research support for the Czech LANNA domain.

## Pipeline

Natural-duration audio is encoded through the 12 HuBERT Base transformer layers. ScalarMix, a bidirectional LSTM and attention produce one score per recording; scores are averaged per subject.

## Limitations

- Validated only on the LANNA Czech cohort and acquisition protocol.
- A promoted bundle requires a final non-fold checkpoint and frozen threshold.
- Performance can change with age, recording device, task and site.
- The output is not a clinical diagnosis.
