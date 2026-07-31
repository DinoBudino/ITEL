# Unified Multilingual CZ/EN

## Intended use

Research support for comparing subject-level acoustic patterns between the LANNA Czech and ENNI English domains.

## Pipeline

Natural-duration audio is normalized with the thesis RMS profile, encoded with HuBERT Base, summarized into six temporal-statistics blocks and averaged per subject. A target-specific frozen `StandardScaler -> PCA -> LogisticRegression` head produces the score.

## Limitations

- This is supervised domain adaptation, not zero-shot transfer.
- A labeled healthy reference cohort is required when adapting to a new domain.
- English LI/TD categories are not an independent clinical validation of TEL.
- The evaluated `centroid_shift` improvement over the equivalent raw pipeline was not statistically significant.
- The threshold was selected from out-of-fold evidence without an independent deployment cohort.
- The output is not a diagnosis or treatment recommendation.
