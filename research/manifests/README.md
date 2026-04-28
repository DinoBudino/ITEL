# Research manifests

This directory stores small, versioned manifests that describe datasets and
experimental inputs for the multilingual research pipeline.

Data roots used by the new flow:
- `research/LANNA_Czech_Data`
- `research/English_Data`

English diarization outputs:
- `research/processed_data/audio/english_external/child_only`
- `research/processed_data/audio/english_external/interviewer_only`
- `research/processed_data/diarization`

Keep in git:
- dataset-level sample manifests
- manifest schemas
- small split definitions

Do not keep in git:
- regenerated embedding indexes
- large segment catalogs
- model outputs
- cached audio artifacts

Current canonical manifests:
- `cz_main_samples.csv`
- `english_external_samples.csv`
- `english_external_metadata.csv`
- `bootstrap_en_metadata.py` can auto-generate `english_external_metadata.csv`
- `english_external_metadata.csv` also stores English-only diarization QC and
  manual review fields such as `review_status`, `review_reason`, and
  `auto_qc_flag`
- `sample_manifest_schema.json`
- `embedding_manifest_schema.json`
