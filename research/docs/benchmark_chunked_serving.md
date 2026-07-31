# Offline Chunked Serving Benchmark

`research/scripts/benchmark_chunked_serving.py` is an experimental, offline-only
benchmark. It does not enable chunking in the API and rejects output directories
inside `models/registry`.

The sequence manifest supplies the existing precomputed full-sequence baseline.
If it does not also contain `raw_audio_path`, pass an audio manifest; rows are
joined by `sample_id`. The benchmark intentionally uses the serving recipe's
last HuBERT layer.

```powershell
python research/scripts/benchmark_chunked_serving.py `
  --sequence-manifest research/processed_data/manifests/hubert_sequences_en_test_v2.csv `
  --audio-manifest research/manifests/english_test_samples.csv `
  --sample-ids SAMPLE_1 SAMPLE_2 `
  --chunk-seconds 10 20 30 `
  --overlap-seconds 1 `
  --precisions fp32 fp16 `
  --head-path models/registry/multilingual-cz-en/1.0.0/target-en.npz
```

`fp16` requires CUDA. The default precision is `fp32`; default chunk sizes are
10, 20, and 30 seconds. Overlap is removed approximately in HuBERT frame space
before chunk sequences are concatenated and pooled with the shared statistical
recipe.

Outputs default to `research/research_results/serving_benchmarks`: per-sample
CSV, aggregate CSV, and aggregate JSON. CUDA memory is the maximum allocated
memory observed during each sample/configuration run.
