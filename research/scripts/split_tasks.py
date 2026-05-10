import pandas as pd

samples = pd.read_csv("research/manifests/cz_main_cleaned.csv")
seq = pd.read_csv("research/processed_data/manifests/hubert_sequences_cleaned.csv")

print(f"Cleaned: {len(samples)} samples, {samples.task_type.value_counts().to_dict()}")
print()

for task, tag in [("controlled_word", "word"), ("controlled_consonant", "consonant")]:
    s_task = samples[samples.task_type == task].copy()
    s_task.to_csv(f"research/manifests/cz_main_cleaned_{tag}.csv", index=False)

    ids = s_task.sample_id.unique()
    seq_task = seq[seq.sample_id.isin(ids)].copy()
    seq_task.to_csv(f"research/processed_data/manifests/hubert_sequences_cleaned_{tag}.csv", index=False)

    print(f"{tag.upper()}: {len(s_task)} samples, {len(seq_task)} sequences")
    print(f"  Labels: {s_task.label.value_counts().to_dict()}")
    for s in ["H27", "H29", "H26", "H28"]:
        n = len(s_task[s_task.subject_id == s])
        print(f"  {s}: {n}")
    print(f"  Seq lengths: min={seq_task.sequence_length.min()} max={seq_task.sequence_length.max()} mean={seq_task.sequence_length.mean():.1f}")
    print()

print("Done.")
