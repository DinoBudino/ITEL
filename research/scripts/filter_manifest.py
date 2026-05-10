from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import librosa
import numpy as np
import pandas as pd

manifest = pd.read_csv("research/manifests/cz_main_samples.csv")

print("Computing durations ...")
durs = []
for p in manifest["raw_audio_path"]:
    try:
        durs.append(librosa.get_duration(path=p))
    except Exception:
        durs.append(np.nan)
manifest["duration_s"] = durs

before = len(manifest)
print(f"\nBefore: {before} samples")
print(f"  controlled_vowel: {(manifest.task_type=='controlled_vowel').sum()}")
print(f"  <0.3s:            {(manifest.duration_s < 0.3).sum()}")

cleaned = manifest[
    (manifest.task_type != "controlled_vowel") &
    (manifest.duration_s >= 0.3)
].copy()

after = len(cleaned)
removed = before - after
print(f"\nAfter:  {after} samples  (removed {removed}, {100*removed/before:.1f}%)")
print(f"Label:  {cleaned.label.value_counts().to_dict()}")
print(f"Tasks:  {cleaned.task_type.value_counts().to_dict()}")

for s in ["H27", "H29", "H26", "H28"]:
    n = len(cleaned[cleaned.subject_id == s])
    print(f"  {s}: {n} samples remain")

out_path = "research/manifests/cz_main_cleaned.csv"
cols = [c for c in manifest.columns if c in cleaned.columns]
cleaned[cols].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
