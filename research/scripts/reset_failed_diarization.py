"""Reset failed/pending Patients rows so they get reprocessed by diarization."""
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
csv = ROOT_DIR / "research" / "manifests" / "english_external_metadata.csv"

df = pd.read_csv(csv)
mask = df.index >= 0  # all rows
n = int(mask.sum())
print(f"Resetting {n} rows to pending: {df.loc[mask, 'file_name'].tolist()}")
df.loc[mask, "diarization_status"] = "pending"
df.to_csv(csv, index=False)
print("Done.")
