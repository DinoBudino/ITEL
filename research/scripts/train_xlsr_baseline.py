"""Fix Czech manifest paths and train the XLSR logistic-regression baseline.

Steps:
  1. Patch raw_audio_path in cz_main_samples.csv to point at research/Datos_Brutos/
  2. Extract XLSR embeddings for Czech data (train_default mode)
  3. Train LogisticRegression baseline and save to assets/models/xlsr_logreg_baseline.joblib

Usage:
    python research/scripts/train_xlsr_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from research.src.config import CONFIG


# ---------------------------------------------------------------------------
# Step 1: Patch the Czech manifest paths
# ---------------------------------------------------------------------------

OLD_PREFIX = r"C:\Users\Zahir\Desktop\ITEL\research\LANNA_Czech_Data\\"
NEW_PREFIX = str(ROOT_DIR / "research" / "Datos_Brutos") + "\\"

MANIFEST_PATH = CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name


def patch_manifest() -> Path:
    """Replace old raw_audio_path prefix with current machine's path."""
    df = pd.read_csv(MANIFEST_PATH)

    needs_patch = df["raw_audio_path"].str.startswith(OLD_PREFIX)
    n_needs = int(needs_patch.sum())

    if n_needs == 0:
        print("  Paths already up to date — no patching needed.")
        return MANIFEST_PATH

    df.loc[needs_patch, "raw_audio_path"] = (
        df.loc[needs_patch, "raw_audio_path"]
        .str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)
    )

    # Verify a few
    sample = df.loc[needs_patch, "raw_audio_path"].iloc[0]
    if not Path(sample).exists():
        raise FileNotFoundError(
            f"Patched path does not exist: {sample}\n"
            "Check that research/Datos_Brutos/ contains the Czech WAV files."
        )

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"  Patched {n_needs} raw_audio_path entries → research/Datos_Brutos/")
    return MANIFEST_PATH


# ---------------------------------------------------------------------------
# Step 2: Build Czech XLSR embeddings (train_default mode)
# ---------------------------------------------------------------------------

def build_czech_embeddings() -> Path:
    import subprocess
    embedding_manifest = (
        CONFIG.paths.processed_manifest_dir
        / f"{CONFIG.datasets.cz_dataset_id}_embeddings_train_default.csv"
    )
    if embedding_manifest.exists():
        n = len(pd.read_csv(embedding_manifest))
        print(f"  Czech embeddings already exist ({n} rows). Skipping extraction.")
        return embedding_manifest

    cmd = [
        sys.executable,
        str(ROOT_DIR / "research" / "scripts" / "build_embeddings.py"),
        "--dataset", CONFIG.datasets.cz_dataset_id,
        "--eval-mode", "train_default",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Embedding extraction failed.")
        sys.exit(result.returncode)
    return embedding_manifest


# ---------------------------------------------------------------------------
# Step 3: Train LogisticRegression baseline
# ---------------------------------------------------------------------------

def load_embeddings(embedding_manifest: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .npy embeddings listed in the manifest and return X, y, labels."""
    df = pd.read_csv(embedding_manifest)

    # Merge with sample manifest to get labels
    sample_df = pd.read_csv(MANIFEST_PATH)
    merged = df.merge(
        sample_df[["sample_id", "label"]],
        on="sample_id",
        how="left",
    ).dropna(subset=["label", "embedding_path"])

    X_list, y_list = [], []
    skipped = 0
    for row in merged.itertuples(index=False):
        p = Path(row.embedding_path)
        if not p.exists():
            skipped += 1
            continue
        vec = np.load(str(p))
        X_list.append(vec.flatten())
        y_list.append(row.label)

    if skipped:
        print(f"  Skipped {skipped} missing .npy files")

    return np.array(X_list, dtype=np.float32), np.array(y_list), list(merged["label"].unique())


def train_baseline(embedding_manifest: Path) -> Path:
    print("  Loading embeddings…")
    X, y, classes = load_embeddings(embedding_manifest)
    print(f"  X shape: {X.shape}  classes: {sorted(set(y))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
    )
    clf.fit(X_scaled, y)

    # Wrap scaler + classifier in a simple pipeline-compatible object
    from sklearn.pipeline import Pipeline
    model = Pipeline([("scaler", scaler), ("clf", clf)])
    # Re-fit on scaled data using pipeline
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
    ))])
    model.fit(X, y)

    out_path = CONFIG.paths.model_dir / "xlsr_logreg_baseline.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(out_path))

    train_acc = model.score(X, y)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Saved model to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 60

    print(f"\n{sep}")
    print("STEP 1/3  Patching Czech manifest paths")
    print(sep)
    patch_manifest()

    print(f"\n{sep}")
    print("STEP 2/3  Building Czech XLSR embeddings (train_default)")
    print(sep)
    embedding_manifest = build_czech_embeddings()

    print(f"\n{sep}")
    print("STEP 3/3  Training LogisticRegression baseline")
    print(sep)
    model_path = train_baseline(embedding_manifest)

    print(f"\n{sep}")
    print(f"✓ Done. Model saved to: {model_path}")
    print(f"  Now run: python research/scripts/evaluate_external.py")
    print(sep)


if __name__ == "__main__":
    main()
