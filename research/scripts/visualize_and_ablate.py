from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

from research.src.config import CONFIG
from research.src.sequence_model import AttentionBiLSTMClassifier, HubertSequenceEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Attention visualization, key-frame extraction, and causal ablation.")
    parser.add_argument(
        "--predictions",
        default=str(CONFIG.paths.reports_dir / "CSV" / "predictions_attention_bilstm_natural.csv"),
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(CONFIG.paths.manifests_dir / CONFIG.datasets.cz_manifest_name),
    )
    parser.add_argument(
        "--sequence-manifest",
        default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_natural.csv"),
    )
    parser.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "attention_viz"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _find_case(df, label_str, pred_label_val, task_type, duration_range, subject_include=None, subject_exclude=None):
    mask = (
        (df["label_str"] == label_str) &
        (df["pred_label"] == pred_label_val) &
        (df["task_type"] == task_type) &
        (df["duration_ms"] >= duration_range[0]) &
        (df["duration_ms"] < duration_range[1])
    )
    if subject_include:
        mask &= df["subject_id"].isin(subject_include)
    if subject_exclude:
        mask &= ~df["subject_id"].isin(subject_exclude)
    candidates = df[mask].sort_values("score_patients", ascending=False)
    return candidates.iloc[0] if len(candidates) > 0 else None


def _extract_f0_rms(audio_path, sr_target=16000):
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=500, sr=sr, hop_length=80, fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=80)[0]
    intensity_db = 20 * np.log10(rms + 1e-8)
    t_f0 = np.arange(len(f0)) * 80 / sr
    t_rms = np.arange(len(intensity_db)) * 80 / sr
    return y, sr, t_f0, f0, t_rms, intensity_db


def _align_to_timeline(values, src_t, tgt_t):
    if len(src_t) < 2:
        return np.full(len(tgt_t), np.nan)
    return np.interp(tgt_t, src_t, values)


def _high_res_tick_labels(n_frames, duration_ms):
    return [f"{i * duration_ms / n_frames:.0f}" for i in range(n_frames)]


def _plot_case(ax1, ax2, ax3, sample_id, label_str, pred_label_val, score, word):
    ax1.set_title(f"{sample_id}\nLabel={label_str}  Pred={pred_label_val}  Score={score:.4f}  Task={word}", fontsize=8)
    ax1.set_ylabel("Amplitude", fontsize=7)
    ax2.set_ylabel("F0 (Hz)", fontsize=7, color="tab:red")
    ax3.set_xlabel("Time (ms)", fontsize=8)
    ax3.set_ylabel("Attention", fontsize=7, color="tab:blue")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    samples = pd.read_csv(args.sample_manifest)
    seq_manifest = pd.read_csv(args.sequence_manifest)

    df = predictions.merge(samples[["sample_id", "task_type", "raw_audio_path"]], on="sample_id")
    df["label_str"] = df["label"].apply(lambda x: "Patients" if x == 1.0 else "Healthy")

    durations_ms = []
    for path in df["raw_audio_path"]:
        try:
            durations_ms.append(librosa.get_duration(path=path) * 1000)
        except Exception:
            durations_ms.append(np.nan)
    df["duration_ms"] = durations_ms

    # --- SELECT CASES ---
    case_a = _find_case(df, "Patients", "Patients", "controlled_word", (600, 5000))
    case_high_fp_subjects = [s for s in ["H27", "H29", "H28", "H26"]
                             if s in df["subject_id"].values]
    case_b = _find_case(df, "Healthy", "Patients", "controlled_word", (300, 2500),
                        subject_include=case_high_fp_subjects[:2])
    case_c = _find_case(df, "Healthy", "Patients", "controlled_vowel", (100, 2000))

    cases = {"A_TP_word": case_a, "B_FP_H27_H29_word": case_b, "C_FP_vowel": case_c}
    case_ids = []
    for tag, row in cases.items():
        if row is not None:
            case_ids.append(row["sample_id"])
            print(f"Case {tag}: {row['sample_id']} | dur={row['duration_ms']:.0f}ms | score={row['score_patients']:.4f}")
        else:
            print(f"Case {tag}: NOT FOUND")

    # --- TRAIN QUICK MODEL ON FULL DATA ---
    print("\nTraining visualization model on all data (10 epochs) ...")
    seq_df = seq_manifest[seq_manifest["sample_id"].isin(df["sample_id"].unique())].copy()
    seq_df = seq_df.merge(df[["sample_id", "label_str", "subject_id"]].drop_duplicates("sample_id"), on="sample_id")
    seq_df["label"] = seq_df["label_str"]

    from research.src.sequence_model import SequenceDataset, collate_sequences, LABEL_TO_IDX
    from torch.utils.data import DataLoader

    dataset = SequenceDataset(seq_df)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_sequences, num_workers=0)

    model = AttentionBiLSTMClassifier(input_dim=768, lstm_hidden=256, lstm_layers=2, dropout=0.3, fc_hidden=128).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.4], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(10):
        running = 0.0
        for batch in loader:
            x = batch["sequence"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, m), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        print(f"  epoch {epoch+1:02d} loss={running/max(len(loader),1):.4f}")

    model.eval()

    # --- PROCESS EACH CASE ---
    for case_tag, case_row in cases.items():
        if case_row is None:
            continue
        sample_id = case_row["sample_id"]
        audio_path = case_row["raw_audio_path"]
        true_label = case_row["label_str"]
        score_patients = case_row["score_patients"]
        task_type = case_row["task_type"]
        duration_ms = case_row["duration_ms"]

        pt_path = Path(seq_manifest.loc[seq_manifest["sample_id"] == sample_id, "sequence_path"].values[0])
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        n_frames = int(mask.sum().item())

        with torch.no_grad():
            logits, attn = model(seq, mask, return_attention=True)
        attn_np = attn[0, :n_frames].cpu().numpy()
        logit_val = logits.item()
        prob_val = float(torch.sigmoid(logits[0]).item())

        y, sr, t_f0, f0, t_rms, intensity_db = _extract_f0_rms(audio_path)
        t_audio = np.arange(len(y)) / sr * 1000
        t_hubert = np.linspace(0, duration_ms, n_frames)

        f0_on_hubert = _align_to_timeline(f0, t_f0 * 1000, t_hubert)
        intensity_on_hubert = _align_to_timeline(intensity_db, t_rms * 1000, t_hubert)

        p90 = np.percentile(attn_np, 90)
        key_mask = attn_np >= p90

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

        ax0 = axes[0]
        ax0.plot(t_audio, y, color="black", linewidth=0.5)
        ax0.set_ylabel("Waveform", fontsize=7)
        ax0.set_title(
            f"[{case_tag}]  {sample_id}  |  "
            f"Label={true_label}  Pred={'Patients' if prob_val >= 0.5 else 'Healthy'}  "
            f"Score={prob_val:.4f}  Task={task_type}  Dur={duration_ms:.0f}ms",
            fontsize=8,
        )

        ax1 = axes[1]
        ax1.plot(t_hubert, f0_on_hubert, color="tab:red", linewidth=1.0, label="F0 (Hz)")
        ax1.set_ylabel("F0 (Hz)", fontsize=7, color="tab:red")
        ax1.set_ylim(0, 500)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(t_hubert, intensity_on_hubert, color="tab:orange", linewidth=0.8, alpha=0.7, label="Intensity (dB)")
        ax1_twin.set_ylabel("Intensity (dB)", fontsize=7, color="tab:orange")

        ax2 = axes[2]
        ax2.fill_between(t_hubert, 0, attn_np, color="tab:blue", alpha=0.4, label="Attention α")
        ax2.plot(t_hubert, attn_np, color="tab:blue", linewidth=1.2)
        ax2.axhline(y=p90, color="red", linestyle="--", linewidth=1.0, label=f"P90 = {p90:.4f}")
        for k in range(n_frames):
            if key_mask[k]:
                ax2.axvspan(t_hubert[k] - (duration_ms/n_frames/2), t_hubert[k] + (duration_ms/n_frames/2),
                           alpha=0.15, color="red")
        ax2.set_ylabel("Attention α", fontsize=7, color="tab:blue")
        ax2.set_xlabel("Time (ms)", fontsize=8)
        ax2.set_xlim(0, duration_ms)
        ax2.set_ylim(0, max(1.05 * attn_np.max(), 0.001))
        ax2.legend(fontsize=6)

        plt.tight_layout()
        png_path = output_dir / f"attention_{case_tag}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {png_path}")

        # --- KEY REGIONS ---
        regions = []
        in_region = False
        start_ms = 0.0
        region_attns = []
        for k in range(n_frames):
            t_ms = float(k) / n_frames * duration_ms
            if key_mask[k]:
                if not in_region:
                    start_ms = t_ms
                    in_region = True
                    region_attns = [float(attn_np[k])]
                else:
                    region_attns.append(float(attn_np[k]))
            else:
                if in_region:
                    regions.append({
                        "start_ms": round(start_ms, 1), "end_ms": round(t_ms, 1),
                        "duration_ms": round(t_ms - start_ms, 1),
                        "mean_attention": round(float(np.mean(region_attns)), 5),
                        "max_attention": round(float(np.max(region_attns)), 5),
                        "n_frames": len(region_attns),
                    })
                    in_region = False
        if in_region:
            regions.append({
                "start_ms": round(start_ms, 1), "end_ms": round(duration_ms, 1),
                "duration_ms": round(duration_ms - start_ms, 1),
                "mean_attention": round(float(np.mean(region_attns)), 5) if region_attns else 0,
                "max_attention": round(float(np.max(region_attns)), 5) if region_attns else 0,
                "n_frames": len(region_attns),
            })

        if regions:
            regions_df = pd.DataFrame(regions)
            regions_csv = output_dir / f"key_regions_{case_tag}.csv"
            regions_df.to_csv(regions_csv, index=False)
            print(f"  Key regions: {len(regions)}  ->  {regions_csv}")
            for r in regions:
                print(f"    [{r['start_ms']:6.1f} - {r['end_ms']:6.1f}] ms  "
                      f"dur={r['duration_ms']:5.1f}ms  "
                      f"mean_attn={r['mean_attention']:.5f}  max_attn={r['max_attention']:.5f}  "
                      f"nframes={r['n_frames']}")

        # --- ABLATION (Case A only) ---
        if case_tag == "A_TP_word" and np.any(key_mask):
            print(f"\n[ABLATION] {sample_id}")
            seq_ablated = seq.clone()
            for k in range(n_frames):
                if key_mask[k]:
                    seq_ablated[0, k, :] = 0.0

            with torch.no_grad():
                logits_abl = model(seq_ablated, mask)
            prob_abl = float(torch.sigmoid(logits_abl[0]).item())

            seq_random = seq.clone()
            non_key = np.where(~key_mask)[0]
            rng = np.random.default_rng(42)
            random_indices = rng.choice(non_key, size=min(len(non_key), int(key_mask.sum())), replace=False) if len(non_key) > 0 else np.array([])
            for ri in random_indices:
                seq_random[0, ri, :] = 0.0

            with torch.no_grad():
                logits_rand = model(seq_random, mask)
            prob_rand = float(torch.sigmoid(logits_rand[0]).item())

            print(f"  Original score:       {prob_val:.5f}")
            print(f"  Ablated  (>P90):      {prob_abl:.5f}  (drop = {prob_val-prob_abl:+.5f})")
            print(f"  Ablated  (random):    {prob_rand:.5f}  (drop = {prob_val-prob_rand:+.5f})")
            print(f"  N_key_frames={int(key_mask.sum())}/{n_frames}  P90={p90:.5f}")

            ablation_txt = output_dir / f"ablation_{case_tag}.txt"
            ablation_txt.write_text(
                f"sample_id={sample_id}\n"
                f"original_prob={prob_val:.5f}\n"
                f"ablated_p90_prob={prob_abl:.5f}  delta={prob_val-prob_abl:+.5f}\n"
                f"ablated_random_prob={prob_rand:.5f}  delta={prob_val-prob_rand:+.5f}\n"
                f"n_key_frames={int(key_mask.sum())}/{n_frames}\n"
                f"p90={p90:.5f}\n",
                encoding="utf-8",
            )
            print(f"  Saved: {ablation_txt}")


if __name__ == "__main__":
    main()
