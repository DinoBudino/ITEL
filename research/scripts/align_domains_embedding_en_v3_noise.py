"""Domain alignment a NIVEL DE EMBEDDING (entrada) para el modelo V3 noise.

A diferencia de align_domains_en_v3_noise.py (que alineaba el vector de contexto justo
antes de la decision y NO movio el AUC), aqui alineamos los frames de entrada de HuBERT
ANTES de pasarlos por el LSTM + atencion + clasificador. La diferencia de dominio
(idioma, microfono, tarea) vive sobre todo en esa capa de entrada, y al propagarse por
todas las no linealidades posteriores la alineacion SI puede cambiar el ranking (AUC).

Se calcula la estadistica de los frames del checo (fuente) y del ingles (objetivo) y se
reescala cada frame del ingles para que se parezca al checo. NO se usan etiquetas del
ingles -> el test EN sigue limpio.

Metodos:
  - "standardize": media/desviacion por dimension (streaming, barato).
  - "coral":       tambien la covarianza 768x768 (CORAL clasico).

Uso:
  python research/scripts/align_domains_embedding_en_v3_noise.py --method standardize
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np, pandas as pd, torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)

from research.src.config import CONFIG
from research.src.sequence_model import AttentionBiLSTMClassifierV3


def _metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true_arr = np.array(y_true, dtype=np.float64)
    m = {"n_samples": int(len(y_true_arr)), "threshold": float(threshold),
         "accuracy": float(accuracy_score(y_true_arr, y_pred)),
         "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
         "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
         "f1": float(f1_score(y_true_arr, y_pred, zero_division=0))}
    try:
        m["auc"] = float(roc_auc_score(y_true_arr, y_prob))
    except ValueError:
        m["auc"] = None
    return m


def _threshold_sweep(y_true, y_prob):
    rows, best = [], None
    for thr in np.arange(0.05, 1.0, 0.05):
        row = _metrics(y_true, y_prob, threshold=round(float(thr), 2))
        rows.append(row)
        if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
            best = row
    return rows, best


def load_checkpoint_v3(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    model = AttentionBiLSTMClassifierV3(
        input_dim=mc["input_dim"], lstm_hidden=mc["lstm_hidden"], lstm_layers=mc["lstm_layers"],
        dropout=mc["dropout"], fc_hidden=mc["fc_hidden"], num_heads=mc["num_heads"],
        use_layer_mix=mc.get("use_layer_mix", False), num_hubert_layers=mc.get("num_hubert_layers", 12),
        embedding_noise_std=mc.get("embedding_noise_std", 0.0),
        attention_entropy_lambda=mc.get("attention_entropy_lambda", 0.0))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, mc, payload.get("best_threshold", 0.5)


def frame_statistics(manifest_df, want_cov=False, dim=768):
    """Estadistica streaming sobre TODOS los frames (sin cargar todo a memoria)."""
    n = 0
    s = np.zeros(dim, dtype=np.float64)
    ss = np.zeros(dim, dtype=np.float64)
    cov = np.zeros((dim, dim), dtype=np.float64) if want_cov else None
    for row in manifest_df.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().numpy()           # [T, D]
        mask = payload["mask"].bool().numpy()
        frames = seq[mask] if mask.shape[0] == seq.shape[0] else seq
        n += frames.shape[0]
        s += frames.sum(0)
        ss += (frames * frames).sum(0)
        if want_cov:
            cov += frames.T @ frames
    mean = s / max(n, 1)
    var = np.clip(ss / max(n, 1) - mean * mean, 1e-12, None)
    std = np.sqrt(var)
    cov_mat = None
    if want_cov:
        cov_mat = cov / max(n - 1, 1) - np.outer(mean, mean)
    return {"n": n, "mean": mean, "std": std, "cov": cov_mat}


def _sqrtm(mat, inverse, eps=1e-5):
    mat = mat + eps * np.eye(mat.shape[0])
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    d = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (v * d) @ v.T


def build_transform(src_stats, tgt_stats, method):
    """Devuelve una funcion que transforma frames del ingles -> escala del checo."""
    if method == "standardize":
        mu_s, sd_s = src_stats["mean"], src_stats["std"]
        mu_t, sd_t = tgt_stats["mean"], tgt_stats["std"]
        def _t(frames):
            return (frames - mu_t) / sd_t * sd_s + mu_s
        return _t
    # coral
    whiten = _sqrtm(tgt_stats["cov"], inverse=True)
    color = _sqrtm(src_stats["cov"], inverse=False)
    mu_s, mu_t = src_stats["mean"], tgt_stats["mean"]
    M = (whiten @ color).astype(np.float32)
    def _t(frames):
        return (frames - mu_t) @ M + mu_s
    return _t


@torch.no_grad()
def predict(model, manifest_df, device, transform=None):
    model.to(device); model.eval()
    probs, labels, sample_ids, subject_ids = [], [], [], []
    for row in manifest_df.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().numpy()
        if transform is not None:
            seq = transform(seq).astype(np.float32)
        seq_t = torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        out = model(seq_t, mask, return_context=True)
        logits = out[0]
        probs.append(float(torch.sigmoid(logits).cpu().item()))
        labels.append(float(row.label == "Patients"))
        sample_ids.append(row.sample_id); subject_ids.append(row.subject_id)
    return pd.DataFrame({"sample_id": sample_ids, "subject_id": subject_ids,
                         "label": labels, "score_patients": probs})


def main():
    p = argparse.ArgumentParser(description="Alineacion de dominio CZ->EN a nivel de embedding (V3 noise).")
    p.add_argument("--checkpoint", default=str(Path("research/checkpoints/v3_final/noise_std003/v3_noise0.03_fold2_of_10.pt")))
    p.add_argument("--source-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_train_default.csv"))
    p.add_argument("--target-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test.csv"))
    p.add_argument("--method", default="standardize", choices=["standardize", "coral"])
    p.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en"))
    p.add_argument("--report-name", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rn = args.report_name or f"domain_align_embedding_{args.method}"
    print(f"Device: {device}  |  alineacion a nivel de EMBEDDING  |  metodo: {args.method}")

    model, mc, ckpt_thr = load_checkpoint_v3(Path(args.checkpoint))
    model.to(device)
    print(f"noise_std={mc.get('embedding_noise_std')}  umbral_checo={ckpt_thr:.3f}")

    src_df = pd.read_csv(args.source_manifest)
    tgt_df = pd.read_csv(args.target_manifest)
    dim = int(mc.get("input_dim", 768))
    print(f"Fuente (CZ): {len(src_df)} muestras  |  Objetivo (EN): {len(tgt_df)} muestras")
    print(f"EN labels: {tgt_df['label'].value_counts().to_dict()}")

    want_cov = args.method == "coral"
    print("Calculando estadistica de frames (CZ)...")
    src_stats = frame_statistics(src_df, want_cov=want_cov, dim=dim)
    print(f"  frames CZ: {src_stats['n']}")
    print("Calculando estadistica de frames (EN)...")
    tgt_stats = frame_statistics(tgt_df, want_cov=want_cov, dim=dim)
    print(f"  frames EN: {tgt_stats['n']}")

    transform = build_transform(src_stats, tgt_stats, args.method)

    print("Prediciendo SIN alinear...")
    raw = predict(model, tgt_df, device, transform=None)
    print("Prediciendo CON alineacion...")
    aligned = predict(model, tgt_df, device, transform=transform)

    def _show(frame, tag):
        ckpt_m = _metrics(frame["label"], frame["score_patients"], threshold=ckpt_thr)
        sweep_rows, sweep_best = _threshold_sweep(frame["label"], frame["score_patients"])
        print(f"\n--- {tag} ---")
        print(f"  umbral checo ({ckpt_thr:.2f}): bal_acc={ckpt_m['balanced_accuracy']:.4f}  "
              f"auc={ckpt_m['auc']}  recall={ckpt_m['recall']:.4f}")
        print(f"  mejor umbral ({sweep_best['threshold']:.2f}): bal_acc={sweep_best['balanced_accuracy']:.4f}")
        return {"sample_at_czech_threshold": ckpt_m, "sample_best_sweep": sweep_best, "sample_sweep": sweep_rows,
                "scores": frame.to_dict(orient="records")}

    res_raw = _show(raw, "SIN alinear (zero-shot)")
    res_aligned = _show(aligned, f"CON alineacion embedding [{args.method}]")

    d_auc = (res_aligned["sample_at_czech_threshold"]["auc"] or 0) - (res_raw["sample_at_czech_threshold"]["auc"] or 0)
    d_ba = (res_aligned["sample_at_czech_threshold"]["balanced_accuracy"]
            - res_raw["sample_at_czech_threshold"]["balanced_accuracy"])
    print(f"\n===== EFECTO (al umbral del checo) =====")
    print(f"  delta AUC          = {d_auc:+.4f}")
    print(f"  delta balanced_acc = {d_ba:+.4f}")

    report = {"config": {"checkpoint": args.checkpoint, "method": args.method, "level": "embedding",
                         "czech_threshold": ckpt_thr, "n_source": int(len(src_df)), "n_target": int(len(tgt_df)),
                         "frames_source": src_stats["n"], "frames_target": tgt_stats["n"]},
              "note": "Alineacion sin etiquetas EN; test EN limpio.",
              "raw_zero_shot": res_raw, "domain_aligned": res_aligned,
              "delta_at_czech_threshold": {"auc": d_auc, "balanced_accuracy": d_ba}}
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{rn}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_dir / f'metrics_{rn}.json'}")


if __name__ == "__main__":
    main()
