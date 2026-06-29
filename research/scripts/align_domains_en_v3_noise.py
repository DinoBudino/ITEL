"""Domain alignment (Czech -> English) for the V3 noise model.

Idea: el modelo se entrena SOLO en checo. Al aplicarlo a ingles, los vectores de
contexto (la representacion interna antes de decidir) estan "desplazados" porque el
ingles suena distinto al checo por razones ajenas a la enfermedad (microfono, idioma,
tarea). Este script ALINEA la distribucion del ingles a la del checo usando solo
estadisticas SIN ETIQUETAS (medias, desviaciones, covarianzas), y vuelve a aplicar la
cabeza clasificadora ya entrenada. Asi el umbral del checo deberia funcionar mejor en
ingles, reduciendo la necesidad de recalibrar por idioma.

IMPORTANTE: no se usa ninguna etiqueta del ingles en la alineacion -> el ingles sigue
siendo un test 100% limpio.

Metodos:
  - "standardize": empareja media y desviacion por dimension (rapido, robusto).
  - "coral":       empareja tambien la covarianza (CORAL clasico).

Uso (en tu .venv con torch):
  python research/scripts/align_domains_en_v3_noise.py --method standardize
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


# --------------------------------------------------------------------------- #
#  Metricas
# --------------------------------------------------------------------------- #
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


def _subject_metrics(frame, threshold):
    subj = frame.groupby(["subject_id", "label"], as_index=False)["score_patients"].mean()
    subj.rename(columns={"score_patients": "score_patients_subject"}, inplace=True)
    return _metrics(subj["label"], subj["score_patients_subject"], threshold=threshold), subj


# --------------------------------------------------------------------------- #
#  Modelo
# --------------------------------------------------------------------------- #
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


@torch.no_grad()
def extract_contexts(model, manifest_df, device):
    """Devuelve (contextos [N, D], frame con sample_id/subject_id/label)."""
    model.to(device); model.eval()
    contexts, labels, sample_ids, subject_ids = [], [], [], []
    for row in manifest_df.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        out = model(seq, mask, return_context=True)
        ctx = out[1]  # (logits, context, ...) -> context
        contexts.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        sample_ids.append(row.sample_id); subject_ids.append(row.subject_id)
    frame = pd.DataFrame({"sample_id": sample_ids, "subject_id": subject_ids, "label": labels})
    return np.stack(contexts, axis=0), frame


@torch.no_grad()
def classify_contexts(model, contexts, device):
    """Aplica SOLO la cabeza clasificadora a vectores de contexto -> prob de Patients."""
    x = torch.from_numpy(np.asarray(contexts, dtype=np.float32)).to(device)
    logits = model.classifier(x).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


# --------------------------------------------------------------------------- #
#  Alineacion de dominio (SIN etiquetas)
# --------------------------------------------------------------------------- #
def align_standardize(src, tgt, eps=1e-6):
    """Empareja media y desviacion por dimension: lleva 'tgt' a la escala de 'src'."""
    mu_s, sd_s = src.mean(0), src.std(0) + eps
    mu_t, sd_t = tgt.mean(0), tgt.std(0) + eps
    return (tgt - mu_t) / sd_t * sd_s + mu_s


def align_coral(src, tgt, eps=1e-5):
    """CORAL: blanquea la covarianza de 'tgt' y la recolorea con la de 'src'."""
    def _cov(x):
        xc = x - x.mean(0, keepdims=True)
        return (xc.T @ xc) / max(len(x) - 1, 1) + eps * np.eye(x.shape[1])

    def _sqrtm_inv(mat, inverse):
        w, v = np.linalg.eigh(mat)
        w = np.clip(w, eps, None)
        d = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
        return (v * d) @ v.T

    cs, ct = _cov(src), _cov(tgt)
    whiten = _sqrtm_inv(ct, inverse=True)
    color = _sqrtm_inv(cs, inverse=False)
    tgt_c = tgt - tgt.mean(0, keepdims=True)
    aligned = tgt_c @ whiten @ color
    return aligned + src.mean(0, keepdims=True)


ALIGNERS = {"standardize": align_standardize, "coral": align_coral}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Alineacion de dominio CZ->EN para V3 noise.")
    p.add_argument("--checkpoint", default=str(Path("research/checkpoints/v3_final/noise_std003/v3_noise0.03_fold2_of_10.pt")))
    p.add_argument("--source-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_train_default.csv"),
                   help="Manifest de secuencias CHECO (dominio fuente, sin etiquetas usadas).")
    p.add_argument("--target-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test.csv"),
                   help="Manifest de secuencias INGLES (dominio objetivo / test).")
    p.add_argument("--method", default="standardize", choices=list(ALIGNERS.keys()))
    p.add_argument("--output-dir", default=str(CONFIG.paths.reports_dir / "CSV" / "transfer_en"))
    p.add_argument("--report-name", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rn = args.report_name or f"domain_align_{args.method}"
    print(f"Device: {device}  |  metodo de alineacion: {args.method}")

    model, mc, ckpt_thr = load_checkpoint_v3(Path(args.checkpoint))
    model.to(device)
    print(f"Checkpoint cargado. noise_std={mc.get('embedding_noise_std')}  umbral_checo={ckpt_thr:.3f}")

    src_df = pd.read_csv(args.source_manifest)
    tgt_df = pd.read_csv(args.target_manifest)
    print(f"Fuente (CZ): {len(src_df)} muestras  |  Objetivo (EN): {len(tgt_df)} muestras")
    print(f"EN labels: {tgt_df['label'].value_counts().to_dict()}")

    print("Extrayendo vectores de contexto (CZ)...")
    src_ctx, _ = extract_contexts(model, src_df, device)
    print("Extrayendo vectores de contexto (EN)...")
    tgt_ctx, tgt_meta = extract_contexts(model, tgt_df, device)

    # --- Scores SIN alinear (= zero-shot, linea base) ---
    prob_raw = classify_contexts(model, tgt_ctx, device)
    # --- Scores CON alineacion de dominio ---
    tgt_aligned = ALIGNERS[args.method](src_ctx, tgt_ctx)
    prob_aligned = classify_contexts(model, tgt_aligned, device)

    def _evaluate(prob, tag):
        frame = tgt_meta.copy()
        frame["score_patients"] = prob
        sample_ckpt = _metrics(frame["label"], frame["score_patients"], threshold=ckpt_thr)
        sweep_rows, sweep_best = _threshold_sweep(frame["label"], frame["score_patients"])
        subj_ckpt, _ = _subject_metrics(frame, ckpt_thr)
        print(f"\n--- {tag} ---")
        print(f"  umbral checo ({ckpt_thr:.2f}):  bal_acc={sample_ckpt['balanced_accuracy']:.4f}  "
              f"auc={sample_ckpt['auc']}  recall={sample_ckpt['recall']:.4f}")
        print(f"  mejor umbral ({sweep_best['threshold']:.2f}): bal_acc={sweep_best['balanced_accuracy']:.4f}")
        return {"sample_at_czech_threshold": sample_ckpt, "sample_best_sweep": sweep_best,
                "sample_sweep": sweep_rows, "subject_at_czech_threshold": subj_ckpt,
                "scores": frame[["sample_id", "subject_id", "label", "score_patients"]].to_dict(orient="records")}

    res_raw = _evaluate(prob_raw, "SIN alinear (zero-shot)")
    res_aligned = _evaluate(prob_aligned, f"CON alineacion [{args.method}]")

    delta_auc = (res_aligned["sample_at_czech_threshold"]["auc"] or 0) - (res_raw["sample_at_czech_threshold"]["auc"] or 0)
    delta_ba = (res_aligned["sample_at_czech_threshold"]["balanced_accuracy"]
                - res_raw["sample_at_czech_threshold"]["balanced_accuracy"])
    print(f"\n===== EFECTO DE LA ALINEACION (al umbral del checo) =====")
    print(f"  delta AUC          = {delta_auc:+.4f}")
    print(f"  delta balanced_acc = {delta_ba:+.4f}")

    report = {"config": {"checkpoint": args.checkpoint, "method": args.method,
                         "czech_threshold": ckpt_thr,
                         "n_source": int(len(src_df)), "n_target": int(len(tgt_df))},
              "note": "La alineacion usa solo estadisticas sin etiquetas del ingles; el test EN sigue limpio.",
              "raw_zero_shot": res_raw, "domain_aligned": res_aligned,
              "delta_at_czech_threshold": {"auc": delta_auc, "balanced_accuracy": delta_ba}}

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{rn}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {out_dir / f'metrics_{rn}.json'}")


if __name__ == "__main__":
    main()
