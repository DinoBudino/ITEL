from __future__ import annotations

import argparse, json, warnings
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from research.src.config import CONFIG
from research.src.sequence_model import AttentionBiLSTMClassifierV3

OUTPUT_DIR = CONFIG.paths.reports_dir / "transfer_en"


def load_checkpoint_v3(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mc = payload["model_config"]
    model = AttentionBiLSTMClassifierV3(
        input_dim=mc["input_dim"], lstm_hidden=mc["lstm_hidden"], lstm_layers=mc["lstm_layers"],
        dropout=mc["dropout"], fc_hidden=mc["fc_hidden"], num_heads=mc["num_heads"],
        use_layer_mix=mc.get("use_layer_mix", False), num_hubert_layers=mc.get("num_hubert_layers", 12),
        embedding_noise_std=mc.get("embedding_noise_std", 0.0),
        attention_entropy_lambda=mc.get("attention_entropy_lambda", 0.0))
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, mc, payload.get("best_threshold", 0.5)


@torch.no_grad()
def extract_context_vectors(model, seq_manifest, device):
    model.to(device); model.eval()
    contexts, labels, subject_ids = [], [], []
    for row in seq_manifest.itertuples(index=False):
        payload = torch.load(row.sequence_path, map_location="cpu", weights_only=True)
        seq = payload["sequence"].float().unsqueeze(0).to(device)
        mask = payload["mask"].bool().unsqueeze(0).to(device)
        _, ctx = model(seq, mask, return_context=True)
        contexts.append(ctx.cpu().numpy().squeeze(0))
        labels.append(float(row.label == "Patients"))
        subject_ids.append(str(row.subject_id))
    return np.stack(contexts, axis=0), np.array(labels), subject_ids


def loso_logistic_regression(X, y, subjects, C_values=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)):
    unique_subjects = sorted(set(subjects))
    scaler = StandardScaler()
    all_scores, all_preds, all_test_subjects, all_test_labels, fold_results = [], [], [], [], []
    for test_subj in unique_subjects:
        train_mask = np.array([s != test_subj for s in subjects])
        test_mask = np.array([s == test_subj for s in subjects])
        X_train_raw, y_train = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        X_train = scaler.fit_transform(X_train_raw); X_test = scaler.transform(X_test_raw)
        train_subjs_arr = np.array(subjects)[train_mask]
        best_C, best_inner_ba = C_values[0], -1.0
        for C_val in C_values:
            inner_ba_folds = []
            inner_unique = sorted(set(train_subjs_arr.tolist()))
            if len(inner_unique) <= 1: continue
            for inner_test in inner_unique:
                inner_train_mask = np.array([s != inner_test for s in train_subjs_arr])
                inner_test_mask = np.array([s == inner_test for s in train_subjs_arr])
                if inner_train_mask.sum() == 0 or inner_test_mask.sum() == 0: continue
                clf = LogisticRegression(C=C_val, max_iter=5000, class_weight="balanced", random_state=42)
                with warnings.catch_warnings(): warnings.simplefilter("ignore"); clf.fit(X_train[inner_train_mask], y_train[inner_train_mask])
                inner_ba_folds.append(float(balanced_accuracy_score(y_train[inner_test_mask], clf.predict(X_train[inner_test_mask]))))
            if inner_ba_folds and np.mean(inner_ba_folds) > best_inner_ba:
                best_inner_ba = float(np.mean(inner_ba_folds)); best_C = C_val
        clf = LogisticRegression(C=best_C, max_iter=5000, class_weight="balanced", random_state=42)
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        patient_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        score = float(proba[0, patient_idx]); pred = int(score >= 0.5)
        all_scores.append(score); all_preds.append(pred)
        all_test_subjects.append(test_subj); all_test_labels.append(int(y_test[0]))
        fold_results.append({"test_subject": test_subj, "label": int(y_test[0]), "score": score, "pred": pred, "best_C": best_C})
    y_true = np.array(all_test_labels)
    y_scores, y_pred = np.array(all_scores), np.array(all_preds)
    m = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
         "precision": float(precision_score(y_true, y_pred, zero_division=0)), "recall": float(recall_score(y_true, y_pred, zero_division=0)),
         "f1": float(f1_score(y_true, y_pred, zero_division=0)),
         "auc": float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else None}
    return m, fold_results, y_scores, y_true, all_test_subjects


def main():
    p = argparse.ArgumentParser(description="Linear probing V3 noise on English.")
    p.add_argument("--checkpoint", default=str(Path("research/checkpoints/v3_final/noise_std003/v3_noise0.03_fold2_of_10.pt")))
    p.add_argument("--sequence-manifest", default=str(CONFIG.paths.processed_manifest_dir / "hubert_sequences_en_test.csv"))
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model, mc, ckpt_thr = load_checkpoint_v3(args.checkpoint)
    print(f"noise_std={mc.get('embedding_noise_std')}  threshold={ckpt_thr:.3f}")

    manifest = pd.read_csv(args.sequence_manifest)
    print(f"Extracting context vectors for {len(manifest)} samples...")
    X_en, y_en, subjects_en = extract_context_vectors(model, manifest, device)
    print(f"  Shape: {X_en.shape}")

    metrics_lr, fold_results, scores_lr, y_true, test_subjects = loso_logistic_regression(X_en, y_en, subjects_en)

    print(f"\nLinear Probe V3 Noise (LOSO LR):")
    for k, v in metrics_lr.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    subject_details = [{"subject_id": sid, "label": int(y_true[i]), "lr_probe_score": float(scores_lr[i]),
                        "lr_pred": int(scores_lr[i] >= 0.5), "lr_correct": int(scores_lr[i] >= 0.5) == int(y_true[i])}
                       for i, sid in enumerate(test_subjects)]

    report = {"environment": {"cuda_available": torch.cuda.is_available()}, "checkpoint": str(args.checkpoint),
              "linear_probe_loso": {**{k: round(v, 6) if v is not None else None for k, v in metrics_lr.items()}, "folds": fold_results},
              "per_subject": subject_details}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "linear_probe_v3_noise_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(subject_details).to_csv(OUTPUT_DIR / "linear_probe_v3_noise_per_subject.csv", index=False)

    print(f"\nPer-subject:")
    for d in subject_details:
        print(f"  {d['subject_id']:>6s}  label={d['label']}  score={d['lr_probe_score']:.4f}  {'OK' if d['lr_correct'] else 'ERR'}")
    print(f"\nReport: {OUTPUT_DIR / 'linear_probe_v3_noise_report.json'}")


if __name__ == "__main__":
    main()
