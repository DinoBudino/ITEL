import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import soundfile as sf
import pickle
import torch
import torchaudio
from scipy.signal import butter, filtfilt

# ===========================
# Configuración de la página
# ===========================
st.set_page_config(page_title="ITELv4 · Predicción (HuBERT + RegLog)", layout="centered")
st.title("ITELv4 · Predicción de audio")
st.caption("Pipeline: HuBERT embeddings (4608) → StandardScaler → LogisticRegression")

# ===========================
# Parámetros EXACTOS
# ===========================
TARGET_SR = 16000           # Hz
TARGET_DURATION = 2.0       # segundos
HP_CUTOFF = 100             # Hz, filtro pasa-altas
HIGHPASS_ORDER = 5

# ===========================
# Carga de artefactos
# ===========================
@st.cache_resource
def load_artifacts():
    # Rutas esperadas
    model_path = Path("artifacts/mejor_modelo_LOGREG.pkl")
    scaler_path = Path("artifacts/scaler_4608.pkl")

    if not model_path.exists():
        st.error("No se encontró el modelo en 'artifacts/mejor_modelo_LOGREG.pkl'.")
        st.stop()
    if not scaler_path.exists():
        st.error("No se encontró el scaler en 'artifacts/scaler_4608.pkl'.")
        st.stop()

    # Modelo scikit-learn
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    # Scaler scikit-learn
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # HuBERT BASE (CPU)
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model()
    hubert.eval()
    torch.set_num_threads(1)  # estabilidad en CPU

    return model, scaler, hubert

model, scaler, hubert = load_artifacts()

# ===========================
# Funciones de preprocesamiento y features
# ===========================
def preprocess_wave(y: np.ndarray, sr: int):
    """Resample → High-pass → Normalize (peak) → Center crop/pad a TARGET_DURATION."""
    # Resample a 16k
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Filtro pasa-altas
    if len(y) > 0:
        b, a = butter(HIGHPASS_ORDER, HP_CUTOFF / (0.5 * sr), btype="high", analog=False)
        y = filtfilt(b, a, y)

    # Normalización a pico
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    # Centrar a duración objetivo
    target_samples = int(TARGET_SR * TARGET_DURATION)
    n = len(y)
    if n > target_samples:
        start = (n - target_samples) // 2
        y = y[start:start + target_samples]
    elif n < target_samples:
        pad_left = (target_samples - n) // 2
        pad_right = target_samples - n - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    return y, sr


def extract_embedding(y_np: np.ndarray, hubert_model) -> np.ndarray:
    """HuBERT BASE → (T,768) → 4608 dims: mean, std, first25, last25, delta, corr(time)."""
    if y_np.size == 0:
        # Evita crash: vector silencioso mínimo
        y_np = np.zeros(int(TARGET_SR * TARGET_DURATION), dtype=np.float32)

    with torch.inference_mode():
        wav = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
        feats, _ = hubert_model(wav)                                # (1, T', 768)
        feats = feats.squeeze(0)                                    # (T', 768)

    T = feats.shape[0]
    if T < 2:
        feats = feats.repeat(2, 1)
        T = 2

    q = max(1, T // 4)

    mean_emb = feats.mean(dim=0)
    std_emb  = feats.std(dim=0)
    first_25 = feats[:q].mean(dim=0)
    last_25  = feats[-q:].mean(dim=0)
    delta    = first_25 - last_25

    # "slope" vía correlación con el tiempo; si es NaN (señal constante), pon 0
    x = torch.linspace(0.0, 1.0, steps=T)
    slope_vals = []
    for d in range(feats.shape[1]):
        y_d = feats[:, d]
        C = torch.corrcoef(torch.stack([x, y_d]))[0, 1]
        if torch.isnan(C):
            C = torch.tensor(0.0)
        slope_vals.append(C)
    slope_emb = torch.stack(slope_vals)

    emb = torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope_emb], dim=0)  # (4608,)
    return emb.numpy()


# ===========================
# KPIs / Métricas auxiliares
# ===========================
def compute_audio_kpis(y, sr, y_proc):
    dur_in = (len(y) / sr) if sr else 0.0
    dur_out = len(y_proc) / TARGET_SR if y_proc.size else 0.0
    peak = float(np.max(np.abs(y_proc))) if y_proc.size else 0.0
    rms = float(np.sqrt(np.mean(y_proc ** 2))) if y_proc.size else 0.0
    return {
        "dur_in_s": dur_in,
        "dur_out_s": dur_out,
        "peak": peak,
        "rms": rms,
    }


def compute_pred_kpis(proba: np.ndarray):
    proba = np.array(proba, dtype=float)
    s = proba.sum()
    if s > 0:
        proba = proba / s

    order = np.argsort(proba)[::-1]
    pmax = float(proba[order[0]])
    p2   = float(proba[order[1]]) if len(proba) > 1 else 0.0
    margin = pmax - p2

    # Entropía (bits)
    eps = 1e-12
    entropy = float(-(proba * np.log2(proba + eps)).sum())

    return {
        "pmax_pct": 100.0 * pmax,
        "margin_pct": 100.0 * margin,
        "entropy_bits": entropy,
        "order": order,
        "proba": proba,
    }


def map_class_label(c):
    # Asume 0 → Healthy, 1 → Patient si es binario; si no, devuelve string del valor
    try:
        return "Healthy" if c == 0 else "Patient" if c == 1 else str(c)
    except Exception:
        return str(c)


# ===========================
# UI principal
# ===========================
uploaded = st.file_uploader(
    "Sube un audio (.wav/.mp3/.ogg/.m4a)",
    type=["wav", "mp3", "ogg", "m4a"]
)

if uploaded is None:
    st.info("Coloca en `artifacts/` los archivos: `mejor_modelo_LOGREG.pkl` y `scaler_4608.pkl`. Luego sube un audio para predecir.")
    st.stop()

# Lee bytes una sola vez
data = uploaded.read()
if not data:
    st.error("El archivo está vacío o no se pudo leer.")
    st.stop()

# Muestra el audio (sin espectrograma)
st.audio(data)

# ========= Decodificar =========
t0 = time.perf_counter()
try:
    y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
except Exception:
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if hasattr(y, "ndim") and y.ndim > 1:
            y = y.mean(axis=1)  # a mono
    except Exception as e:
        st.error(f"No pude decodificar el audio. Prueba con WAV/MP3. Detalle: {e}")
        st.stop()
t_decode = (time.perf_counter() - t0) * 1000.0

# ========= Preprocesar =========
t1 = time.perf_counter()
y_proc, sr_proc = preprocess_wave(y, sr)
t_pre = (time.perf_counter() - t1) * 1000.0

# ========= HuBERT Embedding =========
t2 = time.perf_counter()
emb = extract_embedding(y_proc, hubert)  # (4608,)
t_emb = (time.perf_counter() - t2) * 1000.0

# ========= Escalar =========
t3 = time.perf_counter()
X = emb.reshape(1, -1)
Xs = scaler.transform(X)
t_scale = (time.perf_counter() - t3) * 1000.0

# ========= Predecir =========
t4 = time.perf_counter()
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(Xs)[0]
    pred_idx = int(np.argmax(proba))
else:
    proba = None
    pred_idx = int(model.predict(Xs)[0])
t_pred = (time.perf_counter() - t4) * 1000.0

# ========= Etiquetas / Probabilidades =========
classes = getattr(model, "classes_", None)
if classes is not None and pred_idx < len(classes):
    pred_class_raw = classes[pred_idx]
    pred_label = map_class_label(pred_class_raw)
else:
    pred_label = f"class_{pred_idx}"

st.subheader("Predicción")
st.write(f"Clase: **{pred_label}**")

if proba is not None:
    if classes is None:
        idx_labels = [f"class_{i}" for i in range(len(proba))]
    else:
        idx_labels = [map_class_label(c) for c in classes]

    # Probabilidades en porcentaje
    df_proba = pd.DataFrame(
        {"Probabilidad (%)": np.round(100.0 * proba, 2)},
        index=idx_labels
    )
    st.dataframe(df_proba, use_container_width=True)

    # KPIs de predicción
    pred_kpis = compute_pred_kpis(proba)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confianza máx.", f"{pred_kpis['pmax_pct']:.2f}%")
    c2.metric("Margen top-2", f"{pred_kpis['margin_pct']:.2f}%")
    c3.metric("Entropía", f"{pred_kpis['entropy_bits']:.3f} bits")

    # Logit para 'Patient' si es binario y existe la clase 1
    if classes is not None and len(classes) == 2 and 1 in classes:
        idx_patient = list(classes).index(1)
        p_patient = float(proba[idx_patient])
        logit = np.log((p_patient + 1e-9) / (1.0 - p_patient + 1e-9))
        st.caption(f"Logit(‘Patient’): {logit:.3f}")

# KPIs del audio (entrada/procesado)
audio_kpis = compute_audio_kpis(y, sr, y_proc)
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Duración (s)", f"{audio_kpis['dur_in_s']:.3f}")
d2.metric("Dur. proc. (s)", f"{audio_kpis['dur_out_s']:.3f}")
d3.metric("Pico", f"{audio_kpis['peak']:.3f}")
d4.metric("RMS", f"{audio_kpis['rms']:.3f}")
d5.metric("SR final (Hz)", f"{TARGET_SR}")

# Tiempos de inferencia (ms)
st.subheader("Tiempos de inferencia (ms)")
lat_df = pd.DataFrame({
    "Etapa": ["Decodificar", "Preprocesar", "HuBERT", "Escalar", "Predecir"],
    "ms": [t_decode, t_pre, t_emb, t_scale, t_pred],
})
st.dataframe(lat_df, use_container_width=True)
