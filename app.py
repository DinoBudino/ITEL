import io
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import soundfile as sf
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, welch
from transformers import Wav2Vec2Processor, HubertModel

# ===========================
# DEFINICIÓN DE LA CLASE MLP (NECESARIA PARA CARGAR EL MODELO)
# ===========================
# IMPORTANTE: Esta definición debe coincidir EXACTAMENTE con la usada en el notebook 06.
# Si te da error al cargar, verifica que esta estructura sea idéntica a la de tu entrenamiento.
class MLP(nn.Module):
    def __init__(self, input_dim=4608, hidden_dim=512, num_classes=2, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.output_layer(x)
        return logits

# Hacemos que la clase esté disponible globalmente para que torch.load la encuentre
setattr(sys.modules['__main__'], 'MLP', MLP)

# ===========================
# Configuración de la página
# ===========================
st.set_page_config(page_title="ITELv5 · Predicción (HuBERT + MLP)", layout="centered")
# Estilos CSS personalizados para parecerse a tu imagen original si es necesario
st.markdown("""
    <style>
    .main-metric {font-size: 3em !important; font-weight: bold !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("ITELv5 · Predicción de voz")
st.caption("Pipeline V5: Audio (16kHz) → HuBERT Embeddings → Scaler → MLP Deep Learning")

# ===========================
# Parámetros EXACTOS V5
# ===========================
TARGET_SR = 16000
TARGET_DURATION = 2.0
HP_CUTOFF = 100
HIGHPASS_ORDER = 5
CLASSES = ['Healthy', 'Patients']

# ===========================
# Carga de Modelos V5
# ===========================
@st.cache_resource
def load_artifacts_v5():
    base_path = Path("Modelos")
    model_path = base_path / "final_mlp_model.pt"
    scaler_path = base_path / "final_mlp_scaler.joblib"

    if not model_path.exists():
        st.error(f"Falta el modelo: {model_path}")
        st.stop()
    if not scaler_path.exists():
        st.error(f"Falta el scaler: {scaler_path}")
        st.stop()

    try:
        # Cargar Scaler
        scaler = joblib.load(scaler_path)
        # Cargar Modelo PyTorch (mapeado a CPU por seguridad)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model, scaler
    except Exception as e:
        st.error(f"Error fatal cargando artefactos V5: {e}")
        st.stop()

@st.cache_resource
def load_hubert():
    with st.spinner("Cargando HuBERT..."):
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        model.eval()
        return processor, model

# Carga inicial
mlp_model, scaler = load_artifacts_v5()
hubert_processor, hubert_model = load_hubert()

# ===========================
# Funciones de Procesamiento
# ===========================
def compute_audio_kpis(y_orig, sr_orig, y_proc, sr_proc=16000):
    rms = np.sqrt(np.mean(y_orig**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_orig))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_orig, sr=sr_orig))
    noise_est = np.mean(librosa.feature.rms(y=y_orig[y_orig < 0.01*np.max(y_orig)])) if len(y_orig[y_orig < 0.01*np.max(y_orig)]) > 0 else 1e-9
    snr = 20 * np.log10(rms / (noise_est + 1e-9))
    return {"rms": rms, "zcr": zcr, "spec_cent": spec_cent, "snr": snr, "duration_orig": len(y_orig)/sr_orig}

def process_audio_v5(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    y_orig_copy = y.copy()
    sr_orig_copy = sr

    # 1. Resample
    if sr != TARGET_SR:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    # 2. Mono
    if y.ndim > 1: y = np.mean(y, axis=0)
    # 3. High-pass
    nyquist = 0.5 * sr
    b, a = butter(HIGHPASS_ORDER, HP_CUTOFF / nyquist, btype='high', analog=False)
    y_filt = filtfilt(b, a, y)
    # 4. Normalize
    rms = np.sqrt(np.mean(y_filt**2))
    y_norm = y_filt / (rms * 10.0) if rms > 1e-9 else y_filt
    # 5. Padding/Trimming 2s
    target_samples = int(TARGET_DURATION * sr)
    if len(y_norm) < target_samples:
        y_proc = np.pad(y_norm, (0, target_samples - len(y_norm)), mode='constant')
    else:
        y_proc = y_norm[:target_samples]
        
    return y_orig_copy, sr_orig_copy, y_proc

def get_embedding_v5(y_proc):
    inputs = hubert_processor(y_proc, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state[0].numpy()
    
    # Estadísticas temporales (igual que en tu entrenamiento V5)
    e_mean = np.mean(last_hidden_state, axis=0)
    e_std = np.std(last_hidden_state, axis=0)
    e_max = np.max(last_hidden_state, axis=0)
    e_min = np.min(last_hidden_state, axis=0)
    delta = np.diff(last_hidden_state, axis=0) if last_hidden_state.shape[0] > 1 else np.zeros_like(last_hidden_state)
    e_delta_mean = np.mean(delta, axis=0)
    
    return np.hstack([e_mean, e_std, e_max, e_min, e_delta_mean])

# ===========================
# Interfaz
# ===========================
uploaded_file = st.file_uploader("Sube un archivo de audio (WAV)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Ejecutar Análisis V5", type="primary", use_container_width=True):
        with st.spinner("Procesando con HuBERT + MLP..."):
            start_t = time.time()
            
            # 1. Procesamiento
            audio_bytes = uploaded_file.getvalue()
            y_orig, sr_orig, y_proc = process_audio_v5(audio_bytes)
            
            # 2. Embedding y Escalado
            emb = get_embedding_v5(y_proc)
            emb_scaled = scaler.transform(emb.reshape(1, -1))
            
            # 3. Inferencia MLP
            X_tensor = torch.tensor(emb_scaled, dtype=torch.float32)
            with torch.no_grad():
                logits = mlp_model(X_tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]
            
            idx_pred = np.argmax(probs)
            label_pred = CLASSES[idx_pred]
            confianza = probs[idx_pred]
            
            end_t = time.time()

            # === VISUALIZACIÓN DE RESULTADOS (ESTILO ORIGINAL) ===
            st.divider()
            st.header("Resultado del Análisis")
            
            # Métricas principales
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Predicción V5", label_pred, delta=None)
            kpi2.metric("Confianza", f"{confianza*100:.2f}%")
            kpi3.metric("Tiempo Inferencia", f"{end_t-start_t:.3f}s")

            # Tabla de probabilidades
            st.subheader("Detalles de Probabilidades")
            df_probs = pd.DataFrame([probs*100], columns=CLASSES, index=["Probabilidad (%)"])
            st.dataframe(df_probs.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1))

            # KPIs de señal (manteniendo tu funcionalidad anterior)
            st.subheader("KPIs de la Señal de Audio")
            audio_kpis = compute_audio_kpis(y_orig, sr_orig, y_proc)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMS (Original)", f"{audio_kpis['rms']:.4f}")
            c2.metric("ZCR Medio", f"{audio_kpis['zcr']:.4f}")
            c3.metric("Centroide Espectral", f"{int(audio_kpis['spec_cent'])} Hz")
            c4.metric("SNR Aprox.", f"{audio_kpis['snr']:.1f} dB")

            # Visualizaciones Avanzadas
            st.subheader("Visualización de Señales")
            tab1, tab2 = st.tabs(["Onda Procesada (Input Modelo)", "Análisis Espectral"])
            
            with tab1:
                st.line_chart(y_proc[::50], height=250) # Diezmado para velocidad
                
            with tab2:
                # Periodograma de Welch
                freqs, psd = welch(y_proc, fs=TARGET_SR, nperseg=1024)
                st.area_chart(pd.DataFrame({"PSD (dB)": 10 * np.log10(psd + 1e-9)}, index=freqs).iloc[:500]) # Mostrar hasta ~8kHz