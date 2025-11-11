import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import StandardScaler

# ===========================
# DEFINICIÓN DE LA CLASE MLP
# ===========================
class MLP(nn.Module):
    def __init__(self, in_dim=4608, hidden=512, dropout=0.3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===========================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================
st.set_page_config(page_title="ITEL", layout="centered")
st.title("ITEL · Identificación del TEL")
st.caption("Pipeline: Audio (16kHz) → HuBERT_BASE Embeddings → Scaler → MLP Ensemble (5 folds)")

# ===========================
# PARÁMETROS
# ===========================
TARGET_SR = 16000
TARGET_DURATION = 2.0
HP_CUTOFF = 100
HIGHPASS_ORDER = 5
CLASSES = ['Healthy', 'Patients']

# ===========================
# CARGA DE MODELOS (ENSAMBLE DE 5 FOLDS)
# ===========================
@st.cache_resource
def load_ensemble_models():
    """Carga los 5 modelos del ensamblaje como en el notebook original"""
    base_path = Path("Modelos")
    
    # Verificar que existen los 5 folds
    fold_files = [base_path / f'itelv5_mlp_fold_{i}.pt' for i in range(1, 6)]
    for fold_file in fold_files:
        if not fold_file.exists():
            st.error(f"No se encontró: {fold_file.name}")
            st.error("Asegúrate de tener los 5 modelos de fold en la carpeta 'Modelos'")
            st.stop()
    
    # Cargar los 5 modelos
    models = []
    for i in range(1, 6):
        model_path = base_path / f'itelv5_mlp_fold_{i}.pt'
        model = MLP(in_dim=4608, hidden=256, dropout=0.5, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        models.append(model)
    
    st.success(f"Cargados {len(models)} modelos de ensamblaje")
    return models

@st.cache_resource
def load_ensemble_scaler():
    """Carga el scaler del ensamblaje (igual al notebook)"""
    base_path = Path("Modelos")
    scaler_path = base_path / "linear_head_scaler.npz"
    
    if not scaler_path.exists():
        st.error(f"No se encontró: {scaler_path.name}")
        st.error("Usa el scaler del notebook (linear_head_scaler.npz) no el de joblib")
        st.stop()
    
    # Cargar como en el notebook
    scaler_data = np.load(scaler_path)
    scaler = StandardScaler()
    scaler.mean_ = scaler_data['mean']
    scaler.scale_ = scaler_data['scale']
    
    st.success("Scaler de ensamblaje cargado")
    return scaler

@st.cache_resource
def load_hubert():
    with st.spinner("Cargando HuBERT BASE (torchaudio.pipelines.HUBERT_BASE)..."):
        bundle = torchaudio.pipelines.HUBERT_BASE
        model = bundle.get_model()
        model.eval()
        return model

# Cargar todos los modelos
hubert_model = load_hubert()
ensemble_models = load_ensemble_models()
ensemble_scaler = load_ensemble_scaler()

# ===========================
# FUNCIONES AUXILIARES (IGUAL AL NOTEBOOK)
# ===========================
def highpass_filter(y, sr, cutoff=HP_CUTOFF, order=5):
    b, a = butter(order, cutoff / (0.5 * sr), btype='high', analog=False)
    return filtfilt(b, a, y)

def ajustar_duracion(y, sr, target_dur=TARGET_DURATION):
    target_len = int(sr * target_dur)
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        pad = (target_len - len(y)) // 2
        y = np.pad(y, (pad, target_len - len(y) - pad))
    return y

def process_audio_v5(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # Resample si es necesario
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # Convertir a mono si es estéreo
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    
    # Aplicar filtro pasa-altas
    y = highpass_filter(y, sr, cutoff=HP_CUTOFF)
    
    # Normalización (igual que en el notebook)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    
    # Ajustar duración
    y = ajustar_duracion(y, sr)
    
    return y.astype(np.float32)

def calculate_slope_robust(x, y):
    """Calcula slope usando fórmula de regresión lineal robusta"""
    n = len(x)
    if n < 2:
        return 0.0
        
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return torch.clamp(slope, -2.0, 2.0)

def get_embedding_v5_corrected(y):
    """Versión CORREGIDA que replica exactamente el notebook"""
    
    # Convertir a tensor como en el notebook
    wav_tensor = torch.tensor(y).unsqueeze(0)  # (1, num_samples)
    
    with torch.no_grad():
        features, _ = hubert_model(wav_tensor)  # (1, T, 768) - IGUAL AL NOTEBOOK
    
    features = features.squeeze(0)  # (T, 768)
    T = features.shape[0]
    
    # Mover a CPU antes de operaciones numpy
    features = features.cpu()
    
    # CALCULAR EXACTAMENTE LAS MISMAS 6 ESTADÍSTICAS QUE EL NOTEBOOK
    mean_emb = features.mean(dim=0)
    std_emb = features.std(dim=0)
    first_25 = features[:max(1, T//4)].mean(dim=0)
    last_25 = features[-max(1, T//4):].mean(dim=0)
    delta = first_25 - last_25
    
    slope_emb = torch.zeros(768)
    x = torch.arange(T, dtype=torch.float32)
    
    for dim in range(768):
        y_dim = features[:, dim]
        if torch.var(y_dim) == 0 or T < 2:
            slope_emb[dim] = 0
        else:
            slope_emb[dim] = calculate_slope_robust(x, y_dim)
    
    # Concatenar en el MISMO ORDEN que el notebook
    emb = torch.cat([mean_emb, std_emb, first_25, last_25, delta, slope_emb]).numpy()
    
    if emb.shape[0] != 4608:
        st.error(f"Dimensión incorrecta: {emb.shape[0]} (esperaba 4608)")
        st.stop()
    
    st.success(f"Embedding generado: {emb.shape[0]} dimensiones")
    return emb

def compute_audio_kpis(y_orig, sr_orig):
    rms = np.sqrt(np.mean(y_orig**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_orig))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_orig, sr=sr_orig))
    noise_est = np.mean(librosa.feature.rms(y=y_orig[y_orig < 0.01*np.max(y_orig)])) if len(y_orig[y_orig < 0.01*np.max(y_orig)]) > 0 else 1e-9
    snr = 20 * np.log10(rms / (noise_est + 1e-9))
    return {"rms": rms, "zcr": zcr, "spec_cent": spec_cent, "snr": snr, "duration": len(y_orig)/sr_orig}

def ensemble_predict(emb_scaled):
    """Realiza predicción por ensamblaje igual al notebook"""
    X_tensor = torch.tensor(emb_scaled, dtype=torch.float32)
    
    all_probas = []
    individual_details = []
    
    with torch.no_grad():
        for i, model in enumerate(ensemble_models):
            logits = model(X_tensor)
            probas = F.softmax(logits, dim=1).numpy()[0]
            all_probas.append(probas)
            
            individual_details.append({
                'model': f'fold_{i+1}',
                'probas': probas,
                'pred_class': 'Healthy' if probas[0] > probas[1] else 'Patient'
            })
    
    # Promediar las probabilidades [N_models, N_classes]
    avg_probas = np.mean(np.array(all_probas), axis=0)
    
    # Mostrar variabilidad entre modelos
    probas_array = np.array(all_probas)
    variability = np.std(probas_array, axis=0)
    
    return avg_probas, variability, individual_details

# ===========================
# INTERFAZ - RESULTADOS PRECARGADOS
# ===========================

# Inicializar variables en session_state si no existen
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'individual_details' not in st.session_state:
    st.session_state.individual_details = None
if 'audio_kpis' not in st.session_state:
    st.session_state.audio_kpis = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'audio_waveform' not in st.session_state:
    st.session_state.audio_waveform = None

# =======================
# SECCIÓN DE SUBIDA DE AUDIO
# =======================
st.header("Subir Audio para Análisis")
uploaded_file = st.file_uploader("Selecciona un archivo de audio WAV", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Ejecutar Análisis V5", type="primary", use_container_width=True):
        with st.spinner("Procesando con HuBERT + MLP Ensemble (5 modelos)..."):
            start = time.time()

            # Procesar audio
            audio_bytes = uploaded_file.getvalue()
            y_proc = process_audio_v5(audio_bytes)

            # Embedding HuBERT
            emb = get_embedding_v5_corrected(y_proc)
            
            # Escalar embedding
            emb_scaled = ensemble_scaler.transform(emb.reshape(1, -1))

            # Inferencia por ENSAMBLE
            avg_probas, variability, individual_details = ensemble_predict(emb_scaled)

            pred_idx = np.argmax(avg_probas)
            label = CLASSES[pred_idx]
            conf = avg_probas[pred_idx]
            end = time.time()

            # Guardar resultados en session_state
            st.session_state.analysis_complete = True
            st.session_state.prediction_result = {
                'label': label,
                'confidence': conf,
                'variability': variability[pred_idx]
            }
            st.session_state.probabilities = avg_probas
            st.session_state.individual_details = individual_details
            st.session_state.audio_kpis = compute_audio_kpis(y_proc, TARGET_SR)
            st.session_state.processing_time = end - start
            st.session_state.audio_waveform = y_proc

# =======================
# SECCIÓN DE RESULTADOS (SIEMPRE VISIBLE)
# =======================
st.divider()
st.header("Resultados del Análisis")

if st.session_state.analysis_complete:
    # Resultados disponibles
    result = st.session_state.prediction_result
    avg_probas = st.session_state.probabilities
    individual_details = st.session_state.individual_details
    audio_kpis = st.session_state.audio_kpis
    
    # Mostrar confianza y variabilidad con tooltips
    st.subheader("Métricas de Predicción")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            "Predicción", 
            result['label'],
            help="Clasificación final basada en el consenso de los 5 modelos"
        )
    
    with kpi2:
        st.metric(
            "Confianza", 
            f"{result['confidence']*100:.2f}%",
            help="Probabilidad promedio de la clase predicha entre todos los modelos"
        )
    
    with kpi3:
        st.metric(
            "Tiempo de Procesamiento", 
            f"{st.session_state.processing_time:.2f}s",
            help="Tiempo total desde la carga del audio hasta el resultado final"
        )
    
    with kpi4:
        st.metric(
            "Variabilidad entre Modelos", 
            f"{result['variability']*100:.2f}%",
            help="Desviación estándar en las predicciones de los 5 modelos (valores bajos indican mayor acuerdo)"
        )

    st.subheader("Detalles de Probabilidades")
    st.caption("Distribución de probabilidades promedio del ensamblaje")
    df_probs = pd.DataFrame([avg_probas*100], columns=CLASSES, index=["Probabilidad (%)"])
    st.dataframe(df_probs.style.format("{:.2f}%").background_gradient(cmap='Blues', axis=1))
    
    # Predicciones individuales
    st.subheader("Predicciones Individuales por Modelo")
    st.caption("Resultados detallados de cada uno de los 5 modelos del ensamblaje")
    individual_df = pd.DataFrame([
        {
            'Modelo': detail['model'],
            'Proba Healthy': f"{detail['probas'][0]:.3f}",
            'Proba Patient': f"{detail['probas'][1]:.3f}",
            'Predicción': detail['pred_class']
        }
        for detail in individual_details
    ])
    st.dataframe(individual_df)
    
    # Análisis de consenso
    healthy_votes = sum(1 for d in individual_details if d['pred_class'] == 'Healthy')
    patient_votes = sum(1 for d in individual_details if d['pred_class'] == 'Patient')
    st.info(f"**Consenso**: {healthy_votes}/5 modelos votan por Healthy, {patient_votes}/5 por Patient")

    # KPIs del audio
    st.subheader("Características de la Señal de Audio")
    st.caption("Métricas técnicas de la señal de audio procesada")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric(
            "RMS", 
            f"{audio_kpis['rms']:.4f}",
            help="Root Mean Square - Medida de la amplitud promedio de la señal"
        )
    
    with c2:
        st.metric(
            "Zero Crossing Rate", 
            f"{audio_kpis['zcr']:.4f}",
            help="Tasa de cruces por cero - Indica la frecuencia de cambios de signo en la señal"
        )
    
    with c3:
        st.metric(
            "Centroide Espectral", 
            f"{int(audio_kpis['spec_cent'])} Hz",
            help="Centro de masa del espectro - Representa el 'centro de gravedad' frecuencial"
        )
    
    with c4:
        st.metric(
            "SNR Aproximado", 
            f"{audio_kpis['snr']:.1f} dB",
            help="Relación Señal-Ruido - Ratio entre la potencia de la señal y el ruido de fondo"
        )

    # Visualizaciones
    st.subheader("Visualización de Señales")
    tab1, tab2 = st.tabs(["Onda Procesada (Input Modelo)", "Análisis Espectral"])
    
    with tab1:
        st.caption("Representación temporal de la señal de audio después del preprocesamiento")
        st.line_chart(st.session_state.audio_waveform[::50], height=250)
    
    with tab2:
        st.caption("Densidad espectral de potencia - Distribución de energía en frecuencias")
        freqs, psd = welch(st.session_state.audio_waveform, fs=TARGET_SR, nperseg=1024)
        st.area_chart(pd.DataFrame({"PSD (dB)": 10 * np.log10(psd + 1e-9)}, index=freqs).iloc[:500])

else:
    # Estado inicial - resultados vacíos
    st.info("**Esperando análisis...** Sube un archivo de audio WAV y haz clic en 'Ejecutar Análisis V5' para ver los resultados.")
    
    # Placeholders para mantener el layout
    st.subheader("Métricas de Predicción")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Predicción", "-", help="Clasificación final basada en el consenso de los 5 modelos")
    
    with kpi2:
        st.metric("Confianza", "0%", help="Probabilidad promedio de la clase predicha entre todos los modelos")
    
    with kpi3:
        st.metric("Tiempo de Procesamiento", "0.00s", help="Tiempo total desde la carga del audio hasta el resultado final")
    
    with kpi4:
        st.metric("Variabilidad entre Modelos", "0%", help="Desviación estándar en las predicciones de los 5 modelos")

    st.subheader("Detalles de Probabilidades")
    st.caption("Distribución de probabilidades promedio del ensamblaje")
    empty_probs = pd.DataFrame([[0, 0]], columns=CLASSES, index=["Probabilidad (%)"])
    st.dataframe(empty_probs.style.format("{:.2f}%"))
    
    st.subheader("Predicciones Individuales por Modelo")
    st.caption("Resultados detallados de cada uno de los 5 modelos del ensamblaje")
    empty_individual = pd.DataFrame({
        'Modelo': [f'fold_{i}' for i in range(1, 6)],
        'Proba Healthy': ['0.000' for _ in range(5)],
        'Proba Patient': ['0.000' for _ in range(5)],
        'Predicción': ['-' for _ in range(5)]
    })
    st.dataframe(empty_individual)
    
    st.info("**Consenso**: 0/5 modelos - Esperando análisis")

    # KPIs del audio
    st.subheader("Características de la Señal de Audio")
    st.caption("Métricas técnicas de la señal de audio procesada")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("RMS", "0.0000", help="Root Mean Square - Medida de la amplitud promedio de la señal")
    
    with c2:
        st.metric("Zero Crossing Rate", "0.0000", help="Tasa de cruces por cero - Indica la frecuencia de cambios de signo en la señal")
    
    with c3:
        st.metric("Centroide Espectral", "0 Hz", help="Centro de masa del espectro - Representa el 'centro de gravedad' frecuencial")
    
    with c4:
        st.metric("SNR Aproximado", "0.0 dB", help="Relación Señal-Ruido - Ratio entre la potencia de la señal y el ruido de fondo")

    # Visualizaciones vacías
    st.subheader("Visualización de Señales")
    tab1, tab2 = st.tabs(["Onda Procesada (Input Modelo)", "Análisis Espectral"])
    
    with tab1:
        st.caption("Representación temporal de la señal de audio después del preprocesamiento")
        st.info("Gráfico disponible después del análisis")
    
    with tab2:
        st.caption("Densidad espectral de potencia - Distribución de energía en frecuencias")
        st.info("Gráfico disponible después del análisis")

# ===========================
# INFORMACIÓN ADICIONAL
# ===========================
with st.expander("Información del Sistema"):
    st.write("""
    **Configuración actual:**
    - **Modelo**: MLP Ensemble (5 folds)
    - **Embedding**: HuBERT Base (768 dim → 6 stats → 4608 dim)
    - **Preprocesamiento**: 16kHz, 2s, filtro high-pass 100Hz
    - **Escalado**: StandardScaler del entrenamiento original
    
    **Archivos requeridos:**
    - `itelv5_mlp_fold_1.pt` a `itelv5_mlp_fold_5.pt`
    - `linear_head_scaler.npz`
    """)

# Botón para resetear resultados
if st.session_state.analysis_complete:
    if st.button("Limpiar Resultados", type="secondary"):
        st.session_state.analysis_complete = False
        st.session_state.prediction_result = None
        st.session_state.probabilities = None
        st.session_state.individual_details = None
        st.session_state.audio_kpis = None
        st.session_state.processing_time = None
        st.session_state.audio_waveform = None
        st.rerun()