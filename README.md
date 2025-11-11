
# ITELv4 · Streamlit (HuBERT + LogisticRegression)

- Preprocesamiento: 16 kHz, filtro pasa-altas 100 Hz (orden 5), normalización pico, centrado a 2.0 s.
- Embeddings: `torchaudio.pipelines.HUBERT_BASE` ➜ 4608 dims (mean, std, first25, last25, delta, corr-c/tiempo).
- Transformación: `StandardScaler` entrenado.
- Modelo: `LogisticRegression` con `predict_proba`.

## Estructura
```
.
├── app.py
├── requirements.txt
└── artifacts
```

## Ejecutar
```
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue (Streamlit Community Cloud)
1. Sube estos archivos a un repo de GitHub.
2. En Streamlit Cloud, selecciona `app.py` como Main file.
3. Verifica que la instancia pueda instalar `torch` y `torchaudio` (CPU). 
