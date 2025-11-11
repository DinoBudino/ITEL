# ITELv5 · Streamlit (HuBERT + MLP)

Aplicación web para la clasificación de voz (Saludable vs. Paciente) utilizando un pipeline de Deep Learning.

  - **Preprocesamiento**: Estandarización a 16 kHz, filtro pasa-altas a 100 Hz (orden 5), normalización de amplitud y ajuste de duración a 2.0 segundos.
  - **Embeddings**: Se utiliza el modelo `facebook/hubert-large-ls960-ft`. Se extraen los *hidden states* y se calculan estadísticas temporales (media, std, etc.) para generar un vector de **4608** características.
  - **Transformación**: Se aplica un `StandardScaler` (`.joblib`) entrenado con los datos de embeddings.
  - **Modelo**: Un **MLP (Perceptrón Multicapa)** entrenado con PyTorch (`.pt`) realiza la clasificación final.

## 🚀 Estructura de Archivos

```
.
├── app.py              # El código de la aplicación Streamlit
├── requirements.txt    # Las dependencias de Python
└── Modelos/
    ├── final_mlp_model.pt      # Modelo MLP entrenado
    └── final_mlp_scaler.joblib # Scaler entrenado
```

## ⚙️ Ejecución Local

1.  **Instalar dependencias**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Lanzar la aplicación**:

    ```bash
    streamlit run app.py
    ```

## ☁️ Despliegue (Streamlit Community Cloud)

1.  Sube esta estructura de archivos a un repositorio público de **GitHub**.
2.  En [Streamlit Community Cloud](https://share.streamlit.io/), crea una nueva aplicación y enlaza tu repositorio.
3.  Asegúrate de que el archivo principal sea `app.py`.
4.  ¡Despliega\! La plataforma instalará automáticamente las dependencias de `requirements.txt`.