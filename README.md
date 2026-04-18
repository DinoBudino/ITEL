# ITEL Audio TEL

Proyecto de tesis para clasificacion de voz en las clases `Healthy` y `Patients` a partir de archivos WAV. La solucion queda reorganizada con una estructura profesional separando aplicaciones, paquetes reutilizables, activos del modelo y material de investigacion.

## Estructura del repositorio

```text
itel-audio-tel/
├─ apps/
│  ├─ api/
│  │  ├─ app/                # Entrada FastAPI
│  │  ├─ tests/              # Pruebas del backend
│  │  ├─ requirements.txt    # Dependencias de serving
│  │  └─ Dockerfile
│  └─ web/
│     ├─ src/                # Interfaz React + Vite
│     ├─ package.json
│     └─ Dockerfile
├─ packages/
│  └─ inference_engine/      # Motor reutilizable de inferencia
├─ assets/
│  └─ models/                # Modelo ONNX, scaler y artefactos
├─ research/
│  ├─ notebooks/             # Cuadernos del trabajo experimental
│  ├─ reports/               # Figuras y reportes
│  ├─ processed_data/        # Datos procesados del pipeline
│  └─ requirements.txt       # Dependencias del entorno de investigacion
├─ scripts/
├─ docker-compose.yml
└─ README.md
```

## Arquitectura funcional

La ruta oficial de inferencia se mantiene sin reescribir la logica principal:

`WAV -> preprocessing -> HuBERT embeddings -> scaler -> ONNX classifier -> JSON`

### Backend

- `FastAPI` expone los endpoints `GET /health`, `GET /model-info` y `POST /predict`
- `packages/inference_engine/` conserva el pipeline reutilizable
- `/docs` se mantiene como interfaz tecnica secundaria para validacion y demostracion

### Frontend

- `apps/web/` contiene la SPA en `React + Vite`
- la interfaz principal esta pensada para sustentacion y demo local
- incluye secciones `Inicio`, `Prediccion` y `Como funciona`

## Ejecucion local

### 1. Backend

Instalar dependencias:

```bash
py -3.14 -m pip install -r apps/api/requirements.txt
```

Levantar API:

```bash
py -3.14 -m uvicorn apps.api.app.main:app --reload
```

URLs:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

### 2. Frontend

```bash
cd apps/web
npm install
npm run dev
```

URL:

- Web: `http://127.0.0.1:5173`

Si necesitas apuntar el frontend a otra URL del backend:

```bash
set VITE_API_BASE_URL=http://127.0.0.1:8000 && npm run dev
```

## Docker Compose

Cuando Docker este disponible en tu sistema:

```bash
docker compose up --build
```

Servicios esperados:

- frontend: `http://localhost:5173`
- backend: `http://localhost:8000`
- docs: `http://localhost:8000/docs`

## Variables de entorno del backend

- `ITEL_MODEL_DIR`
- `ITEL_TARGET_SR`
- `ITEL_HP_CUTOFF`
- `ITEL_TARGET_DURATION`
- `ITEL_MAX_UPLOAD_SIZE_MB`
- `ITEL_MODEL_VERSION`
- `ITEL_HUBERT_BUNDLE`
- `ITEL_LOG_LEVEL`
- `ITEL_CORS_ORIGINS`

Valor por defecto importante:

- `ITEL_MODEL_DIR=assets/models`

## Dependencias por contexto

- `apps/api/requirements.txt`: dependencias para servir la API
- `research/requirements.txt`: dependencias del entorno de investigacion y notebooks

## Convencion de nombres aplicada

- carpetas de producto: `apps/api`, `apps/web`
- paquete reusable: `packages/inference_engine`
- activos del modelo: `assets/models`
- material academico: `research/`

## Nota de uso academico

Este sistema esta planteado como apoyo computacional para analisis experimental de voz. No reemplaza evaluacion clinica, diagnostico medico ni decision terapeutica profesional.
