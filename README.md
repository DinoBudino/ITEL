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

Se recomienda usar `Python 3.12` en un entorno virtual.

Importante para el modo LANNA:

- el repositorio no incluye los audios del dataset en `data/Healthy` y `data/Patients`
- si quieres que funcione la tabla "Elegir de LANNA", debes copiar esos WAV manualmente dentro de `data/`
- el archivo `data/.sample_audio_catalog.json` solo sirve como catalogo rapido; no reemplaza los audios

Instalar dependencias:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r apps/api/requirements.txt
```

Levantar API:

```bash
python -m uvicorn apps.api.app.main:app --reload
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

Nota sobre LANNA en Docker:

- antes de levantar los contenedores, coloca el dataset de audios en `data/Healthy` y `data/Patients`
- el volumen de Docker monta `./data` dentro del backend, por eso esa carpeta debe existir localmente
- si no copias los audios, la app seguira funcionando para subir WAV manualmente, pero la biblioteca LANNA no tendra muestras reales para analizar

## Stack productivo local

Para ejecutar el frontend compilado con `Nginx` y el backend detras de un proxy local:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

URLs esperadas:

- app: `http://127.0.0.1`
- api: `http://127.0.0.1/api/health`

Para revisar estado y logs:

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

Para apagar el stack:

```bash
docker compose -f docker-compose.prod.yml down
```

### Produccion local con Cloudflare Tunnel

Para publicar una demo desde tu propia PC sin IP publica y sin abrir puertos, puedes usar el stack de produccion junto con un Quick Tunnel de Cloudflare.

1. Instalar `cloudflared`.
2. Levantar el stack productivo y abrir el tunel:

```bash
scripts\start_cloudflare_tunnel.bat
```

El script:

- levanta `docker-compose.prod.yml`
- espera a que el proxy local responda en `http://127.0.0.1`
- abre un Quick Tunnel apuntando a `http://127.0.0.1:80`

Cloudflare imprimira una URL publica temporal `https://*.trycloudflare.com` que puedes compartir para la demo.

Notas importantes:

- el Quick Tunnel solo dura mientras `cloudflared` siga corriendo
- la URL publica cambia cada vez que se vuelve a iniciar
- este modo esta pensado para demo, no para despliegue permanente

### Cloudflare Tunnel con notificacion a Telegram

Tambien puedes usar un script separado que envie automaticamente la URL publica a Telegram.

La primera vez, el script puede pedirte estos dos valores y guardarlos automaticamente en `.env.telegram.local`:

```powershell
ITEL_TELEGRAM_BOT_TOKEN=<tu-bot-token>
ITEL_TELEGRAM_CHAT_ID=<tu-chat-id>
```

Opcionalmente, tambien puedes definirlos como variables de entorno antes de ejecutar el script.

Ejecucion:

```bash
scripts\start_cloudflare_tunnel_telegram.bat
```

El mensaje enviado al chat sera:

```text
ITEL demo online: https://*.trycloudflare.com
```

Para obtener el token, crea un bot con `@BotFather`. Para obtener el `chat_id`, escribe al bot y consulta `https://api.telegram.org/bot<TOKEN>/getUpdates`.

Si alguna vez quieres cambiar la configuracion guardada, edita o elimina `.env.telegram.local` y el script volvera a preguntarte ambos valores.

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
