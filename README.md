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
│  ├─ audio_pipeline/        # Preprocesamiento y HuBERT compartidos
│  └─ model_runtime/         # Registro y runtimes de modelos promovidos
├─ models/
│  └─ registry/              # Manifests versionados; binarios ignorados
├─ docs/
│  ├─ architecture/          # Contratos y ciclo de vida
│  └─ model-cards/           # Alcance y limitaciones
├─ research/
│  ├─ README.md              # Modelos finales y guia de investigacion
│  ├─ scripts/               # Entrenamiento, evaluacion y generacion de figuras
│  ├─ src/                   # Codigo compartido del pipeline experimental
│  ├─ research_results/      # CSV, visualizaciones y artefactos de investigacion
│  ├─ reports/               # Figuras y reportes de tesis
│  └─ requirements.txt       # Dependencias del entorno de investigacion
├─ scripts/
├─ docker-compose.yml
└─ README.md
```

## Arquitectura funcional

La inferencia oficial es por sujeto y recibe uno o mas WAV:

`WAV[] -> perfil RMS -> HuBERT -> modelo seleccionado -> agregacion por sujeto -> JSON`

Research, evaluacion LOSO y serving son etapas distintas. Los modelos de cada fold no se consideran artefactos desplegables. Consulta `docs/architecture/model-lifecycle.md`.

## Modelos finales

La linea de investigacion activa mantiene dos modelos finales documentados en `research/README.md`.

| Modelo | Dominio | Rol | Metricas clave |
| --- | --- | --- | --- |
| `V3 LayerMix` | Czech (LANNA) | Configuracion experimental final del dominio cerrado | Sample BalAcc `0.8783`, Subject BalAcc `0.9432`, Subject AUC `0.9739` |
| `Unified Multilingual` (`logreg_pca` + `centroid_shift`) | Czech + English | Adaptacion supervisada bilingue, no zero-shot | EN BalAcc `0.8542`, EN AUC `0.9323`, CZ BalAcc `0.9175`, CZ AUC `0.9710` |

Referencias rapidas:
- Documentacion de investigacion: `research/README.md`
- Comparativa maestra: `research/research_results/CSV/final_comparison/master_model_comparison.csv`
- Metricas V3 LayerMix: `research/research_results/CSV/metrics_v3_layermix.json`
- Metricas multilingues: `research/research_results/CSV/multilingual_transfer/multilingual_bidirectional_logreg_pca_centroid.json`

### Backend

- `FastAPI` expone `GET /live`, `GET /ready`, `GET /v1/models` y `POST /v1/predictions/subject`
- `packages/audio_pipeline/` evita diferencias entre research y serving
- `packages/model_runtime/` valida manifests, artefactos y checksums
- `/docs` se mantiene como interfaz tecnica secundaria para validacion y demostracion

### Frontend

- `apps/web/` contiene la SPA en `React + Vite`
- la interfaz principal esta pensada para sustentacion y demo local
- incluye secciones `Inicio`, `Prediccion` y `Como funciona`

## Ejecucion local

### 1. Backend

Se recomienda usar `Python 3.12` en un entorno virtual.

La biblioteca LANNA esta desactivada por defecto para evitar exponer audio, identificadores y etiquetas. Solo puede habilitarse de forma explicita con `ITEL_ENABLE_SAMPLE_LIBRARY=true` en un entorno privado y autorizado.

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

El proceso puede arrancar sin modelos para exponer `/live` y el catalogo. `/ready` devuelve `503` hasta que se generen los bundles reales:

```bash
python research/scripts/promote_multilingual_model.py
python research/scripts/train_czech_final.py
python research/scripts/promote_czech_layermix.py --checkpoint artifacts/czech-layermix-final.pt
```

### Cache privada y progreso

Los WAV ENNI conocidos pueden reutilizar exactamente sus representaciones HuBERT precomputadas sin modificar el modelo `1.0.0`. La cache no contiene audio, labels ni IDs de sujeto y permanece ignorada por Git:

```bash
python research/scripts/build_serving_feature_cache.py
```

El API carga un unico HuBERT compartido durante el arranque, valida la cache y expone progreso real mediante `POST /v1/predictions/subject/stream`. Los embeddings de audios nuevos solo se conservan en una cache LRU en memoria durante la vida del proceso.

Variables operativas:

- `ITEL_FEATURE_CACHE_DIR`, por defecto `artifacts/serving_cache`
- `ITEL_MEMORY_CACHE_ENTRIES`, por defecto `512`

Para GPU se recomienda un unico worker por proceso. El chunking de audio existe exclusivamente como benchmark offline y no forma parte del serving oficial:

```bash
python research/scripts/benchmark_chunked_serving.py --help
```

### Demo temporal con base completa

El perfil temporal monta en solo lectura los `2,975` audios Czech elegibles y los `48` audios English, protege toda la aplicacion con un PIN y muestra referencias unicamente despues de predecir. Los datasets no se copian a las imagenes Docker.

Preparacion local de catalogo y caches:

```bash
python research/scripts/build_demo_catalog.py
python research/scripts/build_serving_feature_cache.py --audio-manifest research/manifests/cz_main_cleaned.csv --sequence-manifest research/processed_data/manifests/hubert_sequences_natural_multilingual_light.csv --output-dir artifacts/serving_cache/multilingual-cz-en/1.0.0/cs
python research/scripts/build_czech_layermix_score_cache.py --batch-size 4
```

Inicio del stack autenticado y Cloudflare Tunnel:

```bat
scripts\start_cloudflare_demo.bat
```

El script muestra el usuario `demo` y un PIN temporal, valida los `146` sujetos, `3,023` audios y las caches, y elimina credenciales/contenedores al cerrar. El stack normal `docker-compose.prod.yml` permanece sin biblioteca.

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

Docker no incorpora ni monta datasets, secretos o resultados de investigacion. Los bundles promovidos se mantienen ignorados dentro de sus directorios del registro local y deben existir antes de construir una imagen lista para inferencia.

## Stack productivo local

Para ejecutar el frontend compilado con `Nginx` y el backend detras de un proxy local:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

URLs esperadas:

- app: `http://127.0.0.1`
- liveness: `http://127.0.0.1/api/live`
- readiness: `http://127.0.0.1/api/ready`

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

Cloudflare imprimira una URL publica temporal `https://*.trycloudflare.com`. No publiques una instancia que tenga habilitada la biblioteca de muestras ni datos sensibles montados.

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

- `ITEL_MODEL_REGISTRY_DIR`
- `ITEL_MAX_UPLOAD_SIZE_MB`
- `ITEL_MAX_TOTAL_UPLOAD_SIZE_MB`
- `ITEL_MAX_FILES_PER_SUBJECT`
- `ITEL_ENABLE_SAMPLE_LIBRARY`
- `ITEL_SAMPLE_AUDIO_DIR`
- `ITEL_FEATURE_CACHE_DIR`
- `ITEL_MEMORY_CACHE_ENTRIES`
- `ITEL_CORS_ORIGINS`

Los parametros que definen features, encoder, clases y thresholds pertenecen al manifest del modelo y no pueden modificarse mediante variables de entorno.

Los limites predeterminados aceptan hasta `30 MB` por archivo y `50 MB` por solicitud completa, cubriendo los WAV naturales de los corpus Czech e English utilizados por el proyecto.

## Dependencias por contexto

- `apps/api/requirements.txt`: dependencias para servir la API
- `research/requirements.txt`: dependencias del entorno de investigacion

## Seguimiento de experimentos con MLflow

- Los scripts de `research/` ya pueden registrar ejecuciones en `MLflow`.
- Si no indicas `--mlflow-tracking-uri`, los runs se guardan localmente en `research/research_results/mlruns`.

Ejemplos:

```bash
python research/scripts/train_optimized_v3.py
python research/scripts/evaluate_internal_cz.py --model-type logreg --n-splits 5
```

Para abrir la interfaz local de MLflow:

```bash
mlflow ui --backend-store-uri research/research_results/mlruns
```

## Convencion de nombres aplicada

- carpetas de producto: `apps/api`, `apps/web`
- paquetes reutilizables: `packages/audio_pipeline`, `packages/model_runtime`
- manifests de modelos: `models/registry`
- artefactos historicos legacy: `assets/models` (no usados por el API)
- material academico: `research/`

Dentro de `research/`, la referencia vigente para scripts, resultados y modelos finales es `research/README.md`.

## Nota de uso academico

Este sistema esta planteado como apoyo computacional para analisis experimental de voz. No reemplaza evaluacion clinica, diagnostico medico ni decision terapeutica profesional.
