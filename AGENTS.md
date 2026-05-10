# AGENTS.md

## Repo Shape
- No root task runner or workspace config. Work directly in the package you are changing.
- `apps/api/app/main.py` is the FastAPI entrypoint. Core serving logic lives in `packages/inference_engine/`.
- `apps/web/src/main.jsx` boots the React app. Routes are defined in `apps/web/src/App.jsx`.
- `research/scripts/*.py` are the main research CLIs; shared research code lives in `research/src/`.

## Verified Commands
- Backend install: `pip install -r apps/api/requirements.txt`
- Backend run: `python -m uvicorn apps.api.app.main:app --reload`
- Backend tests: `pytest apps/api/tests`
- Single backend test: `pytest apps/api/tests/test_api.py -k predict`
- Frontend install: `cd apps/web && npm install`
- Frontend dev server: `cd apps/web && npm run dev`
- Frontend production build: `cd apps/web && npm run build`
- Research env install: `pip install -r research/requirements.txt`

## Frontend Notes
- `apps/web/package.json` only defines `dev`, `build`, and `preview`; there is no configured lint or test command.
- API base URL is resolved in `apps/web/src/constants/api.js`: dev defaults to `http://127.0.0.1:8000`, production defaults to `/api`.
- The web Docker build uses `npm ci`, so keep `apps/web/package-lock.json` in sync with dependency changes.

## Backend Notes
- Backend tests are intentionally lightweight: `apps/api/tests/conftest.py` injects repo paths into `sys.path`, and tests monkeypatch the service so they do not require real model artifacts.
- `GET /health` is not a cheap ping. It calls `get_service()`, which runs `warmup()` and loads the scaler, ONNX model, and torchaudio bundle.
- Serving defaults come from `packages/inference_engine/config.py`. Important paths are relative to repo root, not `apps/api/`.
- The code expects `assets/models/final_mlp_model.onnx` and `assets/models/final_mlp_scaler.joblib`. The scaler exists locally; the ONNX file is not present in this checkout, so the real service will fail unless you provide it or override `ITEL_MODEL_DIR`.

## Local-Only Artifacts
- `assets/models/**` is gitignored except for `.gitkeep`. Do not assume model binaries are versioned or available on another machine.
- `data/Healthy/` and `data/Patients/` are gitignored. The repo only keeps `data/.sample_audio_catalog.json` under version control.
- Treat `.env.telegram.local` as a secret local file used by the Cloudflare Telegram helper; do not print or commit its contents.

## Data And Caching Gotchas
- Sample-library endpoints read from `ITEL_SAMPLE_AUDIO_DIR` (default `data/`).
- `apps/api/app/main.py` caches the sample catalog with `@lru_cache` and also persists `data/.sample_audio_catalog.json`.
- If files under `data/` change, restart the API; if the manifest is stale, delete `data/.sample_audio_catalog.json` so it can be rebuilt.

## Docker And Demo Flow
- `docker-compose.yml` exposes backend on `8000` and frontend on `5173`; it mounts `./data` into the backend as read-only `/service/data`.
- `docker-compose.prod.yml` adds an nginx proxy on `http://127.0.0.1` and builds the frontend with `VITE_API_BASE_URL=/api`.
- `scripts/start_cloudflare_tunnel*.bat` are thin wrappers around PowerShell scripts that first run `docker compose -f docker-compose.prod.yml up -d --build` and then wait for `http://127.0.0.1` before launching `cloudflared`.

## Research Notes
- Research and serving use separate dependency sets; avoid mixing `apps/api/requirements.txt` and `research/requirements.txt` blindly.
- Research scripts log to MLflow by default; the local tracking store is `research/research_results/mlruns/` unless a script gets `--mlflow-tracking-uri`.
