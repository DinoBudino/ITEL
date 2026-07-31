# AGENTS.md

## Repo Shape
- No root task runner or workspace config. Work directly in the package you are changing.
- `apps/api/app/main.py` is the FastAPI entrypoint. Core serving logic lives in `packages/audio_pipeline/` and `packages/model_runtime/`.
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
- `GET /live` (and the compatibility alias `/health`) is a cheap process ping. `GET /ready` validates registered model bundles and returns 503 while artifacts are missing or invalid.
- Active serving code lives in `packages/audio_pipeline/` and `packages/model_runtime/`; `packages/inference_engine/` and `assets/models/` are legacy and are not imported by the API.
- Versioned manifests live under `models/registry/`. Binary artifacts remain ignored and are generated with the promotion scripts under `research/scripts/`.
- Optional content-addressed serving caches live under ignored `artifacts/serving_cache/`; they contain embeddings only and do not change model manifests or model digests.

## Local-Only Artifacts
- `assets/models/**` is gitignored except for `.gitkeep`. Do not assume model binaries are versioned or available on another machine.
- `data/**` is gitignored except for `.gitkeep`; generated sample catalogs and audio must not be committed.
- Treat `.env.telegram.local` as a secret local file used by the Cloudflare Telegram helper; do not print or commit its contents.

## Data And Caching Gotchas
- Sample-library endpoints are disabled by default. Enabling `ITEL_ENABLE_SAMPLE_LIBRARY=true` is only appropriate in an authorized private environment with a prebuilt local `data/.sample_audio_catalog.json`.

## Docker And Demo Flow
- `docker-compose.yml` exposes backend on `8000` and frontend on `5173`; datasets are not mounted into containers by default.
- Compose mounts only `artifacts/serving_cache` read-only. The API preloads one shared HuBERT resource and should run with one worker per GPU.
- `docker-compose.demo.yml` is an explicit temporary overlay that mounts the eligible Czech/English corpora and ignored `artifacts/demo_catalog` read-only behind nginx Basic Auth. Do not merge those mounts into normal production compose.
- `docker-compose.prod.yml` adds an nginx proxy on `http://127.0.0.1` and builds the frontend with `VITE_API_BASE_URL=/api`.
- `scripts/start_cloudflare_tunnel*.bat` are thin wrappers around PowerShell scripts that first run `docker compose -f docker-compose.prod.yml up -d --build` and then wait for `http://127.0.0.1` before launching `cloudflared`.

## Research Notes
- Research and serving use separate dependency sets; avoid mixing `apps/api/requirements.txt` and `research/requirements.txt` blindly.
- Research scripts log to MLflow by default; the local tracking store is `research/research_results/mlruns/` unless a script gets `--mlflow-tracking-uri`.
