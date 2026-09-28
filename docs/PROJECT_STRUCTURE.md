# Project structure

High-level layout of the Findash / Octopus Trading Platform repo and where to find things.

## Root layout

| Path | Purpose |
|------|--------|
| `config/` | Environment template and API key placeholders (no secrets). Use `config/env.example` to create `.env`. |
| `docker/` | Dockerfiles for API, Celery, and LLM inference. Compose files stay at root. |
| `docs/` | Architecture, guides, deployment, and archived notes. |
| `frontend-nextjs/` | Next.js 15 app (public sample dashboard, trading UI, reports). `npm run dev` listens on **:3003**. |
| `requirements/` | Python dependency lists: `requirements.txt` (main), `requirements-dev.txt`, `requirements-llm.txt`, etc. |
| `scripts/` | One-off and automation: deploy, DB init, health checks, `start-dev.sh`, `start-services.sh`. |
| `src/` | Backend Python: FastAPI app, agents, strategies, LLM, data pipelines. |
| `tests/` | Pytest tests for the backend. |
| `wiki-content/` | Wiki / docs content (e.g. for GitHub wiki). |
| `alembic.ini` | Alembic config (migrations). Run from repo root. |
| `Makefile` | Common commands: `make dev`, `make test`, `make setup`, Docker targets. |
| `start.py` | Backend entrypoint (`python3 start.py --reload` from **repo root**). Inserts the project root on `sys.path` before importing `src.*`. `make dev` runs uvicorn on `:8000` and skips those checks. |
| `docker-compose-core.yml` | Core stack (API, frontend, DB, Redis, Celery). |
| `docker-compose-complete.yml` | Full stack including extra services. |

## Backend (`src/`)

- `src/__init__.py` – Marks the backend as a package (needed for `python3 start.py`).
- `src/main_refactored.py` – FastAPI app entry.
- `src/core/` – Config, logging, Celery, security. Default `API_PORT=8000`.
- `src/api/` – REST endpoints and route modules.
- `src/llm/`, `src/strategies/`, `src/trading/`, `src/risk/`, etc. – Feature domains.

## Config and env

- **Secrets**: Never commit `.env` or `env.local`. Copy `config/env.example` to `.env` and fill in values.
- **API keys**: Use env vars (e.g. `ALPHA_VANTAGE_API_KEY`) or the placeholder file `config/api_keys_config.py` for local reference only.

## Quick commands

- Backend: `python3 start.py --reload` (repo root, `:8000`) or `make dev`
- Frontend: `cd frontend-nextjs && npm run dev` → http://localhost:3003
- Full stack: `docker compose -f docker-compose-core.yml up` → UI `:3003`, API `:8011`
- Tests: `make test`
- Public sample dashboard / ports / auth leftovers: [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md)
