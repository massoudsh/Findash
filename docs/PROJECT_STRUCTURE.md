# Project structure

High-level layout of the Findash / Octopus Trading Platform repo and where to find things.

## Root layout

| Path | Purpose |
|------|--------|
| `config/` | Environment template and API key placeholders (no secrets). Use `config/env.example` to create `.env`. |
| `docker/` | Dockerfiles for API, Celery, and LLM inference. Compose files stay at root. |
| `docs/` | Architecture, guides, deployment, and archived notes. |
| `frontend-nextjs/` | Next.js app (dashboard, trading UI, reports). Run: `cd frontend-nextjs && npm run dev`. |
| `requirements/` | Python dependency lists: `requirements.txt` (main), `requirements-dev.txt`, `requirements-llm.txt`, etc. |
| `scripts/` | One-off and automation: deploy, DB init, health checks, `start-dev.sh`, `start-services.sh`. |
| `src/` | Backend Python: FastAPI app, agents, strategies, LLM, data pipelines. |
| `tests/` | Pytest tests for the backend. |
| `wiki-content/` | Wiki / docs content (e.g. for GitHub wiki). |
| `alembic.ini` | Alembic config (migrations). Run from repo root. |
| `Makefile` | Common commands: `make dev`, `make test`, `make setup`, Docker targets. |
| `start.py` | Main backend entrypoint: `python3 start.py` or `make dev`. |
| `docker-compose-core.yml` | Core stack (API, frontend, DB, Redis, Celery). |
| `docker-compose-complete.yml` | Full stack including extra services. |

## Backend (`src/`)

- `src/main_refactored.py` – FastAPI app entry.
- `src/core/` – Config, logging, Celery, security.
- `src/api/` – REST endpoints and route modules.
- `src/llm/`, `src/strategies/`, `src/trading/`, `src/risk/`, etc. – Feature domains.

## Config and env

- **Secrets**: Never commit `.env` or `env.local`. Copy `config/env.example` to `.env` and fill in values.
- **API keys**: Use env vars (e.g. `ALPHA_VANTAGE_API_KEY`) or the placeholder file `config/api_keys_config.py` for local reference only.

## Quick commands

- Backend: `make dev` or `python3 start.py --reload`
- Frontend: `cd frontend-nextjs && npm run dev`
- Full stack: `./scripts/start-services.sh` or `make up`
- Tests: `make test`
