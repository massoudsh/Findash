# Findash Development Roadmap

High-level roadmap for the Octopus Trading Platform (Findash). Detailed issues live on [GitHub](https://github.com/massoudsh/Findash/issues).

---

## Phase 1 ✅ Complete

- Core stack: FastAPI backend, Next.js frontend, PostgreSQL, Redis
- Command Center: Options tab, Trading Bots UI
- Dashboard: Overview / Portfolio tabs, account cards, glass/green theme, market status bar
- Agent panels: M1 (Data Collector), M4 (Strategy), M9 (Sentiment), M11 (Analysis)
- CI/CD: Backend lint (flake8, black, isort, mypy), frontend build, tests, Docker API build
- Ops: Docker core stack, optional LLM profile, healthcheck script
- Docs: LLM report models, env.example, README Docker section

---

## Phase 2 ✅ Complete

- **Trading bots** – Backend CRUD, start/pause/stop; persistence (JSON); Celery `run_bot_tick` execution; backend-health UX.
- **Dashboard real data** – Portfolios/trades/positions from DB when available; sample fallback; backend-health “Connect backend” banner.

---

## Phase 3 In progress

- **E2E tests** – Playwright for critical flows (app load, dashboard, trading bots, backend health).
- **Observability** – Use existing `/health` and `/health/detailed`; optional structured logging and metrics.
- **Production hardening** – Env validation, security headers, deploy configuration.

---

## Backlog

- Technical page: Screener, Watchlist, Economic Calendar data wiring ([#9](https://github.com/massoudsh/Findash/issues/9))
- Development roadmap doc and phase checklist maintenance ([#11](https://github.com/massoudsh/Findash/issues/11))
- E2E or integration tests for critical flows
- Deploy: configure repo secrets and replace placeholder steps (e.g. AWS/ECS)

---

## Phase 3 (in progress)

- E2E tests (Playwright), observability, production hardening — see Phase 3 section above.
- Broader test coverage and accessibility audit in backlog.

---

For full issue list and history, see [docs/ISSUES_AND_ROADMAP.md](docs/ISSUES_AND_ROADMAP.md) and [GitHub Issues](https://github.com/massoudsh/Findash/issues).
