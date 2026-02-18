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

## Phase 2 In progress

- **Trading bots execution** – Wire UI to backend CRUD and start/pause/stop; connect agent panels to real or stub data; implement paper → live execution path. ([#3](https://github.com/massoudsh/Findash/issues/3), [#10](https://github.com/massoudsh/Findash/issues/10))
- **Dashboard real data** – Wire portfolio/accounts to backend with loading/error and timeout fallback. ([#8](https://github.com/massoudsh/Findash/issues/8))

---

## Backlog

- Technical page: Screener, Watchlist, Economic Calendar data wiring ([#9](https://github.com/massoudsh/Findash/issues/9))
- Development roadmap doc and phase checklist maintenance ([#11](https://github.com/massoudsh/Findash/issues/11))
- E2E or integration tests for critical flows
- Deploy: configure repo secrets and replace placeholder steps (e.g. AWS/ECS)

---

## Phase 3 (TBD)

- Scale and observability
- Production hardening and override patterns
- Broader test coverage and accessibility audit

---

For full issue list and history, see [docs/ISSUES_AND_ROADMAP.md](docs/ISSUES_AND_ROADMAP.md) and [GitHub Issues](https://github.com/massoudsh/Findash/issues).
