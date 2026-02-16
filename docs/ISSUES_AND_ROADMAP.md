# Findash GitHub Issues & Roadmap

Use this doc to **close** the existing open issue and **create** new issues aligned with the current platform design and timeline.

---

## Close existing issue

**Issue #1: Phase 1: Core Features Development**

- **Action:** Close as completed (or “Done for Phase 1”).
- **Comment to add when closing (optional):**
  ```
  Phase 1 scope completed per current design: core app (FastAPI + Next.js), data models,
  Trading Center (Market, Live Trading, Trading Bots), dashboard with account cards,
  agent panels (M1/M4/M9/M11), and CI/CD workflow. Remaining work tracked in new issues below.
  ```

---

## New issues to create (design + timeline)

Create these in the Findash repo (e.g. **Issues → New issue**). Copy title and body as needed.

---

### 1. **CI/CD: Harden and optional deploy**

**Title:** `ci: Harden CI/CD and add optional deploy`

**Body:**

```markdown
## Summary
- CI/CD workflow was updated: backend lint (flake8, black, isort, mypy), frontend lint (Next.js), tests (pytest + Postgres/Redis), and Docker API build.
- Deploy jobs are placeholders until AWS/ECS (or other target) secrets are set.

## Tasks
- [ ] Remove `|| true` from lint steps once codebase passes
- [ ] Enable strict mypy (remove `continue-on-error`) when types are fixed
- [ ] Add frontend build job (e.g. `npm run build`) to CI
- [ ] Configure repo secrets for deploy (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) and replace deploy placeholder steps

## Priority
Medium

## Timeline
Next 1–2 sprints
```

---

### 2. **Dashboard: Polish and green fintech theme**

**Title:** `ui: Dashboard account cards and green fintech theme`

**Body:**

```markdown
## Summary
Dashboard has glass-style account cards and a green/emerald fintech background. Remaining polish and consistency.

## Tasks
- [ ] Ensure account cards are responsive and accessible on small screens
- [ ] Align remaining dashboard cards (e.g. Financial Summary) with glass/green theme if desired
- [ ] Add loading/error states for account balances when wired to API

## Priority
Low

## Timeline
Backlog / when wiring real data
```

---

### 3. **Trading Center: Bots and agents**

**Title:** `feat: Trading bots execution and agent panels integration`

**Body:**

```markdown
## Summary
Trading Bots UI (strategy types, risk params, agent sources M4/M9/M11/M6) and agent panels (Data Collector M1, Strategy M4, Sentiment M9, Analysis M11) are in place. Backend and execution still to be wired.

## Tasks
- [ ] Add or align API endpoints for bot CRUD and config (risk, execution mode, agent sources)
- [ ] Wire Data Collector / Strategy / Sentiment panels to real or stub backend
- [ ] Implement “run on platform” execution path (paper then live) for bots using agent signals

## Priority
High

## Timeline
Phase 2 – 2–3 sprints
```

---

### 4. **LLM: Open-source and free tier only**

**Title:** `docs/llm: Document open-source and free LLM usage`

**Body:**

```markdown
## Summary
LLM report generation uses only open-source or free options (Falcon TGI, FinGPT local, HuggingFace free token). No paid API keys.

## Tasks
- [ ] Keep docs/llm-report-models.md and env.example in sync with any new LLM options
- [ ] Optional: add a small “LLM status” indicator in UI when FALCON_TGI_URL / FINGPT_LOCAL_URL / HF_TOKEN are set

## Priority
Low

## Timeline
Backlog
```

---

### 5. **Docker: Core stack and optional LLM**

**Title:** `ops: Docker core stack and optional LLM profile`

**Body:**

```markdown
## Summary
- docker-compose-core.yml: API, frontend, db, redis, celery-worker, celery-beat, prometheus, grafana. Redis host port 6380 by default to avoid conflict with local Redis.
- LLM services (TGI Falcon, FinGPT inference) are under profile `llm` (optional).

## Tasks
- [ ] Document in README how to run core only vs with LLM profile
- [ ] Add healthcheck or smoke test script for core stack
- [ ] Consider adding docker-compose override example for production

## Priority
Medium

## Timeline
1 sprint
```

---

### 6. **Phase 2: Backend and data**

**Title:** `Phase 2: Backend APIs and data wiring`

**Body:**

```markdown
## Overview
Wire frontend to backend for trading bots, agent status/signals, and dashboard data.

## Tasks
- [ ] Trading bots API (create, list, update, pause/start, risk and agent source config)
- [ ] Agent status/signals endpoints for M1, M4, M9, M11 panels
- [ ] Dashboard portfolio and account summary from API
- [ ] Tests and OpenAPI docs for new endpoints

## Priority
High

## Estimated effort
3–4 sprints
```

---

## Suggested order

1. **Close** issue #1 with the comment above.
2. **Create** the 6 new issues (copy titles/bodies from this file).
3. **Label** (e.g. `ci`, `ui`, `feat`, `docs`, `ops`, `phase-2`) and set milestone if you use them.

---

*Last updated: 2026-02-16 (after CI/CD fix and push).*
