# 🚀 Octopus Trading Platform – Developer Quick Start

This guide gives you a **single, clean path** to get the full stack running for local development.

---

## 1. Prerequisites

Install these before you start:

- **Python**: 3.11+ (recommended)
- **Docker**: latest stable (for Postgres/Redis/Celery and other services)
- **Node.js**: 18+ and matching **npm** (for the Next.js frontend)

> All commands below assume you start in the `Modules/` directory of this repo.

---

## 2. Backend Setup (FastAPI + Services)

### 2.1 Create and activate a virtual environment

```bash
cd Modules
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2.2 Install backend dependencies

Use the development requirements file so you get the full tooling (linting, tests, etc.):

```bash
pip install -r requirements-dev.txt
```

### 2.3 Provision infrastructure (database, Redis, Celery, monitoring)

Use the existing automation instead of starting containers manually. The Makefile (or equivalent script) is responsible for:

- Starting **PostgreSQL** and **Redis** (via Docker)
- Applying database migrations
- Starting **Celery** workers/beat processes
- Wiring monitoring components (Prometheus/Grafana) if configured

```bash
make setup
```

If your environment requires additional configuration (for example `.env` files or secrets), follow the notes in `Modules/README.md` and the security/architecture docs.

---

## 3. Frontend Setup (Next.js + Tailwind)

The trading UI lives in `frontend-nextjs/` and uses Next.js App Router with TypeScript and Tailwind CSS.

```bash
cd Modules/frontend-nextjs
npm install
npm run dev
```

By default the frontend is available at:

- **Next.js frontend**: `http://localhost:3000`

The frontend talks to the backend either through mock services (for isolated UI development) or via Next.js API routes that proxy to FastAPI.

---

## 4. Run the Full Stack

### 4.1 Start the backend

From the `Modules/` directory (and with your virtualenv activated):

```bash
make start-backend
```

This command is responsible for running the FastAPI application (and any supporting workers) in development mode. The standard ports are:

- **FastAPI backend**: `http://localhost:8000`
- **API docs** (OpenAPI/Swagger): `http://localhost:8000/docs`

### 4.2 Start the frontend

In a separate terminal:

```bash
cd Modules/frontend-nextjs
npm run dev
```

The main dashboard and trading UI are then reachable at:

- **Next.js frontend**: `http://localhost:3000`

---

## 5. What to Read Next

- **Platform Overview**: `Modules/README.md` – architecture, security model, and services.
- **Backend Architecture**: `Modules/BACKEND_ARCHITECTURE.md` – FastAPI, agents, and data flow.
- **Complete System Flow**: `Modules/COMPLETE_SYSTEM_FLOW.md` – end‑to‑end trading and monitoring flows.
- **Frontend Documentation**: `Modules/frontend-nextjs/README.md` – routes, components, and API communication.

Once you can hit `http://localhost:3000` and `http://localhost:8000/docs`, you have a complete local environment suitable for backend, frontend, and AI agent development.


