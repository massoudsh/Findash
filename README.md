# Octopus Trading Platform

<div align="center">

**AI-powered trading platform with real-time analytics and 11 orchestrated agents**

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

[Features](#-features) · [Agents](#-ai-agents) · [Installation](#-installation) · [Docs](#-documentation)

</div>

---

## Overview

Octopus is an AI-powered trading system that combines real-time market data, analytics, ML models, and automated trading in a single interface. The backend coordinates **11 AI agents** (M1–M11) for data collection, strategy, risk, sentiment, and reporting.

---

## System architecture

```mermaid
graph TB
    subgraph UI["Frontend"]
        F[Next.js]
        F --> D[Dashboard]
        F --> T[Trading Center]
        F --> A[Analytics]
    end
    
    subgraph Backend["Backend"]
        API[FastAPI]
        WS[WebSocket]
        API --> M[Orchestrator & Agents]
    end
    
    subgraph Data["Data"]
        PG[(PostgreSQL)]
        R[(Redis)]
    end
    
    UI --> API
    API --> PG
    API --> R
    M --> PG
```

---

## AI Agents

The platform uses **11 orchestrated agents** (M1–M11) with distinct roles and personas. Each has a character name used across the UI (Trading Center, Risk, Reports, etc.).

| ID  | Character | Role | Responsibility |
|-----|-----------|------|----------------|
| M1  | **Nexus** | Data Collection | Market data, news, alternative data pipelines |
| M2  | **Vault** | Data Warehouse | Storage, validation, historical datasets |
| M3  | **Pulse** | Real-time Processor | Streaming data, live analytics, alerts |
| M4  | **Atlas** | Strategy Agent | Signals, strategy execution, backtesting |
| M5  | **Neuron** | ML Models | Prediction, classification, deep learning |
| M6  | **Guardian** | Risk Management | VaR, position sizing, compliance |
| M7  | **Oracle** | Price Prediction | Time-series and price forecasting |
| M8  | **Shadow** | Paper Trading | Simulated execution, paper portfolio |
| M9  | **Echo** | Market Sentiment | News and social sentiment analysis |
| M10 | **Chronicle** | Backtesting | Historical testing, strategy validation |
| M11 | **Lens** | Visualization | Charts, dashboards, report insights |

Pipeline flow: **Data (M1, M3, M9)** → **ML & prediction (M5, M7)** → **Risk & strategy (M6, M4)** → **Backtest & viz (M10, M11)**. The orchestrator routes tasks via `submit_task()` and runs full pipelines via `coordinate_pipeline()`.

---

## Features

- **Dashboard** – Portfolio overview, market watchlists, live data
- **Trading Center** – Order entry, positions, bots, real-time data
- **Options** – Options chain and strategies
- **Portfolio & Risk** – Multi-asset tracking, VaR, stress tests
- **Strategies & Backtesting** – Strategy builder and historical backtests
- **AI Models** – Training, predictions, insights
- **Reports & Visualization** – AI-powered reports and charts
- **Platform search** – Search anything (pages, commands) via ⌘K

---

## Installation

### Prerequisites

- Node.js 18+, Python 3.10+, PostgreSQL 14+, Redis (optional)

### Quick start

```bash
# Clone
git clone https://github.com/massoudsh/Findash.git
cd Findash

# Backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend-nextjs
npm install
```

### Run

```bash
# Terminal 1 – backend
python3 start.py --reload

# Terminal 2 – frontend
cd frontend-nextjs && npm run dev
```

- **Frontend:** http://localhost:3000  
- **API docs:** http://localhost:8000/docs  

---

## Documentation

- **API:** Swagger at `/docs`, ReDoc at `/redoc`
- **Architecture:** See `docs/ARCHITECTURE_DIAGRAMS.md` and `docs/orchestrator-architecture.md` for detailed diagrams

---

## Tech stack

| Layer      | Stack |
|-----------|--------|
| Frontend  | Next.js 14, TypeScript, Tailwind CSS, Shadcn UI |
| Backend   | FastAPI, Python 3.10+ |
| Data      | PostgreSQL (TimescaleDB), Redis |
| ML/AI     | PyTorch, scikit-learn, Celery workers |

---

## License

MIT – see [LICENSE](LICENSE).
