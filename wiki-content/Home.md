# Octopus Trading Platform – Findash Wiki

<p align="center">
  <strong>Elaborate project wiki: onboarding, architecture, development, and operations</strong>
</p>

<p align="center">
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"></a>
  <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"></a>
</p>

---

## Start here

| If you want to… | Go to |
|------------------|--------|
| **Get the platform running quickly** | [[Onboarding]] |
| **Understand the repo layout** | [[Project Structure]] |
| **Full installation options** | [[Getting Started]] |
| **Configure env and API keys** | [[Configuration]], [[Data Sources]] |
| **See how the system is built** | [[Architecture]], [[AI Agents]] |
| **Fix common problems** | [[Troubleshooting]] |
| **Deploy to production** | [[Deployment]] |
| **Use or extend the API** | [[API Reference]] |
| **Work on the frontend** | [[Frontend]] |
| **Contribute** | [[Contributing]] |

---

## Overview

The **Octopus Trading Platform (Findash)** is an AI-powered trading system that combines real-time market data, analytics, machine learning, and automated trading in a single interface. The backend coordinates **11 AI agents** (M1–M11) for data collection, strategy, risk, sentiment, and reporting.

### Platform flow (high-level)

```mermaid
flowchart LR
    subgraph Users
        U[👤 Trader]
    end
    subgraph Frontend
        D[📊 Dashboard]
        T[💹 Trading]
        P[📈 Portfolio]
    end
    subgraph Backend
        API[🔒 API Gateway]
        FAST[⚡ FastAPI]
        ORCH[🧠 11 AI Agents]
    end
    subgraph Data
        DB[(🗄️ PostgreSQL)]
        REDIS[(⚡ Redis)]
    end
    subgraph External
        MKT[📈 Market Data]
        NEWS[📰 News/Social]
    end
    U --> D & T & P
    D & T & P --> API --> FAST --> ORCH
    ORCH --> DB & REDIS
    ORCH --> MKT & NEWS
```

### User journey

```mermaid
flowchart TD
    A[Landing / Login] --> B[Dashboard]
    B --> C{User Action}
    C -->|View portfolio| D[Portfolio & Positions]
    C -->|Trade| E[Command Center]
    C -->|Analyze| F[Market Data & Charts]
    C -->|Automate| G[Trading Bots]
    C -->|Risk| H[Risk & Analytics]
    D --> I[Real-time prices & PnL]
    E --> J[Order entry → Execution]
    F --> K[Technical + Sentiment]
    G --> L[Bot config → Signals]
    H --> M[VaR, Stress, Reports]
    I & J & K & L & M --> B
```

---

## Wiki pages (full index)

### Onboarding & setup
| Page | Description |
|------|-------------|
| [[Onboarding]] | Step-by-step first-time setup and “where to go next” |
| [[Getting Started]] | All installation methods (local, Docker, Makefile) |
| [[Configuration]] | Environment variables, security, Docker, frontend config |
| [[Project Structure]] | Repo layout, backend folders, quick commands |

### Architecture & data
| Page | Description |
|------|-------------|
| [[Architecture]] | System layers, data flow, scaling, auth, monitoring |
| [[AI Agents]] | The 11 agents (M1–M11), roles, and collaboration |
| [[Database]] | Schema, entity-relationship, migrations |
| [[Data Sources]] | Market/news providers, API keys, free tiers |

### Development
| Page | Description |
|------|-------------|
| [[API Reference]] | REST API overview and request lifecycle |
| [[Frontend]] | Next.js app structure, pages, components |
| [[Contributing]] | How to contribute to the project |

### Operations
| Page | Description |
|------|-------------|
| [[Deployment]] | Production and Docker deployment |
| [[Troubleshooting]] | Common issues and fixes |

---

## Features overview

### Core trading
- **Dashboard** – Portfolio overview, watchlists, live data  
- **Real-time market data** – Prices, orderbook, tick data  
- **Options** – Options chain and strategies  
- **Trading bots** – Automated trading with configurable rules  
- **Portfolio** – Multi-asset tracking and optimization  
- **Market analysis** – Technical, fundamental, on-chain tools  

### AI & ML
- **Price prediction** – Pre-trained forecasting models  
- **Sentiment** – News and social sentiment analysis  
- **Strategy optimization** – Backtesting and parameter tuning  
- **Insights** – Automated market recommendations  

### Risk & analytics
- **Risk** – VaR, stress testing, correlation  
- **Backtesting** – Historical strategy testing  
- **Reports** – Trading analytics  
- **Data explorer** – Advanced querying  

---

## Tech stack (overview)

```mermaid
flowchart TB
    subgraph Frontend["🖥️ Frontend"]
        N[Next.js 14]
        TS[TypeScript]
        TW[Tailwind]
        SH[Shadcn UI]
        RC[Recharts]
    end
    subgraph Backend["⚙️ Backend"]
        FA[FastAPI]
        PY[Python 3.10+]
        CE[Celery]
    end
    subgraph Data["🗄️ Data & Infra"]
        PG[(PostgreSQL)]
        TDB[TimescaleDB]
        RD[(Redis)]
        DOCK[Docker]
    end
    N --> FA
    FA --> PG & TDB & RD
    CE --> PG & RD
```

| Layer | Technologies |
|-------|---------------|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Shadcn UI, Recharts |
| **Backend** | FastAPI, Python 3.10+, SQLAlchemy, Celery, Redis |
| **Data** | PostgreSQL, TimescaleDB, Docker, Prometheus, Grafana |

---

## Quick start (copy-paste)

```bash
git clone https://github.com/massoudsh/Findash.git
cd Findash
cp config/env.example .env
# Set SECRET_KEY and JWT_SECRET_KEY in .env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements/requirements.txt
python3 start.py --reload
# Second terminal:
cd frontend-nextjs && npm install && npm run dev
# Frontend: http://localhost:3000  |  API: http://localhost:8000  |  Docs: http://localhost:8000/docs
```

---

## Publishing this wiki

The wiki lives in the **wiki-content/** folder. To publish to GitHub Wiki, see [wiki-content/PUBLISH_WIKI.md](https://github.com/massoudsh/Findash/blob/main/wiki-content/PUBLISH_WIKI.md).

---

## Support

- **Issues**: [GitHub Issues](https://github.com/massoudsh/Findash/issues)  
- **Repo**: [GitHub Repository](https://github.com/massoudsh/Findash)  
- **API docs**: http://localhost:8000/docs (when backend is running)

<p align="center"><strong>Octopus Trading Platform (Findash) – Wiki</strong></p>
