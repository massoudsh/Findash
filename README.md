<div align="center">

<br/>

<img src="https://raw.githubusercontent.com/massoudsh/Findash/main/frontend-nextjs/public/logo.png" alt="Findash" width="72" height="72" />

# Findash

**A full-stack fintech dashboard built for the Iranian market.**
Real-time market data · Portfolio tracking · Risk management · Persian UI (RTL) · ZarinPal payments

<br/>

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **Live demo · screenshots · GIF walkthrough coming soon**
> *Clone → configure `.env` → `docker compose up` — that's it.*

</div>

---

## What is Findash?

Findash is an open-source **Iranian fintech dashboard** that consolidates everything a trader or investor needs in one clean, Persian-first interface:

- Live prices for gold, currencies, crypto, and housing
- Portfolio P&L with trade history
- Real-time risk gauge (VaR, drawdown, beta)
- ZarinPal payment integration (create → redirect → verify)
- AI-powered analysis via 11 orchestrated agents
- Full RTL support with the Dana font

---

## Demo

<div align="center">

<!-- Replace with your actual GIF/screenshot once available -->
```
┌─────────────────────────────────────────────────────┐
│  📊 Dashboard  💼 Portfolio  📈 Markets  ⚠️ Alerts  │
│ ─────────────────────────────────────────────────── │
│  Portfolio Value     ↑ 12.4%    Risk Level: Medium  │
│  ₿ BTC  47,200 $     طلا  3,850,000 ت               │
│  دلار   58,200 ت     سکه  42,000,000 ت              │
│  ─────────────────────────────────────── ──────────  │
│  [Chart] ████████████░░░  Sharpe: 1.42              │
└─────────────────────────────────────────────────────┘
```

*Full animated demo GIF will be added here.*

</div>

---

## Features

| Category | Highlights |
|---|---|
| 📊 **Dashboard** | Live ticker, portfolio overview, cash-flow charts, asset allocation |
| 💼 **Portfolio** | Trade tracker, P&L, Iranian physical assets (gold, silver, housing, crypto) |
| ⚡ **Realtime** | WebSocket market feed with auto-reconnect hook |
| ⚠️ **Risk Engine** | Live risk gauge, VaR, max drawdown, portfolio beta |
| 🧠 **AI Agents** | 11-agent orchestrator for data collection, analysis, strategy & reports |
| 💳 **Payments** | Full ZarinPal cycle — create, redirect, callback, verify, history |
| 🔐 **Auth** | JWT-based sign-in / sign-up with route protection |
| 🌐 **Persian-first** | RTL layout, Jalali dates, Toman/Dollar toggle, Dana font |
| 📱 **Mobile-ready** | Mobile-first design, max 5 nav items, readable card density |

---

## Architecture

```
User
 │
 ├─► Next.js 15 (Frontend · port 3003)
 │       │ REST / WebSocket
 │       ▼
 └─► FastAPI (Backend · port 8011)
         ├─► Auth (JWT)
         ├─► ZarinPal Payment
         ├─► AI Agents (×11)
         ├─► PostgreSQL / TimescaleDB
         └─► Redis Cache
```

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS, Shadcn UI, Recharts |
| Backend | FastAPI, Python 3.10+, Celery |
| Database | PostgreSQL 14+, TimescaleDB |
| Cache / Queue | Redis, Celery Workers |
| Realtime | WebSocket (custom hook) |
| AI / ML | PyTorch, scikit-learn, 11 orchestrated agents |
| Payments | ZarinPal (sandbox + production) |
| Monitoring | Prometheus (9090), Grafana (3001) |

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/massoudsh/Findash.git
cd Findash
cp .env.example .env          # fill in your values
docker compose -f docker-compose-core.yml up --build -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3003 |
| Backend API | http://localhost:8011 |
| Swagger docs | http://localhost:8011/docs |
| Grafana | http://localhost:3001 |

```bash
# View logs
docker compose -f docker-compose-core.yml logs -f

# Stop
docker compose -f docker-compose-core.yml down
```

---

### Option B — Local (without Docker)

**Backend**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements/requirements.txt
python3 start.py --reload
```

**Frontend**

```bash
cd frontend-nextjs
npm install
npm run dev        # http://localhost:3003
```

---

## Environment Variables

Create a `.env` file at the project root:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_db

# Cache
REDIS_URL=redis://localhost:6379/0

# Security  (change before production!)
SECRET_KEY=change-this-secret-key-min-32-chars
JWT_SECRET_KEY=change-this-jwt-secret-min-32-chars

# API
NEXT_PUBLIC_API_URL=http://localhost:8011
APP_BASE_URL=http://localhost:3003

# ZarinPal
ZARINPAL_MERCHANT_ID=your-zarinpal-merchant-id
```

> **Sandbox mode:** use ZarinPal's test merchant ID for local development. Switch to a real merchant for production.

---

## Payment Flow

```
POST /create  →  Redirect to ZarinPal  →  GET /callback  →  POST verify  →  ✅ / ❌
```

**Backend routes**

| Route | Description |
|---|---|
| `POST /api/payment/zarinpal/create` | Create payment order |
| `GET  /api/payment/zarinpal/callback` | Handle gateway return & verify |
| `GET  /api/payment/zarinpal/status/{id}` | Order status |
| `GET  /api/payment/zarinpal/history` | User payment history |

**Frontend pages**

| Path | Description |
|---|---|
| `/payment/checkout` | Plan selection & payment initiation |
| `/payment/callback/zarinpal` | Gateway return bridge |
| `/payment/success` | Success confirmation |
| `/payment/failed` | Failure page |

Run the DB migration once:

```bash
psql -d trading_db -f database/schemas/payment_orders.sql
```

---

## Project Structure

```
Findash/
├── frontend-nextjs/
│   └── src/
│       ├── app/
│       │   ├── dashboard/          # Main dashboard
│       │   ├── portfolio/          # Portfolio & trades
│       │   ├── auth/               # Sign-in / sign-up
│       │   └── payment/            # Checkout, success, failed
│       ├── components/
│       │   ├── realtime/           # WebSocket feed
│       │   ├── portfolio/          # Trade tracker, P&L
│       │   └── risk/               # Risk gauge
│       └── lib/
│           └── hooks/              # use-market-ws, etc.
├── src/
│   ├── main_refactored.py          # FastAPI app entry
│   ├── api/endpoints/              # payment, auth, assets ...
│   └── core/config.py              # App settings
└── database/
    └── schemas/                    # SQL migrations
```

---

## Roadmap

- [ ] Subscription plan management
- [ ] KYC / financial identity verification
- [ ] Rial wallet integration
- [ ] PDF report generation (Persian)
- [ ] Push & SMS alerts
- [ ] Risk policy engine
- [ ] Live Iranian market data (TGJU, Nobitex)
- [ ] Admin panel for transactions & users

---

## Contributing

1. Fork the repository and create a feature branch
2. Keep changes small and focused
3. Test main user flows manually before opening a PR
4. For payment or auth changes — cover error scenarios too

---

## License

MIT © [massoudsh](https://github.com/massoudsh) — see [`LICENSE`](LICENSE) for details.
