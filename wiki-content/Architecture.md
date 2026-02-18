# System Architecture

The Octopus Trading Platform is built on a modern, scalable microservices architecture designed for high-performance financial data processing.

## Architecture Overview

### Layer Diagram (Mermaid)

```mermaid
flowchart TB
    subgraph Client["👤 Client Layer"]
        WEB[Web App - Next.js]
        MOB[Mobile / API Clients]
        WS[WebSocket Client]
    end
    subgraph Gateway["🔒 API Gateway"]
        NGINX[NGINX / Kong]
    end
    subgraph App["⚡ Application Layer"]
        MKT[Market Data API]
        PORT[Portfolio API]
        RISK[Risk API]
        TRADE[Trading API]
        AUTH[Auth Service]
        WS_MGR[WebSocket Manager]
    end
    subgraph Intelligence["🧠 Intelligence Layer"]
        ORCH[Orchestrator]
        M1[M1 Data] 
        M2[M2 Warehouse]
        M3[M3 Realtime]
        M4[M4 Strategy]
        M5[M5 ML]
        M6[M6 Risk]
        M7[M7 Exec]
        M8[M8 Portfolio]
        M9[M9 Compliance]
        M10[M10 Backtest]
        M11[M11 Alt Data]
    end
    subgraph Data["🗄️ Data Layer"]
        PG[(PostgreSQL / TimescaleDB)]
        REDIS[(Redis)]
        KAFKA[Kafka]
    end
    WEB & MOB & WS --> NGINX --> App
    App --> ORCH
    ORCH --> M1 & M2 & M3 & M4 & M5 & M6 & M7 & M8 & M9 & M10 & M11
    M1 & M2 & M3 & M4 & M5 & M6 & M7 & M8 & M9 & M10 & M11 --> PG & REDIS & KAFKA
```

### ASCII Layer Sketch (Reference)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Web App    │  │ Mobile App  │  │  API Client │  │  WebSocket  │        │
│  │ (Next.js)   │  │  (React    │  │  (Python/   │  │   Client    │        │
│  │             │  │   Native)   │  │    JS/Go)   │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API GATEWAY LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     NGINX / Kong API Gateway                         │   │
│  │  • Load Balancing  • Rate Limiting  • SSL Termination  • Auth       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Backend                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │  Market   │  │ Portfolio │  │   Risk    │  │    AI     │        │   │
│  │  │  Data API │  │    API    │  │    API    │  │  Models   │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │  Trading  │  │ WebSocket │  │   Auth    │  │ Analytics │        │   │
│  │  │    API    │  │  Manager  │  │  Service  │  │  Service  │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTELLIGENCE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Intelligence Orchestrator (11 AI Agents)                │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │   │
│  │  │ M1  │ │ M2  │ │ M3  │ │ M4  │ │ M5  │ │ M6  │                   │   │
│  │  │Data │ │Data │ │Real │ │Strat│ │ ML  │ │Risk │                   │   │
│  │  │Coll.│ │Ware.│ │time │ │egy  │ │Model│ │Mgr  │                   │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                   │   │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                           │   │
│  │  │ M7  │ │ M8  │ │ M9  │ │ M10 │ │ M11 │                           │   │
│  │  │Exec.│ │Port.│ │Comp.│ │Back │ │Alt. │                           │   │
│  │  │Mgr  │ │Opt. │ │lianc│ │test │ │Data │                           │   │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │  PostgreSQL   │  │    Redis      │  │   Kafka       │                   │
│  │  TimescaleDB  │  │    Cache      │  │   Streaming   │                   │
│  │  • Market Data│  │  • Sessions   │  │  • Events     │                   │
│  │  • Users      │  │  • Cache      │  │  • Messages   │                   │
│  │  • Portfolios │  │  • Pub/Sub    │  │  • Real-time  │                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Client Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web App | Next.js 14 | Primary trading interface |
| Mobile App | React Native | Mobile trading (planned) |
| API Clients | Python/JS/Go SDKs | Programmatic access |
| WebSocket | Native WS | Real-time data streaming |

### 2. API Gateway Layer

- **NGINX**: Load balancing and SSL termination
- **Kong/APISIX**: API management and rate limiting
- **Authentication**: JWT token validation
- **Rate Limiting**: Request throttling per user/IP

### 3. Application Layer (FastAPI)

```
src/
├── api/
│   ├── endpoints/
│   │   ├── agents.py          # AI agent monitoring
│   │   ├── auth.py            # Authentication
│   │   ├── backtesting.py     # Strategy backtesting
│   │   ├── market_data.py     # Market data endpoints
│   │   ├── portfolio_api.py   # Portfolio management
│   │   ├── risk.py            # Risk analysis
│   │   ├── trading.py         # Trading operations
│   │   └── websocket.py       # WebSocket endpoints
│   └── routes/
│       └── ...                # Route definitions
├── core/
│   ├── config.py              # Configuration management
│   ├── cache.py               # Redis cache manager
│   ├── celery_app.py          # Celery configuration
│   └── security.py            # Security utilities
└── main_refactored.py         # FastAPI application
```

### 4. Intelligence Layer

See [[AI Agents]] for detailed documentation on the 11 AI agents.

### 5. Data Layer

| Database | Purpose | Features |
|----------|---------|----------|
| PostgreSQL | Primary storage | ACID compliance, relations |
| TimescaleDB | Time-series | Hypertables, compression |
| Redis | Caching | Sub-millisecond latency |
| Kafka | Streaming | Event-driven architecture |

---

## Data Flow Patterns

### Market Data Pipeline

```mermaid
flowchart LR
    EXT[External APIs] --> M1[M1 Data Collector]
    M1 --> M2[M2 Data Warehouse]
    M2 --> M3[M3 Real-time Processor]
    M1 --> CACHE[Cache/Storage]
    M2 --> HIST[Historical Analysis]
    M3 --> LIVE[Live Streaming]
    M2 --> M5[M5 ML Models]
    M2 --> M4[M4 Strategy Agent]
    M3 --> WS[WebSocket Clients]
```

*Text version:*
```
External APIs → Data Collector (M1) → Data Warehouse (M2) → Real-time Processor (M3)
                     ↓                         ↓                        ↓
               Cache/Storage              Historical Analysis      Live Streaming
                     ↓                         ↓                        ↓
                ML Models (M5)         Strategy Agent (M4)      WebSocket Clients
```

### Trading Decision Flow

```mermaid
sequenceDiagram
    participant MD as Market Data
    participant M4 as Strategy Agent
    participant M6 as Risk Manager
    participant M7 as Execution
    MD->>M4: Prices + Indicators
    M4->>M4: Technical + Fundamental + Fusion
    M4->>M6: Signals + Sizing
    M6->>M6: VaR, Limits, Position Sizing
    M6->>M7: Approved Order
    M7->>M7: Order Routing (TWAP/VWAP)
```

*Text version:*
```
Market Data → Strategy Agent (M4) → Signal Fusion → Risk Manager (M6) → Execution (M7)
      ↓              ↓                    ↓              ↓                    ↓
  Technical      Fundamental         Multi-Strategy   Position           Order
  Analysis       Analysis            Signals          Sizing             Routing
```

### Real-time Communication

```mermaid
flowchart LR
    MKT[Market Updates] --> REDIS[Redis Pub/Sub]
    REDIS --> WS[WebSocket Manager]
    WS --> CLIENTS[Connected Clients]
    REDIS --> CELERY[Celery Workers]
    CELERY --> BG[Background Processing]
```

---

## Scalability Design

### Horizontal Scaling

```mermaid
flowchart TB
    LB[Load Balancer]
    LB --> API1[API-1]
    LB --> API2[API-2]
    LB --> API3[API-3]
    API1 & API2 & API3 --> DB[(Database Primary)]
    DB --> R1[Replica 1]
    DB --> R2[Replica 2]
    DB --> R3[Replica 3]
```

*ASCII reference:*
```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  API-1   │    │  API-2   │    │  API-3   │
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    ┌──────────────┐
                    │  Database    │
                    │  (Primary)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Replica  │ │ Replica  │ │ Replica  │
        │    1     │ │    2     │ │    3     │
        └──────────┘ └──────────┘ └──────────┘
```

### Caching Strategy

```mermaid
flowchart LR
    REQ[Request] --> REDIS{Check Redis}
    REDIS -->|Hit| CACHED[Return Cached]
    REDIS -->|Miss| DB[Query Database]
    DB --> UPDATE[Update Cache]
    UPDATE --> RESP[Return Response]
```

---

## Security Architecture

### Authentication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant G as API Gateway
    participant F as FastAPI
    participant R as Protected Resource
    C->>G: Request
    G->>G: Rate Limit Check
    G->>G: IP Whitelist
    G->>F: JWT Validation
    F->>R: Access Resource
    R-->>C: Response
```

### Security Layers

| Layer | Protection |
|-------|------------|
| Network | Firewall, DDoS protection |
| Transport | TLS 1.3, HSTS |
| Application | JWT, OAuth2, rate limiting |
| Data | Encryption at rest, RLS |

---

## Monitoring Stack

```mermaid
flowchart TB
    APP[Applications] --> PROM[Prometheus\nMetrics]
    PROM --> GRAF[Grafana\nDashboards]
    GRAF --> ALERT[Alertmanager\nAlerts]
    APP --> LOGS[Structured Logs]
    LOGS --> ELK[ELK Stack]
    ELK --> KIB[Kibana]
    ALERT --> PAGER[PagerDuty / Slack]
```

*ASCII:*
```
Applications → Prometheus (Metrics) → Grafana (Dashboards) → Alertmanager (Alerts)
     ↓                                        ↓
Structured Logs → ELK Stack → Kibana         PagerDuty/Slack
```

### Key Metrics

- Request latency (p50, p95, p99)
- Error rates by endpoint
- Database connection pool
- Cache hit/miss ratio
- WebSocket connections
- Celery task queue length

---

## Next Steps

- [[AI Agents]] - Deep dive into the 11 AI agents
- [[API Reference]] - Complete API documentation
- [[Database]] - Database schema details
- [[Deployment]] - Production deployment guide
