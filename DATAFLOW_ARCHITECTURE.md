# 🔄 Octopus Trading Platform - Dataflow Architecture

## Overview

The Octopus Trading Platform uses a **multi-layered, event-driven architecture** with intelligent agent orchestration. Data flows through several stages from external sources to end users, with caching, processing, and real-time distribution at each layer.

---

## 📊 **Complete Dataflow Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  External APIs → Data Collector (M1) → Validation → Cache      │
│  • Yahoo Finance    • Alpha Vantage    • Finnhub                 │
│  • CoinGecko       • Binance          • News APIs               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Real-time Processor (M3) → Intelligence Orchestrator          │
│  ├─ Stream Processing                                            │
│  ├─ Data Validation                                             │
│  ├─ Event Generation                                             │
│  └─ Task Distribution                                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  STORAGE      │ │   CACHING     │ │  REAL-TIME    │
│  LAYER        │ │   LAYER       │ │  DISTRIBUTION │
├───────────────┤ ├───────────────┤ ├───────────────┤
│ PostgreSQL    │ │ Redis Cache   │ │ WebSocket      │
│ TimescaleDB   │ │ • Market Data  │ │ Manager        │
│ • Historical  │ │ • Sessions    │ │ • Pub/Sub      │
│ • Trades      │ │ • Results     │ │ • Broadcast    │
│ • Portfolio  │ │ • Rate Limits │ │ • Channels     │
└───────────────┘ └───────────────┘ └───────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI AGENT LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Intelligence Orchestrator coordinates 11 agents:               │
│  • M4: Strategy Agent → Trading Signals                        │
│  • M5: ML Models → Predictions                                  │
│  • M6: Risk Manager → Risk Assessment                          │
│  • M7: Execution Manager → Trade Execution                     │
│  • M11: Alternative Data → Sentiment Analysis                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Next.js) ← WebSocket ← Real-time Updates            │
│  • Dashboard        • Charts      • Portfolio                    │
│  • Trading UI      • Alerts      • Analytics                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 **Detailed Dataflow Paths**

### **1. Market Data Flow (Real-time)**

```
External API (Yahoo Finance)
    ↓
Data Collector Agent (M1)
    ├─→ Cache (Redis): market_data:{symbol}:latest (5min TTL)
    ├─→ Database (PostgreSQL): Historical storage
    └─→ Real-time Processor (M3)
        ├─→ Validate & Process
        ├─→ Redis Pub/Sub: tasks:market_data:{symbol}
        ├─→ WebSocket Manager: Broadcast to subscribers
        └─→ Celery Task: update_market_data (async processing)
            └─→ Database: Store processed data
```

**Code Path:**
- `src/data_processing/market_data_tasks.py` - Celery tasks
- `src/realtime/websockets.py` - WebSocket broadcasting
- `src/core/cache.py` - Caching layer

### **2. Trading Order Flow**

```
User (Frontend)
    ↓ POST /api/trading/order
FastAPI Endpoint
    ↓
Intelligence Orchestrator
    ├─→ M1: Fetch current price
    ├─→ M6: Risk Manager (check limits)
    │   ├─→ Database: Get portfolio data
    │   └─→ Calculate position size
    ├─→ M4: Strategy Agent (generate signal)
    └─→ M7: Execution Manager
        ├─→ Broker API: Execute trade
        ├─→ Database: Transaction
        │   ├─→ INSERT trades
        │   ├─→ UPDATE positions
        │   └─→ UPDATE portfolio cash
        ├─→ Redis: Update cache
        └─→ WebSocket: Broadcast trade_update
            └─→ Frontend: Real-time notification
```

**Code Path:**
- `src/core/intelligence_orchestrator.py` - Agent coordination
- `src/trading/execution_manager.py` - Trade execution
- `src/portfolio/portfolio_manager.py` - Portfolio updates

### **3. WebSocket Real-time Updates**

```
Data Source (Market Data / Trades / Portfolio)
    ↓
Redis Pub/Sub Channel
    ├─→ market_data
    ├─→ trades
    ├─→ portfolio_updates
    └─→ system_health
    ↓
WebSocket Bridge
    ├─→ Subscribe to channels
    └─→ Forward to WebSocket clients
    ↓
WebSocket Manager
    ├─→ Active connections tracking
    ├─→ Channel subscriptions
    └─→ Message routing
    ↓
Frontend WebSocket Client
    └─→ Real-time UI updates
```

**Code Path:**
- `src/realtime/websocket_bridge.py` - Pub/Sub to WebSocket bridge
- `src/realtime/websockets.py` - WebSocket connection management
- `src/api/endpoints/unified_websocket.py` - WebSocket endpoints

### **4. Background Processing (Celery)**

```
Scheduled Tasks (Celery Beat)
    ├─→ Market Data Fetch (every 5 min)
    ├─→ Portfolio Updates (every 10 min)
    └─→ Data Cleanup (daily)
    ↓
Celery Worker
    ├─→ Process task
    ├─→ Update Redis cache
    ├─→ Store in database
    └─→ Publish to Pub/Sub
    ↓
Redis Pub/Sub
    └─→ WebSocket Bridge → Clients
```

**Code Path:**
- `src/data_processing/market_data_tasks.py` - Celery tasks
- `src/core/celery_app.py` - Celery configuration

---

## 🗄️ **Storage Layers**

### **PostgreSQL (Primary Database)**
- **Users**: Authentication, profiles
- **Portfolios**: Portfolio data, positions
- **Trades**: Trade history, execution records
- **Market Data**: Historical time-series (via TimescaleDB)
- **Risk Metrics**: Risk calculations, VaR
- **Audit Logs**: Compliance, security events

### **Redis (Cache & Pub/Sub)**
- **Cache Namespaces:**
  - `market_data:{symbol}:latest` - Latest prices (5min TTL)
  - `portfolio:{user_id}` - Portfolio cache (5min TTL)
  - `session:{session_id}` - User sessions
  - `rate_limit:{identifier}` - Rate limiting
  - `task:{task_id}:result` - Celery task results

- **Pub/Sub Channels:**
  - `market_data` - Market data updates
  - `trades` - Trade executions
  - `portfolio_updates` - Portfolio changes
  - `system_health` - System status
  - `sentiment` - Sentiment analysis updates

### **TimescaleDB (Time-series)**
- **Hypertables:**
  - `market_data` - High-frequency price data
  - `portfolio_snapshots` - Portfolio value over time
  - `risk_metrics` - Risk calculations over time

- **Continuous Aggregates:**
  - `market_data_1min` - 1-minute OHLCV aggregates
  - `market_data_5min` - 5-minute aggregates
  - `market_data_1hour` - Hourly aggregates

---

## 🤖 **AI Agent Coordination Flow**

```
Intelligence Orchestrator
    │
    ├─→ Stage 1: Data Collection (Parallel)
    │   ├─→ M1: Data Collector
    │   └─→ M3: Real-time Processor
    │
    ├─→ Stage 2: Analysis (Parallel)
    │   ├─→ M5: ML Models (predictions)
    │   └─→ M11: Alternative Data (sentiment)
    │
    ├─→ Stage 3: Strategy (Sequential)
    │   └─→ M4: Strategy Agent (signals)
    │
    ├─→ Stage 4: Risk Assessment
    │   └─→ M6: Risk Manager
    │
    └─→ Stage 5: Execution
        └─→ M7: Execution Manager
            └─→ M9: Compliance Engine
```

**Task Priority System:**
- **Priority 1**: Critical (data collection, risk checks)
- **Priority 2**: Important (analysis, predictions)
- **Priority 3**: Standard (strategy, execution)
- **Priority 4**: Low (reporting, cleanup)

---

## 📈 **Performance Characteristics**

### **Latency Targets**
- **API Response**: < 50ms (p95)
- **WebSocket Updates**: < 10ms
- **Database Queries**: < 100ms (p99)
- **Cache Hits**: < 1ms
- **Trade Execution**: < 200ms

### **Throughput**
- **API Requests**: 1000+ req/sec
- **WebSocket Connections**: 1000+ concurrent
- **Market Data Updates**: 10,000+ updates/sec
- **Database Writes**: 500+ writes/sec

### **Caching Strategy**
- **L1 (Memory)**: Hot data, 1min TTL
- **L2 (Redis)**: Frequently accessed, 5min TTL
- **L3 (Database)**: Historical, persistent

---

## 🔍 **Monitoring & Observability**

### **Metrics Flow**
```
Application Code
    ↓
Prometheus Metrics
    ├─→ trading_trades_total
    ├─→ api_response_time_seconds
    ├─→ websocket_connections
    └─→ cache_hit_ratio
    ↓
Grafana Dashboards
    └─→ Real-time visualization
```

### **Logging Flow**
```
Application Events
    ↓
Structured Logging
    ├─→ Request/Response logs
    ├─→ Error logs
    ├─→ Audit logs
    └─→ Performance logs
    ↓
Centralized Logging (Kibana/ELK)
    └─→ Search & analysis
```

---

## 🚀 **Key Dataflow Features**

### **1. Event-Driven Architecture**
- Redis Pub/Sub for decoupled communication
- WebSocket Bridge for real-time distribution
- Celery for async processing

### **2. Multi-Level Caching**
- In-memory cache for hot data
- Redis for distributed caching
- Database for persistent storage

### **3. Intelligent Agent Coordination**
- Priority-based task distribution
- Parallel processing where possible
- Sequential execution for dependencies

### **4. Real-time Processing**
- Sub-second latency for market data
- WebSocket streaming for live updates
- Event-driven updates to clients

### **5. Fault Tolerance**
- Graceful degradation (local cache if Redis down)
- Automatic fallback to alternative data sources
- Retry mechanisms for failed operations

---

## 📝 **Example: Complete End-to-End Flow**

**Scenario: User views AAPL price in real-time**

1. **Data Ingestion**
   - Yahoo Finance API → M1 Data Collector
   - Cache: `market_data:AAPL:latest` (Redis)

2. **Processing**
   - M3 Real-time Processor validates data
   - Publishes to `market_data` Pub/Sub channel

3. **Distribution**
   - WebSocket Bridge subscribes to channel
   - Forwards to WebSocket Manager
   - Broadcasts to subscribed clients

4. **Client Update**
   - Frontend WebSocket receives update
   - React state updates
   - UI re-renders with new price

5. **Storage**
   - Celery task stores in database (async)
   - TimescaleDB hypertable for time-series

**Total Latency: < 100ms from API to UI**

---

## 🔐 **Security in Dataflow**

- **Authentication**: JWT tokens validated at API gateway
- **Authorization**: Role-based access control
- **Rate Limiting**: Redis-based sliding window
- **Data Encryption**: TLS for transport, encryption at rest
- **Audit Trail**: All operations logged to database

---

This architecture ensures **high performance, scalability, and real-time responsiveness** while maintaining **data integrity and security** throughout the entire dataflow pipeline.

