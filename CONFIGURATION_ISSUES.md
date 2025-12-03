# Configuration Issues & Fixes

## 🔴 Critical Issues Found

### 1. **DUPLICATE API PREFIX: `/api/market-data`**

**Problem:**
- `professional_market_data.py` has prefix: `/api/market-data`
- `market_data_workflow.py` has prefix: `/api/market-data`
- Both are registered in `main_refactored.py`, causing route conflicts!

**Location:**
- `Modules/src/api/endpoints/professional_market_data.py:20`
- `Modules/src/api/endpoints/market_data_workflow.py:27`
- `Modules/src/main_refactored.py:135, 159`

**Fix:**
Change one of the prefixes. Recommended:
- Keep `professional_market_data.py` as `/api/market-data`
- Change `market_data_workflow.py` to `/api/market-data/workflow`

### 2. **DUPLICATE AGENTS ROUTER**

**Problem:**
- `agents.py` has prefix: `/api/agents`
- `agents_v2.py` has prefix: `/api/agents` (duplicate!)
- Only `agents.py` is registered, but `agents_v2.py` exists with same prefix

**Location:**
- `Modules/src/api/endpoints/agents.py:35`
- `Modules/src/api/endpoints/agents_v2.py:23`

**Fix:**
- Remove or rename `agents_v2.py` if not needed
- Or change prefix to `/api/agents/v2` if it's a versioned API

### 3. **Celery Worker Missing Queues**

**Problem:**
Worker command only listens to: `default,data-processing,prediction,backtesting,portfolio`

But `task_routes` defines additional queues:
- `ml_training`
- `risk`
- `strategies`
- `analytics`
- `generative`
- `llm`

**Location:**
- `Modules/docker-compose-core.yml:111`
- `Modules/src/core/celery_app.py:49-60`

**Fix:**
Update celery-worker command to include all queues:
```yaml
command: celery -A src.core.celery_app worker -l info -Q default,data-processing,market_data,ml_training,prediction,backtesting,portfolio,risk,strategies,analytics,generative,llm
```

### 4. **Celery App Name Mismatch**

**Problem:**
- Celery app name: `"quantum_trading_matrix"`
- Platform name: `"Octopus Trading Platform"`

**Location:**
- `Modules/src/core/celery_app.py:13`

**Fix:**
Change to: `"octopus_trading_platform"` for consistency

### 5. **Multiple Market Data Routers**

**Problem:**
Multiple market data routers with overlapping functionality:
1. `market_data.router` (from routes) - prefix `/api/v1/market-data`
2. `market_router` (professional_market_data) - prefix `/api/market-data`
3. `market_data_workflow_router` - prefix `/api/market-data` (CONFLICTS!)
4. `simple_data_router` - prefix `/api`

**Location:**
- `Modules/src/main_refactored.py:136, 135, 159, 144`

**Fix:**
Consolidate or clearly separate:
- `/api/v1/market-data` - Legacy/v1 API
- `/api/market-data` - Professional API
- `/api/market-data/workflow` - Workflow API (rename)
- `/api/real-market-data` - Simple real data (rename)

## ⚠️ Medium Priority Issues

### 6. **WebSocket Endpoints**

**Problem:**
Multiple WebSocket implementations:
- `websocket.router` - prefix `/api/v1/websocket`
- `ws_realtime_router` - prefix `/api/ws`
- Direct endpoint `/ws`

**Location:**
- `Modules/src/main_refactored.py:139, 156, 162`

**Fix:**
Consolidate or document which to use:
- `/ws` - Main WebSocket endpoint
- `/api/v1/websocket` - Legacy API
- `/api/ws` - Real-time updates

### 7. **Missing Queue in Task Routes**

**Problem:**
`market_data.*` tasks route to `data_processing` queue, but worker command uses `data-processing` (with hyphen)

**Location:**
- `Modules/src/core/celery_app.py:51`
- `Modules/docker-compose-core.yml:111`

**Fix:**
Ensure consistency: Use `data_processing` (underscore) everywhere

## 📝 Recommended Fixes

### Priority 1 (Critical - Route Conflicts)
1. Fix duplicate `/api/market-data` prefix
2. Remove or rename `agents_v2.py`
3. Update Celery worker queues

### Priority 2 (Consistency)
4. Rename Celery app to match platform
5. Consolidate market data routers
6. Document WebSocket endpoints

### Priority 3 (Cleanup)
7. Remove unused routers if any
8. Standardize naming conventions
9. Add route documentation

