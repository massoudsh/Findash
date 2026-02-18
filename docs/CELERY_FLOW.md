# Celery flow – how to see it

This doc explains **how a Celery task flows** through the stack and **how you can see it** (logs, API, Flower, metrics).

---

## 1. End-to-end flow (high level)

```
  TRIGGER                    BROKER                 WORKER                    RESULT
  ───────                    ──────                  ──────                    ──────

  Option A: Celery Beat (schedule)
  ┌──────────────┐           ┌──────────────┐       ┌──────────────┐         ┌──────────────┐
  │ celery-beat  │  ──────▶  │ Redis        │  ───▶  │ celery-worker│  ───▶   │ Redis        │
  │ (scheduler)   │  publish  │ (broker)     │  consume │ (runs task)  │  store  │ (result      │
  │ every 5s etc │           │              │         │              │         │  backend)    │
  └──────────────┘           └──────────────┘       └──────────────┘         └──────────────┘
                                                                                     │
  Option B: API (on-demand)                                                           │
  ┌──────────────┐           same broker               same worker                     │
  │ FastAPI      │  .delay() ────────────────────────────────────────────────▶  task_id, result
  │ /market-data/│           or .apply_async()                                      │
  │ fetch/async  │                                                                   ▼
  └──────────────┘                                                    Poll GET /task/{id} or
                                                                       wait .get(timeout=…)
```

- **Trigger**: either **Celery Beat** (schedule in `src/core/celery_app.py`) or **API** (e.g. `POST /api/v1/market-data/fetch/async`).
- **Broker**: Redis (e.g. `CELERY_BROKER_URL=redis://redis:6379/0`). Tasks are queued here.
- **Worker**: `celery -A src.core.celery_app worker -l info -Q default,data_processing,...` consumes from Redis and runs the task code.
- **Result**: Stored in Redis (result backend). You can see it via `task_id` (API response or `AsyncResult(task_id).get()`), or in Flower/Prometheus.

---

## 2. Example flows you can see

### A. Beat-scheduled task (no API call)

**Example: BTC price every 5 seconds**

1. **Schedule** (where the flow starts):
   - File: `src/core/celery_app.py`
   - `beat_schedule` → `'fetch-btc-price-realtime'` → task `'market_data.fetch_btc_price_realtime'` every `5.0` seconds.

2. **Task implementation** (what the worker runs):
   - File: `src/data_processing/market_data_tasks.py`
   - Function: `fetch_btc_price_realtime(self)` (decorated with `@celery_app.task(name='market_data.fetch_btc_price_realtime', bind=True)`).

3. **How to see the flow**
   - **Worker logs** (e.g. Docker):
     ```bash
     docker logs -f octopus-celery-worker
     ```
     You should see lines like: task received → `fetch_btc_price_realtime` → "Stored BTC price in Redis" / "Published BTC price to Redis pub/sub".
   - **Beat logs** (scheduler sending the task):
     ```bash
     docker logs -f octopus-celery-beat
     ```
   - **Redis**: Key `btc_price:latest` is written by the task (see `market_data_tasks.py`). You can inspect with `redis-cli` if needed.
   - **Prometheus/Grafana**: If celery-metrics and Prometheus are running, task counts and duration are exposed (see `docs/archive/CELERY_MONITORING_SETUP.md`).

So the **flow you see** is: Beat (every 5s) → Redis (queue) → Worker (runs `fetch_btc_price_realtime`) → Redis (result + `btc_price:latest`) + logs.

### B. API-triggered task (on-demand)

**Example: Fetch market data for a symbol**

1. **Trigger** (where the flow starts):
   - Endpoint: `POST /api/v1/market-data/fetch/async` (or the route under your API prefix).
   - File: `src/api/endpoints/unified_market_data.py` (or `market_data_workflow.py`).
   - Code: `fetch_single_market_data.delay(symbol, force_refresh)` or `fetch_multiple_market_data.delay(symbols, ...)`.

2. **Task implementation** (what the worker runs):
   - File: `src/data_processing/market_data_tasks.py`
   - Functions: `fetch_single_market_data`, `fetch_multiple_market_data` (decorated with `@celery_app.task(name='market_data.fetch_single', bind=True)` etc.).

3. **How to see the flow**
   - **API response**: Returns `task_id`. Example:
     ```json
     { "status": "accepted", "task_id": "<uuid>", "symbols": ["AAPL"], "message": "Market data fetch task queued" }
     ```
   - **Worker logs**:
     ```bash
     docker logs -f octopus-celery-worker
     ```
     After calling the API you should see the corresponding task received and logs from `fetch_single_market_data` / `fetch_multiple_market_data`.
   - **Result**: Stored in Redis result backend. You can get result by task_id (e.g. in Python: `AsyncResult(task_id).get(timeout=60)`), or use Flower (see below) to see the task and its result.

So the **flow you see** is: HTTP POST → FastAPI calls `.delay()` → Redis (queue) → Worker (runs task) → Redis (result) + logs; client can use `task_id` to poll or wait for result.

---

## 3. Key files (trace the flow)

| Role | File |
|------|------|
| Celery app, beat schedule, task routes | `src/core/celery_app.py` |
| Market data tasks (fetch single/multiple, BTC, watchlist, cleanup) | `src/data_processing/market_data_tasks.py` |
| API that triggers fetch (returns task_id) | `src/api/endpoints/unified_market_data.py` (e.g. `/fetch/async`) |
| Risk task example | `src/risk/tasks.py` (`evaluate_trade_risk`) |
| Portfolio / training / prediction / backtesting tasks | `src/portfolio/tasks.py`, `src/training/tasks.py`, `src/prediction/tasks.py`, `src/backtesting/tasks.py` |
| Worker + Beat (Docker) | `docker-compose-core.yml` → `celery-worker`, `celery-beat` |

---

## 4. Observing the flow in practice

1. **Worker logs (simplest)**  
   ```bash
   docker compose -f docker-compose-core.yml logs -f celery-worker
   ```  
   Trigger either:
   - **Beat**: wait ~5s and watch for `fetch_btc_price_realtime` logs, or  
   - **API**: `curl -X POST http://localhost:8011/api/v1/market-data/fetch/async -H "Content-Type: application/json" -d '{"symbols":["AAPL"]}'` (add auth if required).  
   Then watch the same logs for the received task and the fetch logs.

2. **Task ID from API**  
   Use the `task_id` from the async fetch response to poll or wait for the result (e.g. in a script with `celery.result.AsyncResult(task_id).get(timeout=60)`).

3. **Flower (if you use docker-compose with Flower)**  
   In `docker-compose-complete.yml`, Flower is on port 5555. Open `http://localhost:5555` to see tasks, queues, and task history (you can see the flow as “sent → received → success/failure”).

4. **Prometheus / Grafana**  
   If celery-metrics and Prometheus are running, use the Celery dashboards (see `docs/archive/CELERY_MONITORING_SETUP.md`) to see task counts and duration per task name/queue.

---

## 5. Quick “see the flow” checklist

- [ ] Start stack: `docker compose -f docker-compose-core.yml up -d` (at least redis, celery-worker, celery-beat).
- [ ] Tail worker: `docker compose -f docker-compose-core.yml logs -f celery-worker`.
- [ ] **Beat path**: Wait a few seconds and confirm `market_data.fetch_btc_price_realtime` (or similar) in logs.
- [ ] **API path**: Call `POST .../market-data/fetch/async` with `{"symbols":["AAPL"]}`, note `task_id`, then confirm the same task in worker logs.
- [ ] (Optional) Add Flower and open its UI to see the same tasks in the queue and in history.

That’s the **Celery flow** and how you can see it end to end.
