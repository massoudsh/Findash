# BTC Real-time Price Tracking - Complete Data Flow

## Overview

This system demonstrates a complete real-time data flow from free API calls to UI visualization, with full observability in Grafana, Prometheus, and Flower.

## Data Flow Architecture

```
┌─────────────────┐
│ Free API        │
│ CoinGecko API   │  (Every 5 seconds)
│ Binance (fallback)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Celery Beat     │  Scheduled task: fetch_btc_price_realtime
│ (Scheduler)     │  Runs every 5 seconds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Celery Worker   │  Executes task
│ (Task Runner)   │  Tracks in Flower
└────────┬────────┘
         │
    ┌────┴────┐
    │        │
    ▼        ▼
┌────────┐ ┌──────────────┐
│ Redis  │ │ Prometheus   │
│ Cache  │ │ Metrics       │
│        │ │ (port 8003)   │
└───┬────┘ └───────┬───────┘
    │              │
    │              ▼
    │         ┌──────────┐
    │         │ Grafana  │
    │         │ Dashboard│
    │         └──────────┘
    │
    ▼
┌──────────────┐
│ Frontend API │  /api/btc-price
│ (Next.js)    │  Reads from Redis
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ UI Widget    │  BTCPriceWidget
│ (React)      │  Updates every 5s
└──────────────┘
```

## Components

### 1. BTC Price Tracker (`src/data_processing/btc_price_tracker.py`)
- Fetches from CoinGecko free API
- Fallback to Binance API
- Tracks Prometheus metrics:
  - API call count (success/error)
  - API latency
  - Current price
  - 24h change
  - Cache hits/misses

### 2. Celery Task (`market_data.fetch_btc_price_realtime`)
- Scheduled every 5 seconds via Celery Beat
- Fetches BTC price from free API
- Stores in Redis cache (`btc_price:latest`)
- Publishes to Redis pub/sub (`btc_price_updates`)
- Tracked in Flower

### 3. Metrics Exporter (`src/monitoring/btc_metrics_exporter.py`)
- Exposes Prometheus metrics on port 8003
- Scraped by Prometheus every 5 seconds

### 4. Grafana Dashboard (`07-btc-realtime-tracking.json`)
- **Current Price**: Real-time BTC price in USD
- **24h Change**: Percentage change
- **API Call Rate**: Calls per second
- **API Latency**: p50 and p95 latency
- **Price History**: Last 5 minutes
- **Cache Performance**: Hit/miss rates
- **Success Rate**: API success percentage
- **Task Execution**: Celery task metrics

### 5. Frontend Widget (`BTCPriceWidget`)
- Real-time BTC price display
- Updates every 5 seconds
- Shows:
  - Current price
  - 24h change with trend indicator
  - Volume 24h
  - API latency
  - Data source

### 6. API Route (`/api/btc-price`)
- Reads from Redis cache
- Fallback to CoinGecko if Redis unavailable

## Access Points

### Grafana
- **URL**: http://localhost:3001
- **Login**: admin / admin
- **Dashboard**: "BTC Real-time Price Tracking"
- **Location**: Dashboards → Trading

### Prometheus
- **URL**: http://localhost:9090
- **Query Examples**:
  ```promql
  # Current BTC price
  btc_price_current_usd
  
  # API call rate
  rate(btc_api_calls_total[1m])
  
  # API latency p95
  histogram_quantile(0.95, rate(btc_api_latency_seconds_bucket[1m]))
  
  # Task execution rate
  rate(celery_task_succeeded_total{task_name="fetch_btc_price_realtime"}[1m])
  ```

### Flower
- **URL**: http://localhost:5555
- **View**:
  - Tasks tab: `fetch_btc_price_realtime` tasks
  - Workers tab: Active workers
  - Monitor tab: Real-time task execution

### Frontend UI
- **URL**: http://localhost:3000/realtime
- **Widget**: BTC price widget at top of page
- **Updates**: Every 5 seconds automatically

## Metrics Available

### Prometheus Metrics
- `btc_api_calls_total{status, source}` - Total API calls
- `btc_api_latency_seconds{source}` - API latency histogram
- `btc_price_current_usd{source}` - Current BTC price
- `btc_price_change_24h_percent{source}` - 24h change %
- `btc_api_cache_hits_total{cache_type}` - Cache hits
- `btc_api_cache_misses_total{cache_type}` - Cache misses

### Celery Metrics (via Flower/Prometheus)
- `celery_task_started_total{task_name="fetch_btc_price_realtime"}`
- `celery_task_succeeded_total{task_name="fetch_btc_price_realtime"}`
- `celery_task_duration_seconds{task_name="fetch_btc_price_realtime"}`

## Verification

### Check Task is Running
```bash
# Check celery-beat logs
docker logs octopus-celery-beat --tail 20 | grep btc

# Check worker logs
docker logs octopus-celery-worker --tail 20 | grep BTC

# Check Redis cache
docker exec octopus-redis redis-cli GET btc_price:latest
```

### Check Metrics
```bash
# BTC metrics endpoint
curl http://localhost:8003/metrics | grep btc

# Prometheus query
curl 'http://localhost:9090/api/v1/query?query=btc_price_current_usd'
```

### Check Flower
```bash
# View tasks
curl http://localhost:5555/api/tasks?limit=10 | jq '.[] | select(.name | contains("btc"))'
```

## Troubleshooting

### Task Not Running
1. Check celery-beat is running: `docker ps | grep celery-beat`
2. Check schedule: `docker logs octopus-celery-beat | grep schedule`
3. Restart: `docker-compose restart celery-beat`

### No Metrics
1. Check btc-metrics service: `docker ps | grep btc-metrics`
2. Check metrics endpoint: `curl http://localhost:8003/metrics`
3. Check Prometheus targets: http://localhost:9090/targets

### No Data in Grafana
1. Check Prometheus has data: Query `btc_price_current_usd`
2. Check dashboard is loaded: Grafana → Dashboards → Trading
3. Check time range: Set to "Last 5 minutes"

### UI Not Updating
1. Check API endpoint: `curl http://localhost:3000/api/btc-price`
2. Check Redis has data: `docker exec octopus-redis redis-cli GET btc_price:latest`
3. Check browser console for errors

## Expected Behavior

1. **Every 5 seconds**:
   - Celery Beat triggers `fetch_btc_price_realtime` task
   - Worker fetches from CoinGecko API
   - Data stored in Redis
   - Metrics updated in Prometheus
   - Task appears in Flower

2. **In Grafana**:
   - Price updates every 5 seconds
   - API latency shows in graph
   - Task execution rate visible

3. **In Flower**:
   - New task every 5 seconds
   - Task history shows success/failure
   - Real-time monitoring active

4. **In UI**:
   - Widget updates every 5 seconds
   - Price, change, and latency displayed
   - Source indicator shows API used

## Summary

This system provides **complete end-to-end observability** of real-time API data:
- ✅ **API Calls**: Tracked with latency and success rate
- ✅ **Task Execution**: Monitored in Flower
- ✅ **Metrics**: Exposed to Prometheus
- ✅ **Visualization**: Dashboard in Grafana
- ✅ **UI**: Real-time updates in frontend
- ✅ **Data Flow**: API → Redis → Database → UI fully tracked

