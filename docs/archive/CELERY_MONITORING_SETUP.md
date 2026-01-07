# Celery Workers with Redis Pub/Sub & Prometheus/Grafana Monitoring

## Overview

This setup implements:
1. **Redis Pub/Sub Pattern** for efficient Celery task allocation
2. **Prometheus Metrics** for comprehensive monitoring
3. **Grafana Dashboards** for visualization

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Celery    │─────▶│     Redis    │─────▶│  Prometheus │
│   Workers   │◀─────│  (Pub/Sub)   │◀─────│   Metrics   │
└─────────────┘      └──────────────┘      └─────────────┘
      │                      │                      │
      │                      │                      ▼
      │                      │              ┌─────────────┐
      └──────────────────────┴─────────────▶│   Grafana   │
                                             │  Dashboard  │
                                             └─────────────┘
```

## Quick Start

### 1. Start Services

```bash
# Start all services including monitoring
docker-compose -f docker-compose-complete.yml up -d

# Or start specific services
docker-compose up -d redis celery-worker celery-metrics prometheus grafana
```

### 2. Verify Services

```bash
# Check Celery metrics exporter
curl http://localhost:9540/metrics

# Check Prometheus targets
open http://localhost:9090/targets

# Access Grafana
open http://localhost:3001
# Default credentials: admin / admin (change on first login)
```

### 3. View Dashboard

1. Navigate to Grafana: `http://localhost:3001`
2. Go to **Infrastructure** folder
3. Open **Celery Workers & Task Monitoring** dashboard

## Components

### Celery App (`src/core/celery_app.py`)
- Enhanced with Redis pub/sub configuration
- Task routing with topic exchanges
- Signal handlers for Prometheus metrics
- Automatic task tracking

### Metrics Exporter (`src/monitoring/celery_exporter_service.py`)
- Standalone service on port 9540
- Collects metrics from Redis and workers
- Updates every 5 seconds
- Exposes Prometheus format metrics

### Pub/Sub Allocator (`src/monitoring/celery_pubsub.py`)
- Worker registration and management
- Task publishing via Redis channels
- Queue statistics
- Allocation tracking

### Prometheus Configuration
- Scrapes `celery-metrics:9540` every 10 seconds
- Configured in `monitoring/prometheus-hybrid.yml`

### Grafana Dashboard
- 11 panels covering all aspects of Celery operations
- Real-time updates (10s refresh)
- Located at `monitoring/grafana/dashboards/05-celery-monitoring.json`

## Usage Examples

### Publishing Tasks via Pub/Sub

```python
from src.monitoring.celery_pubsub import get_allocator

allocator = get_allocator()
allocator.connect()

# Publish a high-priority task
allocator.publish_task(
    task_name='process_market_data',
    queue='data_processing',
    task_data={'symbol': 'AAPL'},
    priority=9,  # High priority
    allocation_method='pubsub'
)
```

### Registering Workers

```python
allocator.register_worker(
    worker_name='worker-1',
    queues=['data_processing', 'prediction'],
    capabilities={
        'cpu': 4,
        'memory': 8192,
        'gpu': False
    }
)
```

### Using Celery Tasks (Standard)

```python
from src.core.celery_app import celery_app

@celery_app.task(name='data_processing.process_data')
def process_data(symbol: str):
    # Task implementation
    pass

# Task is automatically tracked by metrics
process_data.delay('AAPL')
```

## Metrics Available

### Task Metrics
- `celery_task_started_total` - Tasks started
- `celery_task_succeeded_total` - Tasks completed successfully
- `celery_task_failed_total` - Tasks failed
- `celery_task_duration_seconds` - Execution duration histogram

### Worker Metrics
- `celery_worker_active_tasks` - Active tasks per worker
- `celery_worker_pool_size` - Worker pool capacity
- `celery_worker_reserved_tasks` - Reserved tasks

### Queue Metrics
- `celery_queue_length` - Tasks waiting in queue
- `celery_queue_consumers` - Number of consumers

### Pub/Sub Metrics
- `celery_redis_pubsub_messages_total` - Pub/sub throughput
- `celery_redis_pubsub_latency_seconds` - Message processing latency
- `celery_task_allocation_total` - Task allocations
- `celery_task_allocation_duration_seconds` - Allocation time

## Configuration

### Environment Variables

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Metrics Exporter
METRICS_PORT=9540
METRICS_HOST=0.0.0.0
```

### Docker Compose

The `celery-metrics` service is configured in `docker-compose-complete.yml`:

```yaml
celery-metrics:
  build:
    context: .
    dockerfile: Dockerfile.celery
  ports:
    - "9540:9540"
  environment:
    - REDIS_URL=redis://redis:6379/0
    - METRICS_PORT=9540
```

## Monitoring Dashboard Panels

1. **Task Execution Rate** - Real-time task throughput
2. **Task Success vs Failure** - Success rate visualization
3. **Task Duration (p95)** - Performance metrics
4. **Queue Length** - Backlog monitoring
5. **Active Tasks per Worker** - Worker utilization
6. **Worker Pool Size** - Capacity tracking
7. **Redis Pub/Sub Message Rate** - Pub/sub throughput
8. **Pub/Sub Latency** - Message processing speed
9. **Task Allocation Rate** - Allocation frequency
10. **Queue Consumers** - Consumer count
11. **Task Failure Rate by Exception** - Error analysis

## Troubleshooting

### Metrics Not Appearing

1. **Check metrics exporter is running:**
   ```bash
   docker ps | grep celery-metrics
   curl http://localhost:9540/metrics
   ```

2. **Verify Prometheus is scraping:**
   - Open `http://localhost:9090/targets`
   - Check `celery-workers` target status

3. **Check Redis connection:**
   ```bash
   docker exec -it octopus-redis redis-cli ping
   ```

### Pub/Sub Not Working

1. **Verify Redis is accessible:**
   ```bash
   docker exec -it octopus-redis redis-cli PUBSUB CHANNELS
   ```

2. **Check worker registration:**
   ```python
   from src.monitoring.celery_pubsub import get_allocator
   allocator = get_allocator()
   allocator.connect()
   stats = allocator.get_queue_stats('data_processing')
   print(stats)
   ```

### High Latency

1. **Monitor pub/sub latency in Grafana**
2. **Check Redis performance:**
   ```bash
   docker exec -it octopus-redis redis-cli --latency
   ```
3. **Consider Redis cluster for high throughput**

## Performance Tuning

### Worker Configuration

```python
# In celery_app.py
celery_app.conf.update(
    worker_prefetch_multiplier=1,  # Don't prefetch too many tasks
    task_acks_late=True,  # Acknowledge after completion
    worker_max_tasks_per_child=1000,  # Restart workers periodically
)
```

### Redis Pub/Sub

- Use separate Redis instances for pub/sub and result backend
- Monitor `celery_redis_pubsub_latency_seconds`
- Scale workers based on queue length metrics

### Metrics Collection

- Adjust scrape interval in Prometheus (default: 10s)
- Reduce collection frequency if high overhead
- Use metric aggregation for high-volume scenarios

## Next Steps

1. **Set up alerts** in Prometheus for critical metrics
2. **Scale workers** based on queue length
3. **Optimize task routing** using priority queues
4. **Monitor pub/sub latency** and optimize Redis configuration
5. **Set up alerting** for task failures and queue backlogs

## References

- [Celery Documentation](https://docs.celeryproject.org/)
- [Redis Pub/Sub](https://redis.io/docs/manual/pubsub/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

