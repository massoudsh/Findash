# Celery Monitoring & Redis Pub/Sub Setup

This module provides comprehensive monitoring for Celery workers using Prometheus metrics and Grafana dashboards, with Redis pub/sub pattern for efficient task allocation.

## Components

### 1. Celery Metrics (`celery_metrics.py`)
- Prometheus metrics for task execution, worker status, and queue metrics
- Tracks task start, success, failure, and duration
- Monitors worker pool size, active tasks, and queue lengths
- Redis pub/sub message tracking

### 2. Redis Pub/Sub Allocator (`celery_pubsub.py`)
- Implements pub/sub pattern for task allocation
- Efficient task distribution across workers
- Worker registration and status tracking
- Queue statistics and monitoring

### 3. Metrics Exporter Service (`celery_exporter_service.py`)
- Standalone service exposing Prometheus metrics
- Collects metrics from Redis and Celery workers
- Runs on port 9540 by default
- Updates metrics every 5 seconds

## Setup

### 1. Start the Metrics Exporter

```bash
# Standalone
python -m src.monitoring.celery_exporter_service

# Or via Docker Compose
docker-compose up celery-metrics
```

### 2. Configure Prometheus

The Prometheus configuration (`monitoring/prometheus-hybrid.yml`) is already set up to scrape metrics from:
- `celery-metrics:9540` - Celery metrics endpoint

### 3. Access Grafana Dashboard

1. Open Grafana at `http://localhost:3001`
2. Navigate to "Infrastructure" folder
3. Open "Celery Workers & Task Monitoring" dashboard

## Usage

### Publishing Tasks via Pub/Sub

```python
from src.monitoring.celery_pubsub import get_allocator

allocator = get_allocator()
allocator.connect()

# Register a worker
allocator.register_worker(
    worker_name='worker-1',
    queues=['data_processing', 'prediction'],
    capabilities={'cpu': 4, 'memory': 8192}
)

# Publish a task
allocator.publish_task(
    task_name='process_market_data',
    queue='data_processing',
    task_data={'symbol': 'AAPL', 'date': '2024-01-01'},
    priority=8
)
```

### Using Metrics in Code

```python
from src.monitoring.celery_metrics import (
    track_task_execution,
    update_queue_length,
    track_pubsub_message
)

# Track task execution
track_task_execution(
    task_name='process_data',
    queue='data_processing',
    duration=2.5,
    status='success'
)

# Update queue length
update_queue_length('data_processing', 42)

# Track pub/sub message
track_pubsub_message('tasks:data_processing', 'task_published', 0.001)
```

## Metrics Exposed

### Task Metrics
- `celery_task_started_total` - Total tasks started
- `celery_task_succeeded_total` - Total tasks succeeded
- `celery_task_failed_total` - Total tasks failed
- `celery_task_duration_seconds` - Task execution duration

### Worker Metrics
- `celery_worker_active_tasks` - Active tasks per worker
- `celery_worker_pool_size` - Worker pool size
- `celery_worker_reserved_tasks` - Reserved tasks per worker

### Queue Metrics
- `celery_queue_length` - Number of tasks in queue
- `celery_queue_consumers` - Number of consumers for queue

### Pub/Sub Metrics
- `celery_redis_pubsub_messages_total` - Total pub/sub messages
- `celery_redis_pubsub_latency_seconds` - Pub/sub message latency
- `celery_task_allocation_total` - Task allocations
- `celery_task_allocation_duration_seconds` - Allocation duration

## Grafana Dashboard Panels

1. **Task Execution Rate** - Tasks started/succeeded/failed per second
2. **Task Success vs Failure Rate** - Pie chart of task outcomes
3. **Task Duration (p95)** - 95th percentile task duration
4. **Queue Length** - Number of tasks in each queue
5. **Active Tasks per Worker** - Current active tasks
6. **Worker Pool Size** - Worker pool capacity
7. **Redis Pub/Sub Message Rate** - Pub/sub throughput
8. **Pub/Sub Latency** - Message processing latency
9. **Task Allocation Rate** - Task allocation frequency
10. **Queue Consumers** - Number of consumers per queue
11. **Task Failure Rate by Exception** - Failure breakdown

## Configuration

Environment variables:
- `METRICS_PORT` - Metrics exporter port (default: 9540)
- `METRICS_HOST` - Metrics exporter host (default: 0.0.0.0)
- `REDIS_URL` - Redis connection URL
- `CELERY_BROKER_URL` - Celery broker URL

## Troubleshooting

### Metrics not appearing
1. Check that `celery-metrics` service is running
2. Verify Redis connection
3. Check Prometheus targets at `http://localhost:9090/targets`
4. Ensure port 9540 is accessible

### Pub/Sub not working
1. Verify Redis is running and accessible
2. Check worker registration
3. Ensure channels are subscribed correctly
4. Check Redis logs for connection errors

### High latency
1. Monitor `celery_redis_pubsub_latency_seconds`
2. Check Redis performance
3. Verify network connectivity
4. Consider Redis cluster for high throughput

