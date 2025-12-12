"""
Prometheus metrics exporter for Celery workers
Provides comprehensive monitoring of task execution, worker status, and queue metrics
"""

import time
import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from prometheus_client.core import CollectorRegistry, REGISTRY

logger = logging.getLogger(__name__)

# Create a separate registry for Celery metrics
CELERY_REGISTRY = CollectorRegistry()

# Task execution metrics
celery_task_started_total = Counter(
    'celery_task_started_total',
    'Total number of Celery tasks started',
    ['task_name', 'queue'],
    registry=CELERY_REGISTRY
)

celery_task_succeeded_total = Counter(
    'celery_task_succeeded_total',
    'Total number of Celery tasks completed successfully',
    ['task_name', 'queue'],
    registry=CELERY_REGISTRY
)

celery_task_failed_total = Counter(
    'celery_task_failed_total',
    'Total number of Celery tasks that failed',
    ['task_name', 'queue', 'exception_type'],
    registry=CELERY_REGISTRY
)

celery_task_duration_seconds = Histogram(
    'celery_task_duration_seconds',
    'Duration of Celery task execution in seconds',
    ['task_name', 'queue', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
    registry=CELERY_REGISTRY
)

# Worker metrics
celery_worker_active_tasks = Gauge(
    'celery_worker_active_tasks',
    'Number of active tasks per worker',
    ['worker_name', 'queue'],
    registry=CELERY_REGISTRY
)

celery_worker_pool_size = Gauge(
    'celery_worker_pool_size',
    'Worker pool size',
    ['worker_name'],
    registry=CELERY_REGISTRY
)

celery_worker_reserved_tasks = Gauge(
    'celery_worker_reserved_tasks',
    'Number of reserved tasks per worker',
    ['worker_name', 'queue'],
    registry=CELERY_REGISTRY
)

# Queue metrics
celery_queue_length = Gauge(
    'celery_queue_length',
    'Number of tasks in queue',
    ['queue_name'],
    registry=CELERY_REGISTRY
)

celery_queue_consumers = Gauge(
    'celery_queue_consumers',
    'Number of consumers for a queue',
    ['queue_name'],
    registry=CELERY_REGISTRY
)

# Redis pub/sub metrics
celery_redis_pubsub_messages = Counter(
    'celery_redis_pubsub_messages_total',
    'Total number of pub/sub messages processed',
    ['channel', 'type'],
    registry=CELERY_REGISTRY
)

celery_redis_pubsub_latency_seconds = Histogram(
    'celery_redis_pubsub_latency_seconds',
    'Latency of pub/sub message processing',
    ['channel'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=CELERY_REGISTRY
)

# Task allocation metrics
celery_task_allocation_total = Counter(
    'celery_task_allocation_total',
    'Total number of task allocations',
    ['queue', 'worker', 'allocation_method'],
    registry=CELERY_REGISTRY
)

celery_task_allocation_duration_seconds = Histogram(
    'celery_task_allocation_duration_seconds',
    'Time taken to allocate a task to a worker',
    ['queue', 'allocation_method'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=CELERY_REGISTRY
)


class CeleryMetricsExporter:
    """
    Exports Celery metrics to Prometheus via HTTP endpoint
    """
    
    def __init__(self, port: int = 9540, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.server = None
    
    def start(self):
        """Start the metrics HTTP server"""
        try:
            from prometheus_client import make_asgi_app
            from prometheus_client.openmetrics.exposition import generate_latest
            
            # For standalone server
            start_http_server(self.port, registry=CELERY_REGISTRY)
            logger.info(f"Celery metrics exporter started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics exporter: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        from prometheus_client import generate_latest
        return generate_latest(CELERY_REGISTRY).decode('utf-8')


def track_task_execution(task_name: str, queue: str, duration: float, status: str = 'success'):
    """Track task execution metrics"""
    celery_task_duration_seconds.labels(
        task_name=task_name,
        queue=queue,
        status=status
    ).observe(duration)


def update_queue_length(queue_name: str, length: int):
    """Update queue length metric"""
    celery_queue_length.labels(queue_name=queue_name).set(length)


def update_worker_active_tasks(worker_name: str, queue: str, count: int):
    """Update active tasks metric for a worker"""
    celery_worker_active_tasks.labels(
        worker_name=worker_name,
        queue=queue
    ).set(count)


def track_pubsub_message(channel: str, message_type: str, latency: Optional[float] = None):
    """Track pub/sub message processing"""
    celery_redis_pubsub_messages.labels(
        channel=channel,
        type=message_type
    ).inc()
    
    if latency is not None:
        celery_redis_pubsub_latency_seconds.labels(channel=channel).observe(latency)


def track_task_allocation(queue: str, worker: str, allocation_method: str, duration: Optional[float] = None):
    """Track task allocation to workers"""
    celery_task_allocation_total.labels(
        queue=queue,
        worker=worker,
        allocation_method=allocation_method
    ).inc()
    
    if duration is not None:
        celery_task_allocation_duration_seconds.labels(
            queue=queue,
            allocation_method=allocation_method
        ).observe(duration)

