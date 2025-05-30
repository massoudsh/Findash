"""Monitoring and metrics collection system."""

import time
from functools import wraps
from typing import Callable, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from contextlib import contextmanager
from core.config import config
from core.logging_config import setup_logging

logger = setup_logging(__name__)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter(
    'app_request_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'app_active_requests',
    'Number of active requests',
    ['method', 'endpoint']
)

API_ERRORS = Counter(
    'app_api_errors_total',
    'Total number of API errors',
    ['api_name', 'error_type']
)

DATA_PROCESSING_TIME = Histogram(
    'app_data_processing_seconds',
    'Time spent processing data',
    ['data_type']
)

MEMORY_USAGE = Gauge(
    'app_memory_usage_bytes',
    'Memory usage in bytes'
)

class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, port: int = 8000):
        """Initialize metrics collector."""
        self.port = port
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")

    @staticmethod
    def track_request(method: str, endpoint: str) -> None:
        """Track HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=200).inc()

    @staticmethod
    def track_error(api_name: str, error_type: str) -> None:
        """Track API error metrics."""
        API_ERRORS.labels(api_name=api_name, error_type=error_type).inc()

    @contextmanager
    def track_operation_time(self, operation_name: str):
        """Track operation execution time."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            DATA_PROCESSING_TIME.labels(data_type=operation_name).observe(duration)

    @staticmethod
    def track_memory(memory_bytes: float):
        """Track memory usage."""
        MEMORY_USAGE.set(memory_bytes)

# Create global metrics collector instance
metrics = MetricsCollector(port=config.get('monitoring.port', 8000))

def monitor_performance(operation: Optional[str] = None) -> Callable:
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = operation or func.__name__
            with metrics.track_operation_time(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator 