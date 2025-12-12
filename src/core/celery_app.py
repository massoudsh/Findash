"""
Celery application configuration for Octopus Trading Platform™
Configures distributed task processing for data ingestion, ML training, and analysis
Enhanced with Redis pub/sub pattern for task allocation and Prometheus monitoring
"""

import logging
from celery import Celery
from celery.signals import (
    task_prerun, task_postrun, task_failure, 
    worker_ready, worker_shutdown, task_sent
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create Celery application
celery_app = Celery(
    "octopus_trading_platform",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "src.data_processing.tasks",
        "src.data_processing.market_data_tasks",
        "src.training.tasks", 
        "src.prediction.tasks",
        "src.backtesting.tasks",
        "src.portfolio.tasks",
        "src.risk.tasks",
        "src.strategies.tasks",
        "src.analytics.service",
        "src.generative.tasks",
        "src.data_processing.collection_tasks",
        # LLM tasks disabled - too many optional dependencies
        # Enable when all dependencies are installed: datasets, transformers, etc.
        # "src.llm.tasks.finetuning_task",
        # "src.llm.tasks.unsloth_finetuning_task",
    ]
)

# Configure Celery with Redis pub/sub pattern
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content_list,
    timezone=settings.celery.timezone,
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    task_compression='gzip',
    result_compression='gzip',
    result_expires=3600,  # 1 hour
    # Redis configuration - simple direct exchange (no Sentinel)
    broker_transport_options={
        'visibility_timeout': 3600,
        'retry_policy': {
            'timeout': 5.0
        },
    },
    # Task routing - Redis uses direct exchanges by default
    task_routes={
        'data_processing.*': {'queue': 'data_processing'},
        'market_data.*': {'queue': 'data_processing'},
        'training.*': {'queue': 'ml_training'},
        'prediction.*': {'queue': 'prediction'},
        'portfolio.*': {'queue': 'portfolio'},
        'risk.*': {'queue': 'risk'},
        'strategies.*': {'queue': 'strategies'},
        'analytics.*': {'queue': 'analytics'},
        'generative.*': {'queue': 'generative'},
        'llm.*': {'queue': 'llm'},
    },
    # Beat schedule
    beat_schedule={
        # Fetch BTC price every 5 seconds (real-time tracking)
        'fetch-btc-price-realtime': {
            'task': 'market_data.fetch_btc_price_realtime',
            'schedule': 5.0,  # 5 seconds - real-time updates
        },
        # Fetch watchlist data every 5 minutes during market hours
        'fetch-watchlist-market-data': {
            'task': 'market_data.fetch_watchlist',
            'schedule': 300.0,  # 5 minutes
        },
        # Update portfolio symbols every 10 minutes
        'update-portfolio-symbols': {
            'task': 'market_data.update_portfolio_symbols',
            'schedule': 600.0,  # 10 minutes
        },
        # Cleanup old data daily at 2 AM
        'cleanup-old-market-data': {
            'task': 'market_data.cleanup_old_data',
            'schedule': 86400.0,  # 24 hours
            'kwargs': {'days_to_keep': 30}
        },
    },
    # Default queue configuration - Redis uses direct exchanges
    task_default_queue='default',
    task_default_exchange='',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
    # Worker pool configuration
    worker_pool='prefork',
    worker_concurrency=4,
    # Result backend configuration
    result_backend_transport_options={
        'retry_policy': {
            'timeout': 5.0
        },
    },
)

# Initialize metrics (will be imported from monitoring module)
try:
    from src.monitoring import (
        celery_task_started_total,
        celery_task_succeeded_total,
        celery_task_failed_total,
        celery_task_duration_seconds,
        celery_worker_active_tasks,
        celery_queue_length,
    )
    METRICS_ENABLED = True
except ImportError:
    logger.warning("Celery metrics module not found. Monitoring will be limited.")
    METRICS_ENABLED = False
    # Create dummy metrics to avoid errors
    celery_task_started_total = None
    celery_task_succeeded_total = None
    celery_task_failed_total = None
    celery_task_duration_seconds = None
    celery_worker_active_tasks = None
    celery_queue_length = None


# Celery signal handlers for Prometheus metrics
@task_sent.connect
def task_sent_handler(sender=None, task_id=None, task=None, **kwargs):
    """Track task sent events"""
    if METRICS_ENABLED and celery_task_started_total:
        try:
            celery_task_started_total.labels(
                task_name=task or 'unknown',
                queue=kwargs.get('queue', 'default')
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to track task sent metric: {e}")
    logger.debug(f"Task {task_id} sent: {task}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Track task start"""
    if METRICS_ENABLED and celery_task_started_total:
        try:
            celery_task_started_total.labels(
                task_name=task.name if task else 'unknown',
                queue=kwargs.get('queue', 'default')
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to track task prerun metric: {e}")
    logger.debug(f"Task {task_id} started: {task.name if task else 'unknown'}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Track task completion"""
    if METRICS_ENABLED and celery_task_succeeded_total and task:
        try:
            celery_task_succeeded_total.labels(
                task_name=task.name,
                queue=kwargs.get('queue', 'default')
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to track task postrun metric: {e}")
    logger.debug(f"Task {task_id} completed: {task.name if task else 'unknown'}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Track task failures"""
    if METRICS_ENABLED and celery_task_failed_total:
        try:
            task_name = kwargs.get('task', 'unknown')
            celery_task_failed_total.labels(
                task_name=task_name,
                queue=kwargs.get('queue', 'default'),
                exception_type=type(exception).__name__ if exception else 'unknown'
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to track task failure metric: {e}")
    logger.error(f"Task {task_id} failed: {exception}")


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Track worker ready"""
    logger.info(f"Worker {sender} is ready")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Track worker shutdown"""
    logger.info(f"Worker {sender} is shutting down") 