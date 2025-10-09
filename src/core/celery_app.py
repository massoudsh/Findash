"""
Celery application configuration for Quantum Trading Matrix™
Configures distributed task processing for data ingestion, ML training, and analysis
"""

from celery import Celery
from src.core.config import get_settings

settings = get_settings()

# Create Celery application
celery_app = Celery(
    "quantum_trading_matrix",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "src.data_processing.tasks",
        "src.training.tasks", 
        "src.prediction.tasks",
        "src.backtesting.tasks",
        "src.portfolio.tasks",
        "src.risk.tasks",
        "src.strategies.tasks",
        "src.analytics.service",
        "src.generative.tasks",
        "src.llm.tasks.finetuning_task",
        "src.llm.tasks.unsloth_finetuning_task",
        "src.data_processing.collection_tasks"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
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
    task_routes={
        'data_processing.*': {'queue': 'data_processing'},
        'training.*': {'queue': 'ml_training'},
        'prediction.*': {'queue': 'prediction'},
        'portfolio.*': {'queue': 'portfolio'},
        'risk.*': {'queue': 'risk'},
        'strategies.*': {'queue': 'strategies'},
        'analytics.*': {'queue': 'analytics'},
        'generative.*': {'queue': 'generative'},
        'llm.*': {'queue': 'llm'},
    },
    task_default_queue='default',
    task_default_exchange='default',
    task_default_exchange_type='direct',
    task_default_routing_key='default',
) 