"""
Robust Prometheus Metrics for Octopus Trading Platform
Comprehensive metrics collection for API, trading, system, and Celery operations
"""

import time
import logging
from typing import Optional, Dict, Any
from functools import wraps
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
from prometheus_client.multiprocess import MultiProcessCollector

logger = logging.getLogger(__name__)

# ============================================
# API METRICS
# ============================================

# HTTP Request Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
)

# API Error Metrics
http_errors_total = Counter(
    'http_errors_total',
    'Total number of HTTP errors',
    ['method', 'endpoint', 'error_type', 'status_code']
)

# ============================================
# TRADING METRICS
# ============================================

trades_total = Counter(
    'trading_trades_total',
    'Total number of trades executed',
    ['symbol', 'side', 'status', 'strategy']
)

trade_execution_duration_seconds = Histogram(
    'trading_trade_execution_duration_seconds',
    'Trade execution duration in seconds',
    ['symbol', 'side', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

portfolio_value_usd = Gauge(
    'trading_portfolio_value_usd',
    'Current portfolio value in USD',
    ['portfolio_id', 'portfolio_name']
)

portfolio_positions = Gauge(
    'trading_portfolio_positions',
    'Number of positions in portfolio',
    ['portfolio_id']
)

strategy_performance_percent = Gauge(
    'trading_strategy_performance_percent',
    'Strategy performance percentage',
    ['strategy_id', 'strategy_name', 'timeframe']
)

strategy_sharpe_ratio = Gauge(
    'trading_strategy_sharpe_ratio',
    'Strategy Sharpe ratio',
    ['strategy_id', 'strategy_name']
)

# ============================================
# RISK METRICS
# ============================================

risk_metrics = Gauge(
    'trading_risk_metrics',
    'Risk metrics values',
    ['metric_type', 'portfolio_id', 'timeframe']
)

risk_violations_total = Counter(
    'trading_risk_violations_total',
    'Total number of risk violations',
    ['violation_type', 'portfolio_id', 'severity']
)

value_at_risk_usd = Gauge(
    'trading_value_at_risk_usd',
    'Value at Risk in USD',
    ['portfolio_id', 'confidence_level', 'timeframe']
)

# ============================================
# MARKET DATA METRICS
# ============================================

market_data_updates_total = Counter(
    'trading_market_data_updates_total',
    'Total number of market data updates',
    ['symbol', 'feed_source', 'data_type']
)

market_data_latency_seconds = Histogram(
    'trading_market_data_latency_seconds',
    'Market data feed latency in seconds',
    ['feed_source', 'symbol'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

market_data_errors_total = Counter(
    'trading_market_data_errors_total',
    'Total number of market data errors',
    ['feed_source', 'error_type']
)

market_data_cache_hits_total = Counter(
    'trading_market_data_cache_hits_total',
    'Total number of market data cache hits',
    ['cache_type', 'symbol']
)

market_data_cache_misses_total = Counter(
    'trading_market_data_cache_misses_total',
    'Total number of market data cache misses',
    ['cache_type', 'symbol']
)

# ============================================
# ML/AI METRICS
# ============================================

ml_model_predictions_total = Counter(
    'trading_ml_model_predictions_total',
    'Total number of ML model predictions',
    ['model_id', 'model_type', 'symbol']
)

ml_model_accuracy_percent = Gauge(
    'trading_ml_model_accuracy_percent',
    'ML model accuracy percentage',
    ['model_id', 'model_type', 'metric_type']
)

ml_model_prediction_duration_seconds = Histogram(
    'trading_ml_model_prediction_duration_seconds',
    'ML model prediction duration in seconds',
    ['model_id', 'model_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ml_model_training_duration_seconds = Histogram(
    'trading_ml_model_training_duration_seconds',
    'ML model training duration in seconds',
    ['model_id', 'model_type'],
    buckets=[10, 30, 60, 300, 600, 1800, 3600]
)

# ============================================
# DATABASE METRICS
# ============================================

db_queries_total = Counter(
    'trading_db_queries_total',
    'Total number of database queries',
    ['operation', 'table', 'status']
)

db_query_duration_seconds = Histogram(
    'trading_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

db_connections_active = Gauge(
    'trading_db_connections_active',
    'Number of active database connections',
    ['pool_name']
)

db_connections_idle = Gauge(
    'trading_db_connections_idle',
    'Number of idle database connections',
    ['pool_name']
)

db_errors_total = Counter(
    'trading_db_errors_total',
    'Total number of database errors',
    ['operation', 'error_type']
)

# ============================================
# REDIS METRICS
# ============================================

redis_operations_total = Counter(
    'trading_redis_operations_total',
    'Total number of Redis operations',
    ['operation', 'status']
)

redis_operation_duration_seconds = Histogram(
    'trading_redis_operation_duration_seconds',
    'Redis operation duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

redis_cache_hits_total = Counter(
    'trading_redis_cache_hits_total',
    'Total number of Redis cache hits',
    ['cache_type', 'key_pattern']
)

redis_cache_misses_total = Counter(
    'trading_redis_cache_misses_total',
    'Total number of Redis cache misses',
    ['cache_type', 'key_pattern']
)

redis_pubsub_messages_total = Counter(
    'trading_redis_pubsub_messages_total',
    'Total number of Redis pub/sub messages',
    ['channel', 'message_type']
)

# ============================================
# WEBSOCKET METRICS
# ============================================

websocket_connections_active = Gauge(
    'trading_websocket_connections_active',
    'Number of active WebSocket connections',
    ['endpoint']
)

websocket_connections_total = Counter(
    'trading_websocket_connections_total',
    'Total number of WebSocket connections',
    ['endpoint', 'status']
)

websocket_messages_sent_total = Counter(
    'trading_websocket_messages_sent_total',
    'Total number of WebSocket messages sent',
    ['endpoint', 'message_type']
)

websocket_messages_received_total = Counter(
    'trading_websocket_messages_received_total',
    'Total number of WebSocket messages received',
    ['endpoint', 'message_type']
)

websocket_message_size_bytes = Histogram(
    'trading_websocket_message_size_bytes',
    'WebSocket message size in bytes',
    ['endpoint', 'direction'],
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
)

# ============================================
# SYSTEM METRICS
# ============================================

system_info = Info(
    'trading_system_info',
    'System information'
)

system_uptime_seconds = Gauge(
    'trading_system_uptime_seconds',
    'System uptime in seconds'
)

system_memory_usage_bytes = Gauge(
    'trading_system_memory_usage_bytes',
    'System memory usage in bytes',
    ['memory_type']
)

system_cpu_usage_percent = Gauge(
    'trading_system_cpu_usage_percent',
    'System CPU usage percentage'
)

# ============================================
# CELERY METRICS (Integrated)
# ============================================

celery_tasks_total = Counter(
    'trading_celery_tasks_total',
    'Total number of Celery tasks',
    ['task_name', 'queue', 'status']
)

celery_task_duration_seconds = Histogram(
    'trading_celery_task_duration_seconds',
    'Celery task duration in seconds',
    ['task_name', 'queue', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
)

celery_workers_active = Gauge(
    'trading_celery_workers_active',
    'Number of active Celery workers',
    ['queue']
)

celery_queue_length = Gauge(
    'trading_celery_queue_length',
    'Number of tasks in Celery queue',
    ['queue_name']
)

# ============================================
# METRICS COLLECTOR CLASS
# ============================================

class MetricsCollector:
    """Centralized metrics collection and management"""
    
    def __init__(self):
        self.start_time = time.time()
        system_info.info({
            'version': '3.0.0',
            'platform': 'octopus-trading-platform',
            'environment': os.getenv('ENVIRONMENT', 'development')
        })
        system_uptime_seconds.set(0)
    
    def update_uptime(self):
        """Update system uptime"""
        system_uptime_seconds.set(time.time() - self.start_time)
    
    @staticmethod
    def record_http_request(method: str, endpoint: str, status_code: int, 
                          duration: float, request_size: Optional[int] = None,
                          response_size: Optional[int] = None):
        """Record HTTP request metrics"""
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(duration)
        
        if request_size is not None:
            http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size is not None:
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).observe(response_size)
        
        if status_code >= 400:
            http_errors_total.labels(
                method=method,
                endpoint=endpoint,
                error_type='http_error',
                status_code=status_code
            ).inc()
    
    @staticmethod
    def record_trade(symbol: str, side: str, status: str, strategy: str = 'unknown'):
        """Record trade execution"""
        trades_total.labels(
            symbol=symbol,
            side=side,
            status=status,
            strategy=strategy
        ).inc()
    
    @staticmethod
    def record_trade_execution_time(symbol: str, side: str, duration: float, status: str = 'success'):
        """Record trade execution time"""
        trade_execution_duration_seconds.labels(
            symbol=symbol,
            side=side,
            status=status
        ).observe(duration)
    
    @staticmethod
    def update_portfolio_value(portfolio_id: str, portfolio_name: str, value: float):
        """Update portfolio value"""
        portfolio_value_usd.labels(
            portfolio_id=portfolio_id,
            portfolio_name=portfolio_name
        ).set(value)
    
    @staticmethod
    def record_market_data_update(symbol: str, feed_source: str, data_type: str):
        """Record market data update"""
        market_data_updates_total.labels(
            symbol=symbol,
            feed_source=feed_source,
            data_type=data_type
        ).inc()
    
    @staticmethod
    def record_market_data_latency(feed_source: str, symbol: str, latency: float):
        """Record market data latency"""
        market_data_latency_seconds.labels(
            feed_source=feed_source,
            symbol=symbol
        ).observe(latency)
    
    @staticmethod
    def record_db_query(operation: str, table: str, duration: float, status: str = 'success'):
        """Record database query"""
        db_queries_total.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        
        db_query_duration_seconds.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    @staticmethod
    def record_redis_operation(operation: str, duration: float, status: str = 'success'):
        """Record Redis operation"""
        redis_operations_total.labels(
            operation=operation,
            status=status
        ).inc()
        
        redis_operation_duration_seconds.labels(
            operation=operation
        ).observe(duration)
    
    @staticmethod
    def record_websocket_connection(endpoint: str, status: str):
        """Record WebSocket connection"""
        websocket_connections_total.labels(
            endpoint=endpoint,
            status=status
        ).inc()
        
        if status == 'connected':
            websocket_connections_active.labels(endpoint=endpoint).inc()
        elif status == 'disconnected':
            websocket_connections_active.labels(endpoint=endpoint).dec()
    
    @staticmethod
    def record_celery_task(task_name: str, queue: str, duration: float, status: str = 'success'):
        """Record Celery task execution"""
        celery_tasks_total.labels(
            task_name=task_name,
            queue=queue,
            status=status
        ).inc()
        
        celery_task_duration_seconds.labels(
            task_name=task_name,
            queue=queue,
            status=status
        ).observe(duration)


# Global metrics collector instance
metrics_collector = MetricsCollector()

# ============================================
# DECORATORS FOR EASY METRICS TRACKING
# ============================================

def track_http_request(endpoint: str):
    """Decorator to track HTTP requests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            method = request.method
            start_time = time.time()
            status_code = 200
            
            try:
                response = await func(request, *args, **kwargs)
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                duration = time.time() - start_time
                
                metrics_collector.record_http_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration
                )
                
                return response
            except Exception as e:
                status_code = 500
                duration = time.time() - start_time
                metrics_collector.record_http_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration
                )
                raise
        
        return wrapper
    return decorator


def track_db_operation(operation: str, table: str):
    """Decorator to track database operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_db_query(operation, table, duration, status)
                return result
            except Exception as e:
                status = 'error'
                duration = time.time() - start_time
                metrics_collector.record_db_query(operation, table, duration, status)
                raise
        
        return wrapper
    return decorator


def track_celery_task(task_name: str, queue: str):
    """Decorator to track Celery tasks"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_celery_task(task_name, queue, duration, status)
                return result
            except Exception as e:
                status = 'error'
                duration = time.time() - start_time
                metrics_collector.record_celery_task(task_name, queue, duration, status)
                raise
        
        return wrapper
    return decorator


# Import os for environment variables
import os

