from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Trading-specific metrics
TRADES_TOTAL = Counter(
    'trading_platform_trades_total',
    'Total number of trades executed',
    ['symbol', 'side', 'status']
)

PORTFOLIO_VALUE = Gauge(
    'trading_platform_portfolio_value_usd',
    'Current portfolio value in USD',
    ['portfolio_id', 'portfolio_name']
)

TRADE_EXECUTION_TIME = Histogram(
    'trading_platform_trade_execution_seconds',
    'Time taken to execute trades',
    ['symbol', 'side']
)

STRATEGY_PERFORMANCE = Gauge(
    'trading_platform_strategy_performance_percent',
    'Strategy performance percentage',
    ['strategy_id', 'strategy_name']
)

RISK_METRICS = Gauge(
    'trading_platform_risk_metrics',
    'Various risk metrics',
    ['metric_type', 'portfolio_id']
)

SOCIAL_SENTIMENT = Gauge(
    'trading_platform_social_sentiment_score',
    'Social sentiment scores for stocks',
    ['symbol', 'platform']
)

MODEL_ACCURACY = Gauge(
    'trading_platform_model_accuracy_percent',
    'ML Model accuracy percentage',
    ['model_id', 'model_type']
)

ACTIVE_CONNECTIONS = Gauge(
    'trading_platform_active_connections',
    'Number of active WebSocket connections'
)

API_REQUEST_DURATION = Histogram(
    'trading_platform_api_request_duration_seconds',
    'Time spent processing API requests',
    ['method', 'endpoint', 'status_code']
)

MARKET_DATA_LATENCY = Histogram(
    'trading_platform_market_data_latency_seconds',
    'Market data feed latency',
    ['feed_source']
)

class TradingMetrics:
    @staticmethod
    def record_trade(symbol: str, side: str, status: str):
        TRADES_TOTAL.labels(symbol=symbol, side=side, status=status).inc()
    
    @staticmethod
    def update_portfolio_value(portfolio_id: str, portfolio_name: str, value: float):
        PORTFOLIO_VALUE.labels(portfolio_id=portfolio_id, portfolio_name=portfolio_name).set(value)
    
    @staticmethod
    def record_trade_execution_time(symbol: str, side: str, duration: float):
        TRADE_EXECUTION_TIME.labels(symbol=symbol, side=side).observe(duration)
    
    @staticmethod
    def update_strategy_performance(strategy_id: str, strategy_name: str, performance: float):
        STRATEGY_PERFORMANCE.labels(strategy_id=strategy_id, strategy_name=strategy_name).set(performance)
    
    @staticmethod
    def update_risk_metric(metric_type: str, portfolio_id: str, value: float):
        RISK_METRICS.labels(metric_type=metric_type, portfolio_id=portfolio_id).set(value)
    
    @staticmethod
    def update_social_sentiment(symbol: str, platform: str, score: float):
        SOCIAL_SENTIMENT.labels(symbol=symbol, platform=platform).set(score)
    
    @staticmethod
    def update_model_accuracy(model_id: str, model_type: str, accuracy: float):
        MODEL_ACCURACY.labels(model_id=model_id, model_type=model_type).set(accuracy)
    
    @staticmethod
    def set_active_connections(count: int):
        ACTIVE_CONNECTIONS.set(count)
    
    @staticmethod
    def record_market_data_latency(feed_source: str, latency: float):
        MARKET_DATA_LATENCY.labels(feed_source=feed_source).observe(latency) 