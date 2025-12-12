"""
Real-time Market Data Streaming with Redis Streams, Redis Cache/PubSub, and Monitoring

The event stream is implemented with:
- Redis Streams (XADD / consumer groups via XREADGROUP)
- Redis cache (SETEX for latest values)
- Optional Redis Pub/Sub fanout (channels: tasks:market_data:{symbol})
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from celery import current_app as celery_app

logger = logging.getLogger(__name__)

# ============================================
# PROMETHEUS METRICS
# ============================================

# Redis Streams metrics
redis_stream_messages_produced = Counter(
    'redis_stream_messages_produced_total',
    'Total number of messages produced to Redis Streams',
    ['stream', 'symbol']
)

redis_stream_messages_consumed = Counter(
    'redis_stream_messages_consumed_total',
    'Total number of messages consumed from Redis Streams',
    ['stream', 'symbol']
)

redis_stream_produce_latency = Histogram(
    'redis_stream_produce_latency_seconds',
    'Latency of producing messages to Redis Streams',
    ['stream'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

redis_stream_consume_latency = Histogram(
    'redis_stream_consume_latency_seconds',
    'Latency of consuming messages from Redis Streams',
    ['stream'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Redis metrics
redis_cache_hits = Counter(
    'redis_cache_hits_total',
    'Total number of Redis cache hits',
    ['cache_type']
)

redis_cache_misses = Counter(
    'redis_cache_misses_total',
    'Total number of Redis cache misses',
    ['cache_type']
)

redis_pubsub_messages = Counter(
    'redis_pubsub_messages_total',
    'Total number of Redis pub/sub messages',
    ['channel']
)

# Market data metrics
market_data_updates = Counter(
    'market_data_updates_total',
    'Total number of market data updates processed',
    ['symbol', 'exchange']
)

market_data_lag = Gauge(
    'market_data_lag_seconds',
    'Lag between market data timestamp and processing time',
    ['symbol']
)

class MarketDataStreamProducer:
    """Produces market data events to a Redis Stream."""

    def __init__(self, redis_url: str = 'redis://localhost:6379/0', stream_key: str = 'market-data-stream'):
        self.redis_url = redis_url
        self.stream_key = stream_key
        self.redis_client: Optional[redis.Redis] = None

    def connect(self):
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        # Basic connectivity check
        self.redis_client.ping()
        logger.info(f"Connected to Redis at {self.redis_url} (stream={self.stream_key})")

    def produce_market_data(self, symbol: str, price: float, volume: int, exchange: str = 'NASDAQ') -> bool:
        if not self.redis_client:
            self.connect()

        start_time = time.time()
        try:
            event = {
                'symbol': symbol,
                'price': str(float(price)),
                'volume': str(int(volume)),
                'exchange': exchange,
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': 'price_update',
            }

            # XADD expects field/value pairs; keep a bounded stream
            self.redis_client.xadd(self.stream_key, event, maxlen=10000, approximate=True)

            latency = time.time() - start_time
            redis_stream_produce_latency.labels(stream=self.stream_key).observe(latency)
            redis_stream_messages_produced.labels(stream=self.stream_key, symbol=symbol).inc()
            return True
        except Exception as e:
            logger.error(f"Error producing market data to Redis Stream: {e}")
            return False

class MarketDataStreamConsumer:
    """Consumes market data from a Redis Stream (consumer group) and processes it."""

    def __init__(
        self,
        redis_url: str = 'redis://localhost:6379/0',
        stream_key: str = 'market-data-stream',
        group_name: str = 'market-data-processors',
        consumer_name: str = 'consumer-1',
        redis_client: Optional[redis.Redis] = None,
    ):
        self.redis_url = redis_url
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.redis_client = redis_client or redis.from_url(redis_url, decode_responses=True)
        self.running = False

    def connect(self):
        # Ensure stream + consumer group exist
        try:
            self.redis_client.xgroup_create(self.stream_key, self.group_name, id='0-0', mkstream=True)
            logger.info(f"Created consumer group {self.group_name} on stream {self.stream_key}")
        except Exception as e:
            # BUSYGROUP means it already exists
            if "BUSYGROUP" not in str(e):
                raise
        logger.info(f"Connected Redis Stream consumer (stream={self.stream_key}, group={self.group_name}, consumer={self.consumer_name})")
    
    def process_message(self, message: Dict[str, Any]) -> bool:
        """Process a market data message"""
        start_time = time.time()
        
        try:
            symbol = message['symbol']
            price = message['price']
            timestamp = datetime.fromisoformat(message['timestamp'])
            
            # Calculate lag
            lag = (datetime.utcnow() - timestamp).total_seconds()
            market_data_lag.labels(symbol=symbol).set(lag)
            
            # Update Redis cache
            if self.redis_client:
                cache_key = f"market_data:{symbol}:latest"
                self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute TTL
                    json.dumps(message)
                )
                redis_cache_hits.labels(cache_type='market_data').inc()
            
            # Publish to Redis pub/sub for Celery task allocation
            if self.redis_client:
                pubsub_channel = f"tasks:market_data:{symbol}"
                self.redis_client.publish(
                    pubsub_channel,
                    json.dumps({
                        'symbol': symbol,
                        'price': price,
                        'action': 'update_portfolio',
                        'priority': 5
                    })
                )
                redis_pubsub_messages.labels(channel=pubsub_channel).inc()
            
            # Trigger Celery task via Redis pub/sub
            self.trigger_celery_task(symbol, message)
            
            # Track metrics
            latency = time.time() - start_time
            redis_stream_consume_latency.labels(stream=self.stream_key).observe(latency)
            redis_stream_messages_consumed.labels(stream=self.stream_key, symbol=symbol).inc()
            market_data_updates.labels(symbol=symbol, exchange=message.get('exchange', 'UNKNOWN')).inc()
            
            logger.debug(f"Processed market data for {symbol}: ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    
    def trigger_celery_task(self, symbol: str, message: Dict[str, Any]):
        """Trigger Celery task via Redis pub/sub pattern"""
        try:
            # Use Celery's send_task to trigger via pub/sub
            celery_app.send_task(
                'data_processing.update_market_data',
                args=[symbol, message],
                queue='data_processing',
                routing_key='data_processing',
                exchange='tasks',
                exchange_type='topic'
            )
            logger.debug(f"Triggered Celery task for {symbol}")
        except Exception as e:
            logger.error(f"Error triggering Celery task: {e}")
    
    def start_consuming(self):
        """Start consuming messages from Redis Streams using a consumer group."""
        self.connect()
        self.running = True
        logger.info(f"Starting to consume from Redis Stream: {self.stream_key}")

        try:
            while self.running:
                # Read new messages for this consumer group
                resp = self.redis_client.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={self.stream_key: '>'},
                    count=100,
                    block=1000,
                )

                if not resp:
                    continue

                for _stream, messages in resp:
                    for msg_id, fields in messages:
                        # fields are already strings (decode_responses=True)
                        parsed = {
                            'symbol': fields.get('symbol'),
                            'price': float(fields.get('price', 0) or 0),
                            'volume': int(float(fields.get('volume', 0) or 0)),
                            'exchange': fields.get('exchange', 'UNKNOWN'),
                            'timestamp': fields.get('timestamp', datetime.utcnow().isoformat()),
                            'event_type': fields.get('event_type', 'price_update'),
                        }

                        ok = self.process_message(parsed)
                        if ok:
                            self.redis_client.xack(self.stream_key, self.group_name, msg_id)
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        except Exception as e:
            logger.error(f"Error consuming Redis Stream messages: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop consuming"""
        self.running = False
        logger.info("Redis Stream consumer stopped")

# ============================================
# INTEGRATED MARKET DATA SERVICE
# ============================================

class IntegratedMarketDataService:
    """
    Complete integration of Redis Streams, Redis Cache/PubSub, Prometheus, and Celery
    Demonstrates real-time market data pipeline
    """
    
    def __init__(self, 
                 redis_url: str = 'redis://localhost:6379/0'):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.producer = MarketDataStreamProducer(redis_url=redis_url)
        self.consumer = MarketDataStreamConsumer(redis_url=redis_url, redis_client=self.redis_client)
        self.metrics_port = 8001
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
    
    def simulate_market_data_stream(self, symbols: list, duration: int = 60):
        """
        Simulate real-time market data stream
        Produces to Redis Streams, consumed by consumer, cached in Redis,
        triggers Celery tasks, and exposes metrics
        """
        logger.info(f"Starting market data simulation for {len(symbols)} symbols")
        
        self.producer.connect()
        self.start_metrics_server()
        
        # Start consumer in background
        import threading
        consumer_thread = threading.Thread(
            target=self.consumer.start_consuming,
            daemon=True
        )
        consumer_thread.start()
        
        # Give consumer time to connect
        time.sleep(2)
        
        # Simulate market data updates
        end_time = time.time() + duration
        update_count = 0
        
        while time.time() < end_time:
            for symbol in symbols:
                # Simulate price movement
                base_price = 100.0 + hash(symbol) % 200
                price_change = (hash(f"{symbol}{time.time()}") % 20 - 10) / 100
                price = base_price * (1 + price_change)
                volume = hash(f"{symbol}{time.time()}") % 10000
                
                # Produce to Redis Stream
                self.producer.produce_market_data(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    exchange='NASDAQ'
                )
                
                update_count += 1
                
                # Rate limit: ~10 updates per second
                time.sleep(0.1)
        
        logger.info(f"Simulation complete. Produced {update_count} market data updates")
        self.consumer.stop()

# ============================================
# CELERY TASK - Process Market Data Updates
# ============================================

@celery_app.task(name='data_processing.update_market_data', bind=True)
def update_market_data_task(self, symbol: str, market_data: Dict[str, Any]):
    """
    Celery task to process market data updates
    Triggered via Redis Streams consumer (and optional Redis pub/sub fanout)
    """
    from src.monitoring.celery_metrics import (
        track_task_execution,
        track_pubsub_message
    )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing market data update for {symbol}")
        
        # Simulate processing (update database, calculate metrics, etc.)
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        
        # Update Redis cache with processed data
        redis_client = redis.from_url('redis://localhost:6379/0', decode_responses=True)
        cache_key = f"processed_data:{symbol}:latest"
        redis_client.setex(
            cache_key,
            600,  # 10 minute TTL
            json.dumps({
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'processed_at': datetime.utcnow().isoformat(),
                'task_id': self.request.id
            })
        )
        
        # Track metrics
        duration = time.time() - start_time
        track_task_execution(
            task_name='update_market_data',
            queue='data_processing',
            duration=duration,
            status='success'
        )
        
        logger.info(f"Successfully processed market data for {symbol}")
        return {'status': 'success', 'symbol': symbol, 'price': price}
        
    except Exception as e:
        logger.error(f"Error processing market data: {e}")
        track_task_execution(
            task_name='update_market_data',
            queue='data_processing',
            duration=time.time() - start_time,
            status='failed'
        )
        raise

# ============================================
# USAGE EXAMPLE
# ============================================

def run_integrated_example():
    """Run the complete integrated example"""
    service = IntegratedMarketDataService(
        redis_url='redis://localhost:6379/0'
    )
    
    # Symbols to stream
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Run for 60 seconds
    service.simulate_market_data_stream(symbols, duration=60)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_integrated_example()

