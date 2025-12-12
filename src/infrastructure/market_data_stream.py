"""
Real-time Market Data Streaming with Kafka, Redis, and Monitoring
Demonstrates integration of Kafka, Redis, Prometheus, Grafana, and Flower
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from celery import current_app as celery_app

logger = logging.getLogger(__name__)

# ============================================
# PROMETHEUS METRICS
# ============================================

# Kafka metrics
kafka_messages_produced = Counter(
    'kafka_messages_produced_total',
    'Total number of messages produced to Kafka',
    ['topic', 'symbol']
)

kafka_messages_consumed = Counter(
    'kafka_messages_consumed_total',
    'Total number of messages consumed from Kafka',
    ['topic', 'symbol']
)

kafka_produce_latency = Histogram(
    'kafka_produce_latency_seconds',
    'Latency of producing messages to Kafka',
    ['topic'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

kafka_consume_latency = Histogram(
    'kafka_consume_latency_seconds',
    'Latency of consuming messages from Kafka',
    ['topic'],
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

# ============================================
# KAFKA PRODUCER - Market Data Stream
# ============================================

class MarketDataKafkaProducer:
    """Produces market data events to Kafka topics"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[KafkaProducer] = None
        self.topic = 'market-data-stream'
    
    def connect(self):
        """Connect to Kafka"""
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed. Install with: pip install kafka-python")
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def produce_market_data(self, symbol: str, price: float, volume: int, 
                           exchange: str = 'NASDAQ') -> bool:
        """Produce market data event to Kafka"""
        if not self.producer:
            self.connect()
        
        start_time = time.time()
        
        try:
            event = {
                'symbol': symbol,
                'price': float(price),
                'volume': volume,
                'exchange': exchange,
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': 'price_update'
            }
            
            # Send to Kafka with symbol as key for partitioning
            future = self.producer.send(
                self.topic,
                key=symbol,
                value=event
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            # Track metrics
            latency = time.time() - start_time
            kafka_produce_latency.labels(topic=self.topic).observe(latency)
            kafka_messages_produced.labels(topic=self.topic, symbol=symbol).inc()
            
            logger.debug(
                f"Produced market data for {symbol} to topic {record_metadata.topic} "
                f"partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error producing message: {e}")
            return False
        except Exception as e:
            logger.error(f"Error producing market data: {e}")
            return False
    
    def close(self):
        """Close producer"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")

# ============================================
# KAFKA CONSUMER - Process Market Data
# ============================================

class MarketDataKafkaConsumer:
    """Consumes market data from Kafka and processes it"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', 
                 redis_client: Optional[redis.Redis] = None):
        self.bootstrap_servers = bootstrap_servers
        self.consumer: Optional[KafkaConsumer] = None
        self.topic = 'market-data-stream'
        self.redis_client = redis_client
        self.running = False
    
    def connect(self):
        """Connect to Kafka"""
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed. Install with: pip install kafka-python")
        
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id='market-data-processors',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000
            )
            logger.info(f"Connected to Kafka consumer at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka consumer: {e}")
            raise
    
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
            kafka_consume_latency.labels(topic=self.topic).observe(latency)
            kafka_messages_consumed.labels(topic=self.topic, symbol=symbol).inc()
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
        """Start consuming messages from Kafka"""
        if not self.consumer:
            self.connect()
        
        self.running = True
        logger.info(f"Starting to consume from topic: {self.topic}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                value = message.value
                if value:
                    self.process_message(value)
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop consuming"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")

# ============================================
# INTEGRATED MARKET DATA SERVICE
# ============================================

class IntegratedMarketDataService:
    """
    Complete integration of Kafka, Redis, Prometheus, and Celery
    Demonstrates real-time market data pipeline
    """
    
    def __init__(self, 
                 kafka_servers: str = 'localhost:9092',
                 redis_url: str = 'redis://localhost:6379/0'):
        self.kafka_producer = MarketDataKafkaProducer(kafka_servers)
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.consumer = MarketDataKafkaConsumer(kafka_servers, self.redis_client)
        self.metrics_port = 8001
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
    
    def simulate_market_data_stream(self, symbols: list, duration: int = 60):
        """
        Simulate real-time market data stream
        Produces to Kafka, consumed by consumer, cached in Redis,
        triggers Celery tasks, and exposes metrics
        """
        logger.info(f"Starting market data simulation for {len(symbols)} symbols")
        
        self.kafka_producer.connect()
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
                
                # Produce to Kafka
                self.kafka_producer.produce_market_data(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    exchange='NASDAQ'
                )
                
                update_count += 1
                
                # Rate limit: ~10 updates per second
                time.sleep(0.1)
        
        logger.info(f"Simulation complete. Produced {update_count} market data updates")
        self.kafka_producer.close()
        self.consumer.stop()

# ============================================
# CELERY TASK - Process Market Data Updates
# ============================================

@celery_app.task(name='data_processing.update_market_data', bind=True)
def update_market_data_task(self, symbol: str, market_data: Dict[str, Any]):
    """
    Celery task to process market data updates
    Triggered via Redis pub/sub from Kafka consumer
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
        kafka_servers='localhost:9092',
        redis_url='redis://localhost:6379/0'
    )
    
    # Symbols to stream
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Run for 60 seconds
    service.simulate_market_data_stream(symbols, duration=60)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_integrated_example()

