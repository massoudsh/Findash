#!/usr/bin/env python3
"""
Standalone Market Data Consumer Service
Consumes from Kafka, processes with Redis, triggers Celery tasks
Exposes Prometheus metrics for Grafana visualization
"""

import os
import sys
import signal
import logging
from src.infrastructure.market_data_stream import IntegratedMarketDataService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the consumer service"""
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    service = IntegratedMarketDataService(
        kafka_servers=kafka_servers,
        redis_url=redis_url
    )
    
    # Start metrics server
    service.start_metrics_server()
    
    # Start consuming
    logger.info("Starting market data consumer service...")
    logger.info(f"Kafka: {kafka_servers}")
    logger.info(f"Redis: {redis_url}")
    logger.info("Prometheus metrics available at http://localhost:8001/metrics")
    
    def signal_handler(sig, frame):
        logger.info("Shutting down consumer service...")
        service.consumer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        service.consumer.start_consuming()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()

