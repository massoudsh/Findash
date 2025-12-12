"""
Standalone Celery metrics exporter service
Runs as a separate service to expose Prometheus metrics for Celery workers
"""

import os
import sys
import time
import logging
import signal
from typing import Optional
from threading import Thread
import redis
from prometheus_client import start_http_server
from src.monitoring.celery_metrics import (
    CELERY_REGISTRY,
    celery_worker_active_tasks,
    celery_worker_pool_size,
    celery_worker_reserved_tasks,
    celery_queue_length,
    celery_queue_consumers,
    update_queue_length,
    update_worker_active_tasks
)
from src.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


class CeleryMetricsCollector:
    """
    Collects metrics from Celery workers and Redis queues
    """
    
    def __init__(self, redis_url: Optional[str] = None, celery_broker_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis.url
        self.celery_broker_url = celery_broker_url or settings.celery.broker_url
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.collector_thread: Optional[Thread] = None
    
    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for metrics collection")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def collect_queue_metrics(self):
        """Collect metrics from Redis queues"""
        if not self.redis_client:
            return
        
        try:
            # Get all queue names from Redis
            queue_keys = self.redis_client.keys('queue:*')
            
            for queue_key in queue_keys:
                queue_name = queue_key.replace('queue:', '')
                queue_length = self.redis_client.llen(queue_key)
                update_queue_length(queue_name, queue_length)
                
                # Get consumer count (workers subscribed to this queue)
                try:
                    # Use PUBSUB NUMSUB to get subscriber count
                    pubsub_info = self.redis_client.execute_command('PUBSUB', 'NUMSUB', f"tasks:{queue_name}")
                    if pubsub_info and len(pubsub_info) >= 2:
                        consumer_count = pubsub_info[1] if isinstance(pubsub_info[1], int) else 0
                        celery_queue_consumers.labels(queue_name=queue_name).set(consumer_count)
                except Exception as e:
                    logger.debug(f"Could not get consumer count for {queue_name}: {e}")
                    # Set to 0 if we can't determine
                    celery_queue_consumers.labels(queue_name=queue_name).set(0)
                    
        except Exception as e:
            logger.error(f"Error collecting queue metrics: {e}")
    
    def collect_worker_metrics(self):
        """Collect metrics from Celery workers"""
        if not self.redis_client:
            return
        
        try:
            # Get all worker keys
            worker_keys = self.redis_client.keys('workers:*')
            
            for worker_key in worker_keys:
                worker_name = worker_key.replace('workers:', '')
                worker_data = self.redis_client.hgetall(worker_key)
                
                if worker_data:
                    # Update worker pool size if available
                    pool_size = worker_data.get('pool_size', 0)
                    if pool_size:
                        celery_worker_pool_size.labels(worker_name=worker_name).set(int(pool_size))
                    
                    # Get active tasks for this worker
                    active_tasks = self.redis_client.keys(f"worker:{worker_name}:active:*")
                    active_count = len(active_tasks)
                    
                    # Get queues this worker is subscribed to
                    queues = worker_data.get('queues', '').split(',') if worker_data.get('queues') else []
                    for queue in queues:
                        if queue:
                            update_worker_active_tasks(worker_name, queue.strip(), active_count)
                            
        except Exception as e:
            logger.error(f"Error collecting worker metrics: {e}")
    
    def collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                self.collect_queue_metrics()
                self.collect_worker_metrics()
                time.sleep(5)  # Collect metrics every 5 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def start(self):
        """Start metrics collection"""
        self.connect()
        self.running = True
        self.collector_thread = Thread(target=self.collect_metrics_loop, daemon=True)
        self.collector_thread.start()
        logger.info("Started Celery metrics collector")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        if self.redis_client:
            self.redis_client.close()
        logger.info("Stopped Celery metrics collector")


def main():
    """Main entry point for the metrics exporter service"""
    port = int(os.getenv('METRICS_PORT', '9540'))
    host = os.getenv('METRICS_HOST', '0.0.0.0')
    
    # Start Prometheus HTTP server
    start_http_server(port, registry=CELERY_REGISTRY, addr=host)
    logger.info(f"Celery metrics exporter started on {host}:{port}")
    
    # Start metrics collector
    collector = CeleryMetricsCollector()
    collector.start()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutting down metrics exporter...")
        collector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == '__main__':
    main()

