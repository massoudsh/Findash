"""
Redis pub/sub pattern implementation for Celery task allocation
Enables efficient task distribution across workers using Redis pub/sub channels
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import redis
from redis.exceptions import ConnectionError, TimeoutError
from src.core.config import get_settings
from src.monitoring.celery_metrics import (
    track_pubsub_message,
    track_task_allocation,
    update_queue_length
)

logger = logging.getLogger(__name__)
settings = get_settings()


class CeleryPubSubAllocator:
    """
    Manages task allocation using Redis pub/sub pattern
    Distributes tasks to workers based on availability and load
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis.url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.client.PubSub] = None
        self.worker_channels: Dict[str, str] = {}
        self.task_channels: Dict[str, str] = {}
        self._connected = False
    
    def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.pubsub_client = self.redis_client.pubsub()
            self._connected = True
            logger.info("Connected to Redis for pub/sub task allocation")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    def disconnect(self):
        """Disconnect from Redis"""
        if self.pubsub_client:
            self.pubsub_client.close()
        if self.redis_client:
            self.redis_client.close()
        self._connected = False
        logger.info("Disconnected from Redis")
    
    def register_worker(self, worker_name: str, queues: list, capabilities: Dict[str, Any] = None):
        """
        Register a worker for task allocation
        
        Args:
            worker_name: Unique identifier for the worker
            queues: List of queues this worker can handle
            capabilities: Additional worker capabilities (CPU, memory, etc.)
        """
        if not self._connected:
            self.connect()
        
        channel = f"worker:{worker_name}"
        self.worker_channels[worker_name] = channel
        
        worker_info = {
            'name': worker_name,
            'queues': queues,
            'capabilities': capabilities or {},
            'registered_at': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        # Store worker info in Redis
        self.redis_client.hset(
            f"workers:{worker_name}",
            mapping=worker_info
        )
        
        # Subscribe to worker-specific channel
        self.pubsub_client.subscribe(channel)
        
        logger.info(f"Registered worker {worker_name} for queues: {queues}")
    
    def publish_task(self, task_name: str, queue: str, task_data: Dict[str, Any], 
                    priority: int = 5, allocation_method: str = 'pubsub') -> bool:
        """
        Publish a task to Redis pub/sub for worker allocation
        
        Args:
            task_name: Name of the task
            queue: Target queue
            task_data: Task data payload
            priority: Task priority (0-10, higher is more important)
            allocation_method: Method used for allocation
            
        Returns:
            True if task was published successfully
        """
        if not self._connected:
            self.connect()
        
        start_time = datetime.utcnow()
        
        try:
            # Create task message
            task_message = {
                'task_name': task_name,
                'queue': queue,
                'data': task_data,
                'priority': priority,
                'timestamp': datetime.utcnow().isoformat(),
                'allocation_method': allocation_method
            }
            
            # Publish to queue-specific channel
            channel = f"tasks:{queue}"
            self.redis_client.publish(
                channel,
                json.dumps(task_message)
            )
            
            # Also publish to priority channel if high priority
            if priority >= 8:
                priority_channel = f"tasks:priority:{queue}"
                self.redis_client.publish(
                    priority_channel,
                    json.dumps(task_message)
                )
            
            # Track metrics
            latency = (datetime.utcnow() - start_time).total_seconds()
            track_pubsub_message(channel, 'task_published', latency)
            
            # Update queue length
            queue_len = self.redis_client.llen(f"queue:{queue}")
            update_queue_length(queue, queue_len)
            
            logger.debug(f"Published task {task_name} to queue {queue} via pub/sub")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish task {task_name}: {e}")
            return False
    
    def subscribe_to_tasks(self, queue: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to tasks for a specific queue
        
        Args:
            queue: Queue name to subscribe to
            callback: Function to call when a task is received
        """
        if not self._connected:
            self.connect()
        
        channel = f"tasks:{queue}"
        self.task_channels[queue] = channel
        self.pubsub_client.subscribe(channel)
        
        logger.info(f"Subscribed to task channel: {channel}")
        
        # Start listening for messages
        for message in self.pubsub_client.listen():
            if message['type'] == 'message':
                try:
                    task_data = json.loads(message['data'])
                    start_time = datetime.utcnow()
                    
                    # Execute callback
                    callback(task_data)
                    
                    # Track metrics
                    latency = (datetime.utcnow() - start_time).total_seconds()
                    track_pubsub_message(channel, 'task_processed', latency)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode task message: {e}")
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
    
    def allocate_task_to_worker(self, task_name: str, queue: str, 
                               worker_name: str, allocation_method: str = 'direct') -> bool:
        """
        Allocate a specific task to a worker
        
        Args:
            task_name: Name of the task
            queue: Queue name
            worker_name: Target worker name
            allocation_method: Method used for allocation
            
        Returns:
            True if allocation was successful
        """
        if not self._connected:
            self.connect()
        
        start_time = datetime.utcnow()
        
        try:
            channel = self.worker_channels.get(worker_name)
            if not channel:
                logger.warning(f"Worker {worker_name} not registered")
                return False
            
            allocation_message = {
                'task_name': task_name,
                'queue': queue,
                'allocated_at': datetime.utcnow().isoformat(),
                'allocation_method': allocation_method
            }
            
            self.redis_client.publish(
                channel,
                json.dumps(allocation_message)
            )
            
            # Track metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            track_task_allocation(queue, worker_name, allocation_method, duration)
            
            logger.debug(f"Allocated task {task_name} to worker {worker_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate task to worker: {e}")
            return False
    
    def get_worker_status(self, worker_name: str) -> Optional[Dict[str, Any]]:
        """Get current status of a worker"""
        if not self._connected:
            self.connect()
        
        try:
            worker_data = self.redis_client.hgetall(f"workers:{worker_name}")
            return worker_data if worker_data else None
        except Exception as e:
            logger.error(f"Failed to get worker status: {e}")
            return None
    
    def get_queue_stats(self, queue: str) -> Dict[str, Any]:
        """Get statistics for a queue"""
        if not self._connected:
            self.connect()
        
        try:
            queue_len = self.redis_client.llen(f"queue:{queue}")
            consumers = len([w for w, ch in self.worker_channels.items() 
                           if queue in self.worker_channels.get(w, {}).get('queues', [])])
            
            return {
                'queue': queue,
                'length': queue_len,
                'consumers': consumers,
                'channel': f"tasks:{queue}"
            }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'queue': queue, 'length': 0, 'consumers': 0}


# Global allocator instance
_allocator: Optional[CeleryPubSubAllocator] = None


def get_allocator() -> CeleryPubSubAllocator:
    """Get or create the global pub/sub allocator"""
    global _allocator
    if _allocator is None:
        _allocator = CeleryPubSubAllocator()
    return _allocator

