import redis.asyncio as redis
import asyncio
import logging
import json
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

class RedisPubSubManager:
    """
    Manages the connection and subscription to Redis Pub/Sub channels
    for real-time message passing.
    """
    def __init__(self, host: str = 'localhost', port: int = 6379):
        self._redis_conn = redis.Redis(host=host, port=port, auto_close_connection_pool=False)
        self._pubsub = self._redis_conn.pubsub()

    async def subscribe(self, channel_name: str):
        """Subscribes to a given Redis channel."""
        await self._pubsub.subscribe(channel_name)
        logger.info(f"Subscribed to Redis channel: {channel_name}")

    async def listen(self, on_message: Callable[[Dict], Awaitable[None]]):
        """
        Listens for messages and calls the callback function.
        This is a long-running task that should be started in the background.
        """
        logger.info("Starting Redis listener...")
        while True:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    data = json.loads(message["data"].decode("utf-8"))
                    await on_message(data)
            except Exception as e:
                logger.error(f"Error in Redis listener: {e}", exc_info=True)
                await asyncio.sleep(1) # Avoid tight loop on continuous errors

    async def close(self):
        """Closes the Redis connection."""
        await self._redis_conn.close()
        logger.info("Redis Pub/Sub manager closed.")

# Singleton instance for the application
redis_pubsub = RedisPubSubManager() 