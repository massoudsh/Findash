"""Caching system using Redis."""

import json
from typing import Any, Optional, Union, Callable
from functools import wraps
import redis
from datetime import datetime, timedelta
from core.config import config
from core.logging_config import setup_logging
from core.models import BaseModel
from core.exceptions import CacheError

logger = setup_logging(__name__)

class CacheManager:
    """Redis-based cache manager."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.Redis(
                host=config.get('cache.host', 'localhost'),
                port=config.get('cache.port', 6379),
                db=config.get('cache.db', 0),
                decode_responses=True
            )
            self.default_ttl = config.get('cache.ttl', 3600)  # 1 hour default
            self.redis.ping()  # Test connection
            logger.info("Cache connection established")
        except redis.ConnectionError as e:
            logger.error(f"Cache connection failed: {str(e)}")
            raise CacheError("Failed to connect to Redis")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            bool: Success status
        """
        try:
            if isinstance(value, BaseModel):
                value = value.dict()
            serialized = json.dumps(value)
            return self.redis.setex(
                key,
                ttl or self.default_ttl,
                serialized
            )
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False

    def clear(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern."""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return bool(self.redis.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False

# Create global cache instance
cache = CacheManager()

def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_builder: Optional[Callable] = None
):
    """
    Cache decorator for function results.
    
    Args:
        ttl: Cache TTL in seconds
        key_prefix: Prefix for cache key
        key_builder: Custom function to build cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key building
                arg_str = ':'.join(str(arg) for arg in args)
                kwarg_str = ':'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug(f"Cached new value for key: {cache_key}")
            return result
        return wrapper
    return decorator 