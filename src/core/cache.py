"""
Comprehensive caching system for Quantum Trading Matrix™
Redis-based caching with intelligent strategies for trading data
"""

import json
import pickle
import logging
import asyncio
from typing import Any, Optional, Union, Dict, List, Callable, TypeVar
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import hashlib
import redis.asyncio as redis
from dataclasses import dataclass, asdict

from src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Type variables for generic cache decorators
P = ParamSpec('P')
T = TypeVar('T')


class CacheStrategy(str, Enum):
    """Cache strategy types"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"  # Write to cache and storage
    WRITE_BACK = "write_back"  # Write to cache, delayed storage
    READ_THROUGH = "read_through"  # Cache miss loads from storage


class CacheNamespace(str, Enum):
    """Cache namespace for different data types"""
    MARKET_DATA = "market_data"
    USER_SESSION = "user_session"
    PORTFOLIO = "portfolio"
    TRADE_HISTORY = "trade_history"
    RISK_METRICS = "risk_metrics"
    ML_PREDICTIONS = "ml_predictions"
    API_RESPONSES = "api_responses"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION = "configuration"


@dataclass
class CacheItem:
    """Cache item with metadata"""
    data: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if cache item is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def access(self):
        """Mark item as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class CacheManager:
    """Comprehensive cache manager with Redis backend"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, CacheItem] = {}
        self.max_local_cache_size = 1000
        self.default_ttl = 3600  # 1 hour
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                decode_responses=False,  # Keep binary for pickle
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Cache manager initialized with Redis")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using local cache only: {e}")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_key(self, namespace: CacheNamespace, key: str) -> str:
        """Generate prefixed cache key"""
        return f"qtm:{namespace.value}:{key}"
    
    def _hash_key(self, *args, **kwargs) -> str:
        """Generate hash key from function arguments"""
        key_data = {
            "args": str(args),
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(
        self, 
        namespace: CacheNamespace, 
        key: str, 
        default: Any = None
    ) -> Any:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            # Try Redis first
            if self.redis_client:
                data = await self.redis_client.get(cache_key)
                if data:
                    try:
                        cache_item = pickle.loads(data)
                        if not cache_item.is_expired():
                            cache_item.access()
                            self.stats["hits"] += 1
                            return cache_item.data
                        else:
                            # Remove expired item
                            await self.redis_client.delete(cache_key)
                    except (pickle.PickleError, AttributeError):
                        # Fallback to JSON for simple data
                        return json.loads(data.decode())
            
            # Try local cache
            if cache_key in self.local_cache:
                cache_item = self.local_cache[cache_key]
                if not cache_item.is_expired():
                    cache_item.access()
                    self.stats["hits"] += 1
                    return cache_item.data
                else:
                    del self.local_cache[cache_key]
            
            self.stats["misses"] += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            self.stats["errors"] += 1
            return default
    
    async def set(
        self,
        namespace: CacheNamespace,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            # Create cache item
            cache_item = CacheItem(
                data=value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None,
                tags=tags or []
            )
            
            # Store in Redis
            if self.redis_client:
                try:
                    data = pickle.dumps(cache_item)
                    await self.redis_client.setex(cache_key, ttl if ttl > 0 else 86400, data)
                except pickle.PickleError:
                    # Fallback to JSON for simple data
                    data = json.dumps(value).encode()
                    await self.redis_client.setex(cache_key, ttl if ttl > 0 else 86400, data)
            
            # Store in local cache (with size limit)
            if len(self.local_cache) >= self.max_local_cache_size:
                await self._evict_local_cache()
            
            self.local_cache[cache_key] = cache_item
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def delete(self, namespace: CacheNamespace, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            deleted = False
            
            # Delete from Redis
            if self.redis_client:
                result = await self.redis_client.delete(cache_key)
                deleted = result > 0
            
            # Delete from local cache
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
                deleted = True
            
            if deleted:
                self.stats["deletes"] += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def delete_by_pattern(self, namespace: CacheNamespace, pattern: str) -> int:
        """Delete multiple keys by pattern"""
        try:
            full_pattern = self._generate_key(namespace, pattern)
            deleted_count = 0
            
            # Delete from Redis
            if self.redis_client:
                keys = await self.redis_client.keys(full_pattern)
                if keys:
                    deleted_count += await self.redis_client.delete(*keys)
            
            # Delete from local cache
            to_delete = [k for k in self.local_cache.keys() if k.startswith(full_pattern.replace("*", ""))]
            for key in to_delete:
                del self.local_cache[key]
                deleted_count += 1
            
            self.stats["deletes"] += deleted_count
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern delete error for {pattern}: {e}")
            self.stats["errors"] += 1
            return 0
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete cache items by tags"""
        # This is simplified - a full implementation would maintain tag indexes
        deleted_count = 0
        
        # Check local cache
        to_delete = []
        for key, item in self.local_cache.items():
            if any(tag in item.tags for tag in tags):
                to_delete.append(key)
        
        for key in to_delete:
            del self.local_cache[key]
            deleted_count += 1
        
        # For Redis, you'd need to implement tag indexing
        # This is a simplified version
        
        return deleted_count
    
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Clear all items in a namespace"""
        return await self.delete_by_pattern(namespace, "*")
    
    async def exists(self, namespace: CacheNamespace, key: str) -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(namespace, key)
        
        try:
            # Check Redis
            if self.redis_client:
                exists = await self.redis_client.exists(cache_key)
                if exists:
                    return True
            
            # Check local cache
            if cache_key in self.local_cache:
                item = self.local_cache[cache_key]
                if not item.is_expired():
                    return True
                else:
                    del self.local_cache[cache_key]
            
            return False
            
        except Exception as e:
            logger.error(f"Cache exists error for {cache_key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        redis_info = {}
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
            except Exception as e:
                logger.error(f"Error getting Redis info: {e}")
        
        return {
            "local_cache_size": len(self.local_cache),
            "local_cache_max_size": self.max_local_cache_size,
            "redis_connected": self.redis_client is not None,
            "redis_info": redis_info,
            "stats": self.stats.copy()
        }
    
    async def _evict_local_cache(self):
        """Evict items from local cache using LRU strategy"""
        if not self.local_cache:
            return
        
        # Sort by last accessed time (LRU)
        sorted_items = sorted(
            self.local_cache.items(),
            key=lambda x: x[1].last_accessed or datetime.min
        )
        
        # Remove oldest 10% of items
        items_to_remove = max(1, len(sorted_items) // 10)
        for i in range(items_to_remove):
            key, _ = sorted_items[i]
            del self.local_cache[key]
    
    def cache_response(
        self,
        namespace: CacheNamespace,
        key_prefix: str = "",
        ttl: int = 3600,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator to cache function responses
        
        Args:
            namespace: Cache namespace
            key_prefix: Prefix for cache keys
            ttl: Time to live in seconds
            key_func: Optional function to generate cache key from arguments
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                elif key_prefix:
                    # Use key_prefix and function arguments
                    key_parts = [key_prefix]
                    # Extract symbol or other key parameters from kwargs
                    if "symbol" in kwargs:
                        key_parts.append(kwargs["symbol"])
                    elif args:
                        key_parts.append(str(args[0]))
                    cache_key = ":".join(key_parts)
                else:
                    cache_key = f"{func.__name__}:{self._hash_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = await self.get(namespace, cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(namespace, cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # For sync functions, we need to run in an event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


# Global cache manager instance
cache_manager = CacheManager()


# Cache decorators
def cached(
    namespace: CacheNamespace,
    ttl: int = 3600,
    key_func: Optional[Callable] = None,
    tags: Optional[List[str]] = None
):
    """
    Cache decorator for functions
    
    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
        tags: Tags for cache invalidation
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache_manager._hash_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(namespace, cache_key, result, ttl, tags)
            return result
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # For sync functions, we need to run in an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_key_generator(*key_parts):
    """Generate cache key from parts"""
    def generator(*args, **kwargs):
        parts = []
        for part in key_parts:
            if callable(part):
                parts.append(str(part(*args, **kwargs)))
            else:
                parts.append(str(part))
        return ":".join(parts)
    return generator


# Specialized cache functions for trading data
class TradingCache:
    """Specialized caching for trading data"""
    
    @staticmethod
    async def cache_market_data(symbol: str, data: Dict[str, Any], ttl: int = 60):
        """Cache market data for a symbol"""
        await cache_manager.set(
            CacheNamespace.MARKET_DATA,
            f"current:{symbol}",
            data,
            ttl,
            tags=["market_data", symbol]
        )
    
    @staticmethod
    async def get_market_data(symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for a symbol"""
        return await cache_manager.get(CacheNamespace.MARKET_DATA, f"current:{symbol}")
    
    @staticmethod
    async def cache_portfolio(user_id: str, portfolio_data: Dict[str, Any], ttl: int = 300):
        """Cache user portfolio data"""
        await cache_manager.set(
            CacheNamespace.PORTFOLIO,
            f"user:{user_id}",
            portfolio_data,
            ttl,
            tags=["portfolio", f"user:{user_id}"]
        )
    
    @staticmethod
    async def get_portfolio(user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached portfolio for a user"""
        return await cache_manager.get(CacheNamespace.PORTFOLIO, f"user:{user_id}")
    
    @staticmethod
    async def invalidate_user_cache(user_id: str):
        """Invalidate all cache entries for a user"""
        await cache_manager.delete_by_pattern(CacheNamespace.PORTFOLIO, f"user:{user_id}*")
        await cache_manager.delete_by_pattern(CacheNamespace.TRADE_HISTORY, f"user:{user_id}*")
    
    @staticmethod
    async def cache_ml_prediction(model_name: str, input_hash: str, prediction: Any, ttl: int = 1800):
        """Cache ML model prediction"""
        await cache_manager.set(
            CacheNamespace.ML_PREDICTIONS,
            f"{model_name}:{input_hash}",
            prediction,
            ttl,
            tags=["ml_predictions", model_name]
        )
    
    @staticmethod
    async def get_ml_prediction(model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached ML prediction"""
        return await cache_manager.get(CacheNamespace.ML_PREDICTIONS, f"{model_name}:{input_hash}")


# Rate limiting cache
class RateLimitCache:
    """Rate limiting using cache"""
    
    @staticmethod
    async def check_rate_limit(
        identifier: str, 
        limit: int, 
        window: int,
        increment: bool = True
    ) -> tuple[bool, int, int]:
        """
        Check rate limit for an identifier
        
        Returns: (is_allowed, current_count, time_remaining)
        """
        key = f"rate_limit:{identifier}"
        current_count = await cache_manager.get(CacheNamespace.RATE_LIMIT, key) or 0
        
        if current_count >= limit:
            # Get TTL from Redis if available
            if cache_manager.redis_client:
                try:
                    cache_key = cache_manager._generate_key(CacheNamespace.RATE_LIMIT, key)
                    ttl = await cache_manager.redis_client.ttl(cache_key)
                    return False, current_count, max(0, ttl)
                except Exception:
                    pass
            return False, current_count, window
        
        if increment:
            await cache_manager.set(CacheNamespace.RATE_LIMIT, key, current_count + 1, window)
            return True, current_count + 1, window
        
        return True, current_count, window


# Initialize cache on startup
async def initialize_cache():
    """Initialize cache manager"""
    await cache_manager.initialize()


async def cleanup_cache():
    """Cleanup cache on shutdown"""
    await cache_manager.close() 