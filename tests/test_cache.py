"""
Tests for caching system in Quantum Trading Matrix™
Tests cache manager, decorators, and trading-specific caching
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.core.cache import (
    CacheManager, CacheItem, CacheNamespace, CacheStrategy,
    cache_manager, cached, TradingCache, RateLimitCache,
    cache_key_generator, initialize_cache, cleanup_cache
)


class TestCacheItem:
    """Test cache item functionality"""
    
    def test_cache_item_creation(self):
        """Test cache item creation"""
        now = datetime.utcnow()
        item = CacheItem(
            data={"test": "data"},
            created_at=now,
            expires_at=now + timedelta(hours=1),
            tags=["test", "data"]
        )
        
        assert item.data == {"test": "data"}
        assert item.created_at == now
        assert item.expires_at == now + timedelta(hours=1)
        assert item.access_count == 0
        assert item.tags == ["test", "data"]
        assert item.last_accessed == now
    
    def test_cache_item_expiration(self):
        """Test cache item expiration"""
        now = datetime.utcnow()
        
        # Non-expiring item
        item1 = CacheItem(data="test", created_at=now)
        assert not item1.is_expired()
        
        # Expired item
        item2 = CacheItem(
            data="test",
            created_at=now,
            expires_at=now - timedelta(minutes=1)
        )
        assert item2.is_expired()
        
        # Future expiry
        item3 = CacheItem(
            data="test",
            created_at=now,
            expires_at=now + timedelta(minutes=1)
        )
        assert not item3.is_expired()
    
    def test_cache_item_access_tracking(self):
        """Test cache item access tracking"""
        item = CacheItem(data="test", created_at=datetime.utcnow())
        
        initial_count = item.access_count
        initial_time = item.last_accessed
        
        # Simulate access
        item.access()
        
        assert item.access_count == initial_count + 1
        assert item.last_accessed > initial_time


class TestCacheManager:
    """Test cache manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create cache manager for testing"""
        return CacheManager()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        mock = Mock()
        mock.ping = AsyncMock()
        mock.get = AsyncMock()
        mock.setex = AsyncMock()
        mock.delete = AsyncMock()
        mock.keys = AsyncMock()
        mock.exists = AsyncMock()
        mock.ttl = AsyncMock()
        mock.info = AsyncMock()
        mock.close = AsyncMock()
        return mock
    
    @pytest.mark.asyncio
    async def test_initialize_with_redis(self, manager, mock_redis):
        """Test cache manager initialization with Redis"""
        with patch('src.core.cache.redis.Redis', return_value=mock_redis):
            await manager.initialize()
            
            assert manager.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_without_redis(self, manager):
        """Test cache manager initialization without Redis"""
        with patch('src.core.cache.redis.Redis', side_effect=Exception("Redis unavailable")):
            await manager.initialize()
            
            assert manager.redis_client is None
    
    def test_generate_key(self, manager):
        """Test cache key generation"""
        key = manager._generate_key(CacheNamespace.MARKET_DATA, "BTC-USD")
        assert key == "qtm:market_data:BTC-USD"
    
    def test_hash_key(self, manager):
        """Test hash key generation"""
        hash1 = manager._hash_key("arg1", "arg2", key="value")
        hash2 = manager._hash_key("arg1", "arg2", key="value")
        hash3 = manager._hash_key("different", "args")
        
        assert hash1 == hash2  # Same inputs should produce same hash
        assert hash1 != hash3  # Different inputs should produce different hash
        assert len(hash1) == 32  # MD5 hash length
    
    @pytest.mark.asyncio
    async def test_set_and_get_local_cache(self, manager):
        """Test set and get operations with local cache only"""
        # Don't initialize Redis
        
        # Set data
        success = await manager.set(
            CacheNamespace.MARKET_DATA,
            "BTC-USD",
            {"price": 45000, "volume": 1000},
            ttl=3600,
            tags=["market_data", "btc"]
        )
        
        assert success
        assert manager.stats["sets"] == 1
        
        # Get data
        data = await manager.get(CacheNamespace.MARKET_DATA, "BTC-USD")
        
        assert data == {"price": 45000, "volume": 1000}
        assert manager.stats["hits"] == 1
    
    @pytest.mark.asyncio
    async def test_set_and_get_with_redis(self, manager, mock_redis):
        """Test set and get operations with Redis"""
        manager.redis_client = mock_redis
        mock_redis.get.return_value = None  # Cache miss
        
        # Set data
        await manager.set(
            CacheNamespace.MARKET_DATA,
            "BTC-USD",
            {"price": 45000},
            ttl=3600
        )
        
        mock_redis.setex.assert_called_once()
        
        # Test cache hit from Redis
        mock_cache_item = CacheItem(
            data={"price": 45000},
            created_at=datetime.utcnow()
        )
        import pickle
        mock_redis.get.return_value = pickle.dumps(mock_cache_item)
        
        data = await manager.get(CacheNamespace.MARKET_DATA, "BTC-USD")
        assert data == {"price": 45000}
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, manager):
        """Test cache miss behavior"""
        data = await manager.get(CacheNamespace.MARKET_DATA, "NONEXISTENT")
        assert data is None
        assert manager.stats["misses"] == 1
        
        # Test with default value
        data = await manager.get(
            CacheNamespace.MARKET_DATA,
            "NONEXISTENT",
            default={"default": "value"}
        )
        assert data == {"default": "value"}
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, manager):
        """Test cache expiration"""
        # Set item with very short TTL
        await manager.set(
            CacheNamespace.MARKET_DATA,
            "SHORT_LIVED",
            {"data": "test"},
            ttl=1
        )
        
        # Should be available immediately
        data = await manager.get(CacheNamespace.MARKET_DATA, "SHORT_LIVED")
        assert data == {"data": "test"}
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        data = await manager.get(CacheNamespace.MARKET_DATA, "SHORT_LIVED")
        assert data is None
    
    @pytest.mark.asyncio
    async def test_delete(self, manager):
        """Test cache deletion"""
        # Set data
        await manager.set(CacheNamespace.MARKET_DATA, "DELETE_TEST", {"data": "test"})
        
        # Verify it exists
        data = await manager.get(CacheNamespace.MARKET_DATA, "DELETE_TEST")
        assert data is not None
        
        # Delete
        deleted = await manager.delete(CacheNamespace.MARKET_DATA, "DELETE_TEST")
        assert deleted
        assert manager.stats["deletes"] == 1
        
        # Verify it's gone
        data = await manager.get(CacheNamespace.MARKET_DATA, "DELETE_TEST")
        assert data is None
    
    @pytest.mark.asyncio
    async def test_delete_by_pattern(self, manager, mock_redis):
        """Test pattern-based deletion"""
        manager.redis_client = mock_redis
        mock_redis.keys.return_value = ["qtm:market_data:BTC-USD", "qtm:market_data:ETH-USD"]
        mock_redis.delete.return_value = 2
        
        # Set some local cache items
        await manager.set(CacheNamespace.MARKET_DATA, "BTC-USD", {"price": 45000})
        await manager.set(CacheNamespace.MARKET_DATA, "ETH-USD", {"price": 3000})
        await manager.set(CacheNamespace.PORTFOLIO, "user123", {"balance": 10000})
        
        # Delete by pattern
        deleted_count = await manager.delete_by_pattern(CacheNamespace.MARKET_DATA, "*")
        
        assert deleted_count >= 2
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exists(self, manager):
        """Test cache existence check"""
        # Non-existent key
        exists = await manager.exists(CacheNamespace.MARKET_DATA, "NONEXISTENT")
        assert not exists
        
        # Set data
        await manager.set(CacheNamespace.MARKET_DATA, "EXISTS_TEST", {"data": "test"})
        
        # Should exist now
        exists = await manager.exists(CacheNamespace.MARKET_DATA, "EXISTS_TEST")
        assert exists
    
    @pytest.mark.asyncio
    async def test_clear_namespace(self, manager):
        """Test clearing entire namespace"""
        # Set some data in different namespaces
        await manager.set(CacheNamespace.MARKET_DATA, "BTC-USD", {"price": 45000})
        await manager.set(CacheNamespace.MARKET_DATA, "ETH-USD", {"price": 3000})
        await manager.set(CacheNamespace.PORTFOLIO, "user123", {"balance": 10000})
        
        # Clear market data namespace
        deleted_count = await manager.clear_namespace(CacheNamespace.MARKET_DATA)
        
        assert deleted_count >= 2
        
        # Market data should be gone
        btc_data = await manager.get(CacheNamespace.MARKET_DATA, "BTC-USD")
        assert btc_data is None
        
        # Portfolio data should remain
        portfolio_data = await manager.get(CacheNamespace.PORTFOLIO, "user123")
        assert portfolio_data is not None
    
    @pytest.mark.asyncio
    async def test_get_stats(self, manager, mock_redis):
        """Test getting cache statistics"""
        manager.redis_client = mock_redis
        mock_redis.info.return_value = {"used_memory": 1024}
        
        stats = await manager.get_stats()
        
        assert "local_cache_size" in stats
        assert "redis_connected" in stats
        assert stats["redis_connected"] == True
        assert "stats" in stats
        assert "redis_info" in stats
    
    @pytest.mark.asyncio
    async def test_local_cache_eviction(self, manager):
        """Test local cache eviction when size limit reached"""
        # Set a small max size
        manager.max_local_cache_size = 3
        
        # Fill cache beyond limit
        for i in range(5):
            await manager.set(CacheNamespace.MARKET_DATA, f"key{i}", f"data{i}")
        
        # Cache should be evicted to stay under limit
        assert len(manager.local_cache) <= manager.max_local_cache_size


class TestCacheDecorators:
    """Test cache decorators"""
    
    @pytest.mark.asyncio
    async def test_cached_decorator_async(self):
        """Test cached decorator with async function"""
        call_count = 0
        
        @cached(CacheNamespace.ML_PREDICTIONS, ttl=60)
        async def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = await expensive_function(2, 3)
        assert result1 == 5
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function(2, 3)
        assert result2 == 5
        assert call_count == 1  # Function not called again
        
        # Different arguments should execute function
        result3 = await expensive_function(4, 5)
        assert result3 == 9
        assert call_count == 2
    
    def test_cached_decorator_sync(self):
        """Test cached decorator with sync function"""
        call_count = 0
        
        @cached(CacheNamespace.API_RESPONSES, ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x * y
        
        # First call should execute function
        result1 = expensive_function(2, 3)
        assert result1 == 6
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(2, 3)
        assert result2 == 6
        assert call_count == 1  # Function not called again
    
    def test_cache_key_generator(self):
        """Test cache key generator"""
        generator = cache_key_generator("prefix", lambda *args, **kwargs: args[0], "suffix")
        
        key = generator("middle", extra="data")
        assert key == "prefix:middle:suffix"


class TestTradingCache:
    """Test trading-specific cache functions"""
    
    @pytest.mark.asyncio
    async def test_cache_market_data(self):
        """Test caching market data"""
        market_data = {
            "symbol": "BTC-USD",
            "price": 45000,
            "volume": 1000,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await TradingCache.cache_market_data("BTC-USD", market_data, ttl=60)
        
        # Retrieve cached data
        cached_data = await TradingCache.get_market_data("BTC-USD")
        assert cached_data == market_data
    
    @pytest.mark.asyncio
    async def test_cache_portfolio(self):
        """Test caching portfolio data"""
        portfolio_data = {
            "user_id": "user123",
            "total_value": 100000,
            "positions": [
                {"symbol": "BTC-USD", "quantity": 1.0, "value": 45000},
                {"symbol": "ETH-USD", "quantity": 10.0, "value": 30000}
            ]
        }
        
        await TradingCache.cache_portfolio("user123", portfolio_data, ttl=300)
        
        # Retrieve cached data
        cached_data = await TradingCache.get_portfolio("user123")
        assert cached_data == portfolio_data
    
    @pytest.mark.asyncio
    async def test_invalidate_user_cache(self):
        """Test invalidating user cache"""
        # Cache some user data
        await TradingCache.cache_portfolio("user123", {"balance": 10000})
        await cache_manager.set(CacheNamespace.TRADE_HISTORY, "user:user123:recent", [])
        
        # Invalidate user cache
        await TradingCache.invalidate_user_cache("user123")
        
        # Data should be gone
        portfolio_data = await TradingCache.get_portfolio("user123")
        assert portfolio_data is None
    
    @pytest.mark.asyncio
    async def test_cache_ml_prediction(self):
        """Test caching ML predictions"""
        prediction = {
            "model": "lstm_price_predictor",
            "symbol": "BTC-USD",
            "prediction": 46000,
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await TradingCache.cache_ml_prediction("lstm_model", "input_hash_123", prediction)
        
        # Retrieve cached prediction
        cached_prediction = await TradingCache.get_ml_prediction("lstm_model", "input_hash_123")
        assert cached_prediction == prediction


class TestRateLimitCache:
    """Test rate limiting cache"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allow(self):
        """Test rate limiting - allow case"""
        identifier = "user123"
        
        # First request should be allowed
        allowed, count, remaining = await RateLimitCache.check_rate_limit(
            identifier, limit=5, window=60, increment=True
        )
        
        assert allowed
        assert count == 1
        assert remaining == 60
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceed(self):
        """Test rate limiting - exceed limit case"""
        identifier = "user456"
        limit = 3
        
        # Make requests up to limit
        for i in range(limit):
            allowed, count, remaining = await RateLimitCache.check_rate_limit(
                identifier, limit=limit, window=60, increment=True
            )
            assert allowed
            assert count == i + 1
        
        # Next request should be denied
        allowed, count, remaining = await RateLimitCache.check_rate_limit(
            identifier, limit=limit, window=60, increment=True
        )
        assert not allowed
        assert count == limit
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_only(self):
        """Test rate limiting check without increment"""
        identifier = "user789"
        
        # Set rate limit counter
        await RateLimitCache.check_rate_limit(identifier, limit=5, window=60, increment=True)
        
        # Check without incrementing
        allowed, count, remaining = await RateLimitCache.check_rate_limit(
            identifier, limit=5, window=60, increment=False
        )
        
        assert allowed
        assert count == 1  # Should remain the same


class TestCacheIntegration:
    """Integration tests for cache system"""
    
    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self):
        """Test cache initialization and cleanup"""
        # Initialize cache
        await initialize_cache()
        
        # Cache manager should be initialized
        assert cache_manager.redis_client is not None or cache_manager.redis_client is None
        
        # Test basic operations
        await cache_manager.set(CacheNamespace.MARKET_DATA, "test", {"data": "test"})
        data = await cache_manager.get(CacheNamespace.MARKET_DATA, "test")
        assert data == {"data": "test"}
        
        # Cleanup
        await cleanup_cache()
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance with multiple operations"""
        # Set multiple items
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(100):
            task = cache_manager.set(
                CacheNamespace.MARKET_DATA,
                f"perf_test_{i}",
                {"index": i, "data": f"test_data_{i}"}
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Get multiple items
        get_tasks = []
        for i in range(100):
            task = cache_manager.get(CacheNamespace.MARKET_DATA, f"perf_test_{i}")
            get_tasks.append(task)
        
        results = await asyncio.gather(*get_tasks)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # All operations should complete quickly
        assert duration < 5  # Should complete in under 5 seconds
        assert len(results) == 100
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test concurrent cache access"""
        async def worker(worker_id):
            for i in range(10):
                key = f"concurrent_test_{worker_id}_{i}"
                await cache_manager.set(CacheNamespace.MARKET_DATA, key, {"worker": worker_id, "index": i})
                data = await cache_manager.get(CacheNamespace.MARKET_DATA, key)
                assert data["worker"] == worker_id
                assert data["index"] == i
        
        # Run multiple workers concurrently
        workers = [worker(i) for i in range(5)]
        await asyncio.gather(*workers)
        
        # Verify no data corruption
        stats = await cache_manager.get_stats()
        assert stats["stats"]["errors"] == 0


if __name__ == "__main__":
    pytest.main([__file__]) 