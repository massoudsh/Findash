"""
Real-time BTC Price Tracker
Fetches BTC price from free API (CoinGecko) every 5 seconds
Tracks complete data flow: API → Redis → Database → UI
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import redis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics for API tracking
btc_api_calls_total = Counter(
    'btc_api_calls_total',
    'Total number of BTC API calls',
    ['status', 'source']
)

btc_api_latency_seconds = Histogram(
    'btc_api_latency_seconds',
    'BTC API call latency in seconds',
    ['source'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

btc_price_current = Gauge(
    'btc_price_current_usd',
    'Current BTC price in USD',
    ['source']
)

btc_price_change_24h = Gauge(
    'btc_price_change_24h_percent',
    'BTC 24h price change percentage',
    ['source']
)

btc_api_cache_hits = Counter(
    'btc_api_cache_hits_total',
    'Total BTC API cache hits',
    ['cache_type']
)

btc_api_cache_misses = Counter(
    'btc_api_cache_misses_total',
    'Total BTC API cache misses',
    ['cache_type']
)

def fetch_btc_price_from_coingecko(redis_client = None) -> Dict[str, Any]:
    """
    Fetch BTC price from CoinGecko free API
    Returns price data with metrics tracking
    """
    start_time = time.time()
    source = 'coingecko'
    
    try:
        # Check Redis cache first (1 second cache to avoid rate limits)
        cache_key = 'btc_price:coingecko:latest'
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                btc_api_cache_hits.labels(cache_type='coingecko').inc()
                logger.debug("BTC price from cache")
                return data
        
        # Fetch from CoinGecko API
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if 'bitcoin' not in data:
            raise ValueError("Bitcoin data not found in API response")
        
        btc_data = data['bitcoin']
        price = float(btc_data['usd'])
        change_24h = float(btc_data.get('usd_24h_change', 0))
        volume_24h = float(btc_data.get('usd_24h_vol', 0))
        
        result = {
            'symbol': 'BTC-USD',
            'price': price,
            'change_24h': change_24h,
            'change_24h_percent': change_24h,
            'volume_24h': volume_24h,
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'api_latency_ms': (time.time() - start_time) * 1000
        }
        
        # Update Prometheus metrics
        latency = time.time() - start_time
        btc_api_latency_seconds.labels(source=source).observe(latency)
        btc_price_current.labels(source=source).set(price)
        btc_price_change_24h.labels(source=source).set(change_24h)
        btc_api_calls_total.labels(status='success', source=source).inc()
        
        # Cache in Redis (1 second TTL to allow frequent updates)
        if redis_client:
            redis_client.setex(
                cache_key,
                1,  # 1 second cache
                json.dumps(result)
            )
            btc_api_cache_misses.labels(cache_type='coingecko').inc()
        
        logger.info(f"BTC price fetched: ${price:,.2f} ({change_24h:+.2f}%)")
        return result
        
    except requests.exceptions.RequestException as e:
        latency = time.time() - start_time
        btc_api_latency_seconds.labels(source=source).observe(latency)
        btc_api_calls_total.labels(status='error', source=source).inc()
        logger.error(f"Error fetching BTC price from CoinGecko: {e}")
        raise
    
    except Exception as e:
        latency = time.time() - start_time
        btc_api_latency_seconds.labels(source=source).observe(latency)
        btc_api_calls_total.labels(status='error', source=source).inc()
        logger.error(f"Unexpected error fetching BTC price: {e}")
        raise

def fetch_btc_price_from_binance(redis_client = None) -> Dict[str, Any]:
    """
    Fallback: Fetch BTC price from Binance free API
    """
    start_time = time.time()
    source = 'binance'
    
    try:
        # Check cache
        cache_key = 'btc_price:binance:latest'
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                btc_api_cache_hits.labels(cache_type='binance').inc()
                return data
        
        # Fetch from Binance
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'BTCUSDT'}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        price = float(data['lastPrice'])
        change_24h = float(data['priceChangePercent'])
        volume_24h = float(data['quoteVolume'])
        
        result = {
            'symbol': 'BTC-USD',
            'price': price,
            'change_24h': change_24h,
            'change_24h_percent': change_24h,
            'volume_24h': volume_24h,
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'api_latency_ms': (time.time() - start_time) * 1000
        }
        
        # Update metrics
        latency = time.time() - start_time
        btc_api_latency_seconds.labels(source=source).observe(latency)
        btc_price_current.labels(source=source).set(price)
        btc_price_change_24h.labels(source=source).set(change_24h)
        btc_api_calls_total.labels(status='success', source=source).inc()
        
        # Cache
        if redis_client:
            redis_client.setex(cache_key, 1, json.dumps(result))
            btc_api_cache_misses.labels(cache_type='binance').inc()
        
        logger.info(f"BTC price from Binance: ${price:,.2f} ({change_24h:+.2f}%)")
        return result
        
    except Exception as e:
        latency = time.time() - start_time
        btc_api_latency_seconds.labels(source=source).observe(latency)
        btc_api_calls_total.labels(status='error', source=source).inc()
        logger.error(f"Error fetching BTC price from Binance: {e}")
        raise

def fetch_btc_price(redis_client = None) -> Dict[str, Any]:
    """
    Fetch BTC price with fallback between multiple free APIs
    """
    # Try CoinGecko first
    try:
        return fetch_btc_price_from_coingecko(redis_client)
    except Exception as e:
        logger.warning(f"CoinGecko failed: {e}, trying Binance...")
        # Fallback to Binance
        try:
            return fetch_btc_price_from_binance(redis_client)
        except Exception as e2:
            logger.error(f"All BTC price APIs failed: {e2}")
            raise

