"""
Enhanced Free Data Sources with Intelligent Fallbacks
Combines multiple free APIs for maximum reliability without paid services
"""

import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
import requests
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    symbol: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    source: str
    confidence: float = 1.0  # Data quality confidence 0-1

class EnhancedFreeDataAggregator:
    """
    Enhanced data aggregator with intelligent source selection,
    caching, and fallback mechanisms using only free data sources
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache
        self.source_reliability = {}  # Track source success rates
        self.request_counts = {}  # Track API usage
        
        # Free data sources with rate limits and reliability scores
        self.sources = {
            'yahoo_finance': {
                'reliability': 0.95,
                'rate_limit': 100,  # requests per minute
                'priority': 1,
                'delay': 0.1,
                'requires_key': False
            },
            'yahoo_query1': {
                'reliability': 0.90,
                'rate_limit': 200,
                'priority': 2,
                'delay': 0.05,
                'requires_key': False
            },
            'coingecko': {
                'reliability': 0.85,
                'rate_limit': 50,
                'priority': 3,
                'delay': 1.2,
                'requires_key': False
            },
            'finnhub_free': {
                'reliability': 0.80,
                'rate_limit': 60,
                'priority': 4,
                'delay': 1.0,
                'requires_key': True,
                'api_key': 'demo'
            },
            'alpha_vantage_free': {
                'reliability': 0.75,
                'rate_limit': 5,
                'priority': 5,
                'delay': 12.0,
                'requires_key': True,
                'api_key': 'demo'
            },
            'iex_cloud_free': {
                'reliability': 0.70,
                'rate_limit': 100,
                'priority': 6,
                'delay': 0.6,
                'requires_key': False
            }
        }
        
        # Initialize source reliability tracking
        for source in self.sources:
            self.source_reliability[source] = self.sources[source]['reliability']
    
    async def get_market_data(self, symbol: str, use_cache: bool = True) -> Optional[MarketDataPoint]:
        """Get market data with intelligent source selection"""
        
        if use_cache:
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                logger.debug(f"Cache hit for {symbol}")
                return cached_data
        
        # Get best available sources
        available_sources = self._get_available_sources()
        
        for source_name in available_sources:
            try:
                await self._apply_rate_limit(source_name)
                data = await self._fetch_from_source(source_name, symbol)
                
                if data and data.price > 0:
                    # Update source reliability (success)
                    self._update_source_reliability(source_name, True)
                    self._cache_data(symbol, data)
                    logger.info(f"✅ Got data for {symbol} from {source_name}")
                    return data
                    
            except Exception as e:
                logger.warning(f"❌ {source_name} failed for {symbol}: {e}")
                self._update_source_reliability(source_name, False)
                continue
        
        logger.error(f"🚨 All sources failed for {symbol}")
        return None
    
    async def get_multiple_symbols(self, symbols: List[str], max_concurrent: int = 5) -> Dict[str, MarketDataPoint]:
        """Get data for multiple symbols with concurrency control"""
        
        # Split into batches to avoid overwhelming APIs
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return await self.get_market_data(symbol)
        
        # Create tasks in batches
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(fetch_with_semaphore(symbol))
            tasks.append((symbol, task))
            
            # Add delay between task creation to spread load
            if len(tasks) % 3 == 0:
                await asyncio.sleep(0.1)
        
        # Gather results
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
        
        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} symbols")
        return results
    
    def _get_available_sources(self) -> List[str]:
        """Get sources sorted by reliability and availability"""
        
        # Filter sources by rate limits and reliability
        available = []
        for source_name, config in self.sources.items():
            current_reliability = self.source_reliability[source_name]
            
            # Skip sources with very low reliability
            if current_reliability < 0.3:
                continue
                
            # Check rate limit
            if self._is_rate_limited(source_name):
                continue
                
            available.append((source_name, current_reliability))
        
        # Sort by reliability (descending)
        available.sort(key=lambda x: x[1], reverse=True)
        return [source[0] for source in available]
    
    def _is_rate_limited(self, source_name: str) -> bool:
        """Check if source is currently rate limited"""
        
        if source_name not in self.request_counts:
            return False
            
        config = self.sources[source_name]
        current_time = time.time()
        
        # Clean old requests
        self.request_counts[source_name] = [
            req_time for req_time in self.request_counts[source_name]
            if current_time - req_time < 60  # Last minute
        ]
        
        return len(self.request_counts[source_name]) >= config['rate_limit']
    
    async def _apply_rate_limit(self, source_name: str):
        """Apply rate limiting for source"""
        
        config = self.sources[source_name]
        
        # Record this request
        if source_name not in self.request_counts:
            self.request_counts[source_name] = []
        
        self.request_counts[source_name].append(time.time())
        
        # Apply delay
        await asyncio.sleep(config['delay'])
    
    def _update_source_reliability(self, source_name: str, success: bool):
        """Update source reliability based on success/failure"""
        
        current_reliability = self.source_reliability[source_name]
        
        if success:
            # Increase reliability slightly on success
            self.source_reliability[source_name] = min(1.0, current_reliability + 0.01)
        else:
            # Decrease reliability more on failure
            self.source_reliability[source_name] = max(0.1, current_reliability - 0.05)
    
    async def _fetch_from_source(self, source_name: str, symbol: str) -> Optional[MarketDataPoint]:
        """Fetch data from specific source"""
        
        if source_name == 'yahoo_finance':
            return await self._fetch_yahoo_finance(symbol)
        elif source_name == 'yahoo_query1':
            return await self._fetch_yahoo_query1(symbol)
        elif source_name == 'coingecko':
            return await self._fetch_coingecko(symbol)
        elif source_name == 'finnhub_free':
            return await self._fetch_finnhub_free(symbol)
        elif source_name == 'alpha_vantage_free':
            return await self._fetch_alpha_vantage_free(symbol)
        elif source_name == 'iex_cloud_free':
            return await self._fetch_iex_cloud_free(symbol)
        
        return None
    
    async def _fetch_yahoo_finance(self, symbol: str) -> Optional[MarketDataPoint]:
        """Primary Yahoo Finance using yfinance library"""
        try:
            # Use thread pool for blocking yfinance calls
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=3) as executor:
                ticker = await loop.run_in_executor(executor, yf.Ticker, symbol)
                info = await loop.run_in_executor(lambda: ticker.info)
                hist = await loop.run_in_executor(
                    lambda: ticker.history(period="1d", interval="1m")
                )
            
            if hist.empty or 'regularMarketPrice' not in info:
                return None
            
            latest = hist.iloc[-1]
            current_price = info.get('regularMarketPrice', latest['Close'])
            
            # Calculate change
            day_open = info.get('regularMarketOpen', latest['Open'])
            change = current_price - day_open
            change_percent = (change / day_open * 100) if day_open > 0 else 0
            
            return MarketDataPoint(
                symbol=symbol,
                price=float(current_price),
                open=float(day_open),
                high=float(info.get('dayHigh', latest['High'])),
                low=float(info.get('dayLow', latest['Low'])),
                volume=int(info.get('regularMarketVolume', latest['Volume'])),
                change=float(change),
                change_percent=float(change_percent),
                timestamp=datetime.now(),
                source='yahoo_finance',
                confidence=0.95
            )
            
        except Exception as e:
            logger.debug(f"Yahoo Finance failed for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_query1(self, symbol: str) -> Optional[MarketDataPoint]:
        """Alternative Yahoo Finance using direct API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data.get('chart', {}).get('result'):
                        return None
                    
                    result = data['chart']['result'][0]
                    meta = result['meta']
                    
                    current_price = meta.get('regularMarketPrice', 0)
                    if current_price <= 0:
                        return None
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        price=float(current_price),
                        open=float(meta.get('regularMarketOpen', current_price)),
                        high=float(meta.get('regularMarketDayHigh', current_price)),
                        low=float(meta.get('regularMarketDayLow', current_price)),
                        volume=int(meta.get('regularMarketVolume', 0)),
                        change=float(meta.get('regularMarketChange', 0)),
                        change_percent=float(meta.get('regularMarketChangePercent', 0)),
                        timestamp=datetime.now(),
                        source='yahoo_query1',
                        confidence=0.90
                    )
                    
        except Exception as e:
            logger.debug(f"Yahoo Query1 failed for {symbol}: {e}")
            return None
    
    async def _fetch_coingecko(self, symbol: str) -> Optional[MarketDataPoint]:
        """Free crypto data from CoinGecko"""
        try:
            # Convert symbol to CoinGecko format
            crypto_id = symbol.lower().replace('-usd', '').replace('btc', 'bitcoin').replace('eth', 'ethereum')
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': crypto_id,
                    'vs_currencies': 'usd',
                    'include_24hr_change': 'true',
                    'include_24hr_vol': 'true'
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if crypto_id not in data:
                        return None
                    
                    crypto_data = data[crypto_id]
                    current_price = crypto_data.get('usd', 0)
                    
                    if current_price <= 0:
                        return None
                    
                    change_24h = crypto_data.get('usd_24h_change', 0)
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        price=float(current_price),
                        open=float(current_price * (1 - change_24h/100)),
                        high=float(current_price * 1.02),  # Estimate
                        low=float(current_price * 0.98),   # Estimate
                        volume=int(crypto_data.get('usd_24h_vol', 0)),
                        change=float(current_price * change_24h / 100),
                        change_percent=float(change_24h),
                        timestamp=datetime.now(),
                        source='coingecko',
                        confidence=0.85
                    )
                    
        except Exception as e:
            logger.debug(f"CoinGecko failed for {symbol}: {e}")
            return None
    
    async def _fetch_finnhub_free(self, symbol: str) -> Optional[MarketDataPoint]:
        """Free Finnhub API (60 calls/minute)"""
        try:
            api_key = self.sources['finnhub_free']['api_key']
            
            async with aiohttp.ClientSession() as session:
                url = f"https://finnhub.io/api/v1/quote"
                params = {'symbol': symbol, 'token': api_key}
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data or 'c' not in data or data['c'] <= 0:
                        return None
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        price=float(data['c']),
                        open=float(data.get('o', data['c'])),
                        high=float(data.get('h', data['c'])),
                        low=float(data.get('l', data['c'])),
                        volume=0,  # Not available in free tier
                        change=float(data.get('d', 0)),
                        change_percent=float(data.get('dp', 0)),
                        timestamp=datetime.now(),
                        source='finnhub_free',
                        confidence=0.80
                    )
                    
        except Exception as e:
            logger.debug(f"Finnhub free failed for {symbol}: {e}")
            return None
    
    async def _fetch_alpha_vantage_free(self, symbol: str) -> Optional[MarketDataPoint]:
        """Free Alpha Vantage API (5 calls/minute)"""
        try:
            api_key = self.sources['alpha_vantage_free']['api_key']
            
            async with aiohttp.ClientSession() as session:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': api_key
                }
                
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'Global Quote' not in data:
                        return None
                    
                    quote = data['Global Quote']
                    
                    if not quote or '05. price' not in quote:
                        return None
                    
                    price = float(quote['05. price'])
                    if price <= 0:
                        return None
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        price=price,
                        open=float(quote.get('02. open', price)),
                        high=float(quote.get('03. high', price)),
                        low=float(quote.get('04. low', price)),
                        volume=int(quote.get('06. volume', 0)),
                        change=float(quote.get('09. change', 0)),
                        change_percent=float(quote.get('10. change percent', '0%').replace('%', '')),
                        timestamp=datetime.now(),
                        source='alpha_vantage_free',
                        confidence=0.75
                    )
                    
        except Exception as e:
            logger.debug(f"Alpha Vantage free failed for {symbol}: {e}")
            return None
    
    async def _fetch_iex_cloud_free(self, symbol: str) -> Optional[MarketDataPoint]:
        """IEX Cloud free tier"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote"
                params = {'token': 'demo'}
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data or 'latestPrice' not in data:
                        return None
                    
                    price = float(data['latestPrice'])
                    if price <= 0:
                        return None
                    
                    return MarketDataPoint(
                        symbol=symbol,
                        price=price,
                        open=float(data.get('open', price)),
                        high=float(data.get('high', price)),
                        low=float(data.get('low', price)),
                        volume=int(data.get('latestVolume', 0)),
                        change=float(data.get('change', 0)),
                        change_percent=float(data.get('changePercent', 0)) * 100,
                        timestamp=datetime.now(),
                        source='iex_cloud_free',
                        confidence=0.70
                    )
                    
        except Exception as e:
            logger.debug(f"IEX Cloud free failed for {symbol}: {e}")
            return None
    
    def _get_cached_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get cached data if still valid"""
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _cache_data(self, symbol: str, data: MarketDataPoint):
        """Cache market data"""
        self.cache[symbol] = (data, time.time())
    
    def get_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {}
        for source_name, config in self.sources.items():
            status[source_name] = {
                'reliability': self.source_reliability[source_name],
                'original_reliability': config['reliability'],
                'rate_limited': self._is_rate_limited(source_name),
                'requests_last_minute': len(self.request_counts.get(source_name, [])),
                'rate_limit': config['rate_limit']
            }
        return status

# Global enhanced aggregator instance
enhanced_data_aggregator = EnhancedFreeDataAggregator()

async def get_enhanced_market_data(symbols: List[str]) -> Dict[str, MarketDataPoint]:
    """Main function to get enhanced market data"""
    return await enhanced_data_aggregator.get_multiple_symbols(symbols)

async def get_single_market_data(symbol: str) -> Optional[MarketDataPoint]:
    """Get data for a single symbol"""
    return await enhanced_data_aggregator.get_market_data(symbol) 