"""
Free Real Data Sources for Trading Platform
Implements multiple free APIs to ensure reliable data access
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
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

class FreeDataAggregator:
    """Aggregates data from multiple free sources"""
    
    def __init__(self):
        self.sources = [
            'yahoo_finance',
            'finnhub',
            'alpha_vantage',
            'polygon',
            'fmp',
            'iex_cloud',
            'twelve_data'
        ]
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
    async def get_market_data(self, symbol: str, use_cache: bool = True) -> Optional[MarketData]:
        """Get market data from the first available source"""
        
        if use_cache:
            cached_data = self._get_cached_data(symbol)
            if cached_data:
                return cached_data
        
        # Try sources in order until one works
        for source in self.sources:
            try:
                data = await self._fetch_from_source(source, symbol)
                if data:
                    self._cache_data(symbol, data)
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch from {source} for {symbol}: {e}")
                continue
        
        logger.error(f"All sources failed for {symbol}")
        return None
    
    async def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get data for multiple symbols concurrently"""
        tasks = [self.get_market_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, MarketData):
                data_dict[symbol] = result
            else:
                logger.error(f"Failed to get data for {symbol}: {result}")
        
        return data_dict
    
    def _get_cached_data(self, symbol: str) -> Optional[MarketData]:
        """Get cached data if still valid"""
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _cache_data(self, symbol: str, data: MarketData):
        """Cache market data"""
        self.cache[symbol] = (data, time.time())
    
    async def _fetch_from_source(self, source: str, symbol: str) -> Optional[MarketData]:
        """Fetch data from specific source"""
        
        if source == 'yahoo_finance':
            return await self._fetch_yahoo_finance(symbol)
        elif source == 'finnhub':
            return await self._fetch_finnhub(symbol)
        elif source == 'alpha_vantage':
            return await self._fetch_alpha_vantage(symbol)
        elif source == 'polygon':
            return await self._fetch_polygon(symbol)
        elif source == 'fmp':
            return await self._fetch_fmp(symbol)
        elif source == 'iex_cloud':
            return await self._fetch_iex_cloud(symbol)
        elif source == 'twelve_data':
            return await self._fetch_twelve_data(symbol)
        
        return None
    
    async def _fetch_yahoo_finance(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Yahoo Finance with rate limiting protection"""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
            
            current_price = info.get('regularMarketPrice', 0)
            open_price = info.get('regularMarketOpen', current_price)
            high_price = info.get('regularMarketDayHigh', current_price)
            low_price = info.get('regularMarketDayLow', current_price)
            volume = info.get('regularMarketVolume', 0)
            change = info.get('regularMarketChange', 0)
            change_percent = info.get('regularMarketChangePercent', 0)
            
            return MarketData(
                symbol=symbol,
                price=current_price,
                open=open_price,
                high=high_price,
                low=low_price,
                volume=volume,
                change=change,
                change_percent=change_percent,
                timestamp=datetime.now(),
                source='yahoo_finance'
            )
            
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            return None
    
    async def _fetch_finnhub(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Finnhub (free tier: 60 calls/minute)"""
        try:
            # Finnhub free API key (you can get one at finnhub.io)
            api_key = "demo"  # Replace with actual free API key
            
            async with aiohttp.ClientSession() as session:
                # Get current price
                quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
                
                async with session.get(quote_url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data or 'c' not in data:
                        return None
                    
                    current_price = data['c']  # Current price
                    open_price = data['o']     # Open price
                    high_price = data['h']     # High price
                    low_price = data['l']      # Low price
                    change = data['d']         # Change
                    change_percent = data['dp'] # Change percent
                    
                    return MarketData(
                        symbol=symbol,
                        price=current_price,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        volume=0,  # Volume not available in quote endpoint
                        change=change,
                        change_percent=change_percent,
                        timestamp=datetime.now(),
                        source='finnhub'
                    )
                    
        except Exception as e:
            logger.warning(f"Finnhub failed for {symbol}: {e}")
            return None
    
    async def _fetch_alpha_vantage(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Alpha Vantage (free tier: 25 calls/day)"""
        try:
            # Use demo API key or set your own
            api_key = "demo"  # Replace with actual free API key
            
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'Global Quote' not in data:
                        return None
                    
                    quote = data['Global Quote']
                    
                    return MarketData(
                        symbol=symbol,
                        price=float(quote['05. price']),
                        open=float(quote['02. open']),
                        high=float(quote['03. high']),
                        low=float(quote['04. low']),
                        volume=int(quote['06. volume']),
                        change=float(quote['09. change']),
                        change_percent=float(quote['10. change percent'].replace('%', '')),
                        timestamp=datetime.now(),
                        source='alpha_vantage'
                    )
                    
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
            return None
    
    async def _fetch_polygon(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Polygon.io (free tier: 5 calls/minute)"""
        try:
            # Polygon free API key
            api_key = "demo"  # Replace with actual free API key
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apikey={api_key}"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'results' not in data or not data['results']:
                        return None
                    
                    result = data['results'][0]
                    
                    return MarketData(
                        symbol=symbol,
                        price=result['c'],  # Close price
                        open=result['o'],   # Open price
                        high=result['h'],   # High price
                        low=result['l'],    # Low price
                        volume=result['v'], # Volume
                        change=0,           # Calculate if needed
                        change_percent=0,   # Calculate if needed
                        timestamp=datetime.now(),
                        source='polygon'
                    )
                    
        except Exception as e:
            logger.warning(f"Polygon failed for {symbol}: {e}")
            return None
    
    async def _fetch_fmp(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Financial Modeling Prep (free tier: 250 calls/day)"""
        try:
            # FMP free API key
            api_key = "demo"  # Replace with actual free API key
            
            async with aiohttp.ClientSession() as session:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data or len(data) == 0:
                        return None
                    
                    quote = data[0]
                    
                    return MarketData(
                        symbol=symbol,
                        price=quote['price'],
                        open=quote['open'],
                        high=quote['dayHigh'],
                        low=quote['dayLow'],
                        volume=quote['volume'],
                        change=quote['change'],
                        change_percent=quote['changesPercentage'],
                        timestamp=datetime.now(),
                        source='fmp'
                    )
                    
        except Exception as e:
            logger.warning(f"FMP failed for {symbol}: {e}")
            return None
    
    async def _fetch_iex_cloud(self, symbol: str) -> Optional[MarketData]:
        """Fetch from IEX Cloud (free tier available)"""
        try:
            # IEX Cloud free API
            async with aiohttp.ClientSession() as session:
                url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote?token=demo"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    return MarketData(
                        symbol=symbol,
                        price=data['latestPrice'],
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        volume=data['latestVolume'],
                        change=data['change'],
                        change_percent=data['changePercent'] * 100,
                        timestamp=datetime.now(),
                        source='iex_cloud'
                    )
                    
        except Exception as e:
            logger.warning(f"IEX Cloud failed for {symbol}: {e}")
            return None
    
    async def _fetch_twelve_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch from Twelve Data (free tier: 800 calls/day)"""
        try:
            # Twelve Data free API
            api_key = "demo"  # Replace with actual free API key
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={api_key}"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'code' in data and data['code'] != 200:
                        return None
                    
                    return MarketData(
                        symbol=symbol,
                        price=float(data['close']),
                        open=float(data['open']),
                        high=float(data['high']),
                        low=float(data['low']),
                        volume=int(data.get('volume', 0)),
                        change=float(data.get('change', 0)),
                        change_percent=float(data.get('percent_change', 0)),
                        timestamp=datetime.now(),
                        source='twelve_data'
                    )
                    
        except Exception as e:
            logger.warning(f"Twelve Data failed for {symbol}: {e}")
            return None

class CryptoDataFetcher:
    """Specialized fetcher for cryptocurrency data using free APIs"""
    
    def __init__(self):
        self.crypto_sources = [
            'coingecko',
            'coinapi',
            'cryptocompare',
            'binance'
        ]
        
    async def get_crypto_data(self, symbol: str) -> Optional[MarketData]:
        """Get cryptocurrency data from free APIs"""
        
        # Convert symbol format (e.g., BTC-USD -> btc)
        crypto_symbol = symbol.replace('-USD', '').lower()
        
        for source in self.crypto_sources:
            try:
                data = await self._fetch_crypto_from_source(source, crypto_symbol, symbol)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch crypto from {source} for {symbol}: {e}")
                continue
        
        return None
    
    async def _fetch_crypto_from_source(self, source: str, crypto_symbol: str, original_symbol: str) -> Optional[MarketData]:
        """Fetch crypto data from specific source"""
        
        if source == 'coingecko':
            return await self._fetch_coingecko(crypto_symbol, original_symbol)
        elif source == 'coinapi':
            return await self._fetch_coinapi(crypto_symbol, original_symbol)
        elif source == 'cryptocompare':
            return await self._fetch_cryptocompare(crypto_symbol, original_symbol)
        elif source == 'binance':
            return await self._fetch_binance(crypto_symbol, original_symbol)
        
        return None
    
    async def _fetch_coingecko(self, crypto_symbol: str, original_symbol: str) -> Optional[MarketData]:
        """Fetch from CoinGecko (free, no API key needed)"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_symbol}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if crypto_symbol not in data:
                        return None
                    
                    coin_data = data[crypto_symbol]
                    price = coin_data['usd']
                    change_percent = coin_data.get('usd_24h_change', 0)
                    volume = coin_data.get('usd_24h_vol', 0)
                    
                    return MarketData(
                        symbol=original_symbol,
                        price=price,
                        open=price,  # Approximate
                        high=price,  # Approximate
                        low=price,   # Approximate
                        volume=int(volume),
                        change=0,    # Calculate if needed
                        change_percent=change_percent,
                        timestamp=datetime.now(),
                        source='coingecko'
                    )
                    
        except Exception as e:
            logger.warning(f"CoinGecko failed for {crypto_symbol}: {e}")
            return None
    
    async def _fetch_binance(self, crypto_symbol: str, original_symbol: str) -> Optional[MarketData]:
        """Fetch from Binance public API (free, no API key needed)"""
        try:
            # Convert to Binance format (e.g., btc -> BTCUSDT)
            binance_symbol = f"{crypto_symbol.upper()}USDT"
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    return MarketData(
                        symbol=original_symbol,
                        price=float(data['lastPrice']),
                        open=float(data['openPrice']),
                        high=float(data['highPrice']),
                        low=float(data['lowPrice']),
                        volume=int(float(data['volume'])),
                        change=float(data['priceChange']),
                        change_percent=float(data['priceChangePercent']),
                        timestamp=datetime.now(),
                        source='binance'
                    )
                    
        except Exception as e:
            logger.warning(f"Binance failed for {crypto_symbol}: {e}")
            return None

# Global instances
data_aggregator = FreeDataAggregator()
crypto_fetcher = CryptoDataFetcher()

async def get_real_market_data(symbols: List[str]) -> Dict[str, MarketData]:
    """Main function to get real market data for multiple symbols"""
    
    # Separate crypto and regular symbols
    crypto_symbols = [s for s in symbols if any(crypto in s.upper() for crypto in ['BTC', 'ETH', 'TRX', 'LINK', 'CAKE', 'USDT', 'USDC'])]
    regular_symbols = [s for s in symbols if s not in crypto_symbols]
    
    results = {}
    
    # Fetch regular market data
    if regular_symbols:
        regular_data = await data_aggregator.get_multiple_symbols(regular_symbols)
        results.update(regular_data)
    
    # Fetch crypto data
    for crypto_symbol in crypto_symbols:
        crypto_data = await crypto_fetcher.get_crypto_data(crypto_symbol)
        if crypto_data:
            results[crypto_symbol] = crypto_data
    
    return results

# Synchronous wrapper for backward compatibility
def get_real_market_data_sync(symbols: List[str]) -> Dict[str, MarketData]:
    """Synchronous wrapper for getting real market data"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_real_market_data(symbols)) 