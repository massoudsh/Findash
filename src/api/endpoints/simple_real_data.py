"""
Simple Real Market Data API Endpoints
Provides real market data using basic requests without complex async imports
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import requests
import logging
from datetime import datetime
import time
import yfinance as yf

logger = logging.getLogger(__name__)
simple_data_router = APIRouter()

# Cache for data to avoid hitting rate limits
data_cache = {}
cache_ttl = 60  # 1 minute cache

def get_cached_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Get cached data if still valid"""
    if symbol in data_cache:
        data, timestamp = data_cache[symbol]
        if time.time() - timestamp < cache_ttl:
            return data
    return None

def cache_data(symbol: str, data: Dict[str, Any]):
    """Cache market data"""
    data_cache[symbol] = (data, time.time())

def fetch_yahoo_data_simple(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch data from Yahoo Finance with error handling"""
    try:
        # Check cache first
        cached = get_cached_data(symbol)
        if cached:
            return cached
        
        # Add delay to avoid rate limiting
        time.sleep(0.1)
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            # Fallback to history data
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            current_price = float(latest['Close'])
            open_price = float(latest['Open'])
            high_price = float(latest['High'])
            low_price = float(latest['Low'])
            volume = int(latest['Volume'])
            change = current_price - open_price
            change_percent = (change / open_price) * 100 if open_price > 0 else 0
        else:
            current_price = info.get('regularMarketPrice', 0)
            open_price = info.get('regularMarketOpen', current_price)
            high_price = info.get('regularMarketDayHigh', current_price)
            low_price = info.get('regularMarketDayLow', current_price)
            volume = info.get('regularMarketVolume', 0)
            change = info.get('regularMarketChange', 0)
            change_percent = info.get('regularMarketChangePercent', 0)
        
        data = {
            "symbol": symbol,
            "price": current_price,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "change": change,
            "change_percent": change_percent,
            "timestamp": datetime.now().isoformat(),
            "source": "yahoo_finance"
        }
        
        # Cache the data
        cache_data(symbol, data)
        return data
        
    except Exception as e:
        logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
        return None

def fetch_binance_crypto_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch crypto data from Binance public API"""
    try:
        # Check cache first
        cached = get_cached_data(f"binance_{symbol}")
        if cached:
            return cached
        
        # Convert symbol format (e.g., BTC-USD -> BTCUSDT)
        if symbol.endswith('-USD'):
            binance_symbol = symbol.replace('-USD', 'USDT')
        else:
            binance_symbol = f"{symbol}USDT"
        
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol}"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        result = {
            "symbol": symbol,
            "price": float(data['lastPrice']),
            "open": float(data['openPrice']),
            "high": float(data['highPrice']),
            "low": float(data['lowPrice']),
            "volume": int(float(data['volume'])),
            "change": float(data['priceChange']),
            "change_percent": float(data['priceChangePercent']),
            "timestamp": datetime.now().isoformat(),
            "source": "binance"
        }
        
        # Cache the data
        cache_data(f"binance_{symbol}", result)
        return result
        
    except Exception as e:
        logger.warning(f"Binance failed for {symbol}: {e}")
        return None

def fetch_coingecko_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch crypto data from CoinGecko"""
    try:
        # Check cache first
        cached = get_cached_data(f"coingecko_{symbol}")
        if cached:
            return cached
        
        # Convert symbol format
        crypto_symbol = symbol.replace('-USD', '').lower()
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_symbol}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if crypto_symbol not in data:
            return None
        
        coin_data = data[crypto_symbol]
        price = coin_data['usd']
        change_percent = coin_data.get('usd_24h_change', 0)
        volume = coin_data.get('usd_24h_vol', 0)
        
        result = {
            "symbol": symbol,
            "price": price,
            "open": price,  # Approximate
            "high": price,  # Approximate
            "low": price,   # Approximate
            "volume": int(volume),
            "change": 0,    # Calculate if needed
            "change_percent": change_percent,
            "timestamp": datetime.now().isoformat(),
            "source": "coingecko"
        }
        
        # Cache the data
        cache_data(f"coingecko_{symbol}", result)
        return result
        
    except Exception as e:
        logger.warning(f"CoinGecko failed for {symbol}: {e}")
        return None

@simple_data_router.get("/simple-market-data/real-time")
def get_simple_real_time_data(
    symbols: str = Query(..., description="Comma-separated list of symbols (e.g., AAPL,MSFT,BTC-USD)")
) -> Dict[str, Any]:
    """
    Get real-time market data for multiple symbols using simple requests
    """
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 10:  # Limit to prevent abuse
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        response_data = {}
        
        for symbol in symbol_list:
            data = None
            
            # Try different sources based on symbol type
            if any(crypto in symbol for crypto in ['BTC', 'ETH', 'TRX', 'LINK', 'CAKE']):
                # Try crypto sources first
                data = fetch_binance_crypto_data(symbol)
                if not data:
                    data = fetch_coingecko_data(symbol)
            
            # Fallback to Yahoo Finance for all symbols
            if not data:
                data = fetch_yahoo_data_simple(symbol)
            
            if data:
                response_data[symbol] = data
        
        return {
            "status": "success",
            "data": response_data,
            "timestamp": datetime.now().isoformat(),
            "symbols_requested": len(symbol_list),
            "symbols_returned": len(response_data),
            "cache_info": f"Cached for {cache_ttl} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@simple_data_router.get("/simple-market-data/portfolio")
def get_simple_portfolio_data() -> Dict[str, Any]:
    """
    Get real-time data for popular portfolio symbols
    """
    # Popular symbols
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA', 'BTC-USD', 'ETH-USD']
    
    try:
        data = {}
        
        for symbol in symbols:
            market_data = None
            
            # Try crypto sources for crypto symbols
            if symbol.endswith('-USD'):
                market_data = fetch_binance_crypto_data(symbol)
                if not market_data:
                    market_data = fetch_coingecko_data(symbol)
            
            # Fallback to Yahoo Finance
            if not market_data:
                market_data = fetch_yahoo_data_simple(symbol)
            
            if market_data:
                data[symbol] = market_data
        
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio data: {str(e)}")

@simple_data_router.get("/simple-market-data/sources")
def get_simple_data_sources() -> Dict[str, Any]:
    """
    Get information about available free data sources
    """
    return {
        "status": "success",
        "sources": {
            "yahoo_finance": {
                "description": "Free stock data with built-in rate limiting",
                "coverage": "Global stocks, ETFs, indices, some crypto",
                "rate_limit": "Automatic delays to prevent 429 errors",
                "reliability": "High for stocks, moderate for crypto"
            },
            "binance": {
                "description": "Free crypto data from Binance public API",
                "coverage": "All Binance trading pairs",
                "rate_limit": "1200 requests/minute",
                "reliability": "Very high for major cryptocurrencies"
            },
            "coingecko": {
                "description": "Free crypto data, no API key needed",
                "coverage": "10,000+ cryptocurrencies",
                "rate_limit": "100 calls/minute",
                "reliability": "High for crypto market data"
            }
        },
        "strategy": "System tries crypto APIs first for crypto symbols, then falls back to Yahoo Finance",
        "caching": f"Data cached for {cache_ttl} seconds to improve performance and reduce API calls",
        "supported_symbols": {
            "stocks": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"],
            "crypto": ["BTC-USD", "ETH-USD", "TRX-USD", "LINK-USD", "CAKE-USD"],
            "etfs": ["SPY", "QQQ", "GLD", "SLV"]
        }
    } 