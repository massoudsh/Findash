"""
Market Data API Routes for FastAPI service
Handles high-frequency market data, real-time quotes, and historical data
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from src.core.cache import TradingCache
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher

router = APIRouter()

# Pydantic models
class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    high: float
    low: float
    open: float

class HistoricalDataResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    data: List[Dict[str, Any]]

class BatchMarketDataRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to fetch")
    fields: Optional[List[str]] = Field(None, description="Specific fields to return")

# Dependencies
async def get_cache() -> TradingCache:
    """Get trading cache instance"""
    return TradingCache()

async def get_data_fetcher() -> TimeSeriesDataFetcher:
    """Get time series data fetcher"""
    return TimeSeriesDataFetcher()

@router.get("/current/{symbol}", response_model=MarketDataResponse)
async def get_current_market_data(
    symbol: str,
    cache: TradingCache = Depends(get_cache)
):
    """Get current market data for a symbol"""
    try:
        # Check cache first
        cache_key = f"market_data:{symbol}"
        cached_data = await cache.get(cache_key)
        
        if cached_data:
            return MarketDataResponse(**cached_data)
        
        # Fetch from Yahoo Finance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        history = ticker.history(period="1d", interval="1m")
        
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        latest = history.iloc[-1]
        previous_close = info.get('previousClose', latest['Close'])
        
        market_data = {
            "symbol": symbol,
            "price": round(latest['Close'], 2),
            "change": round(latest['Close'] - previous_close, 2),
            "change_percent": round(((latest['Close'] - previous_close) / previous_close) * 100, 2),
            "volume": int(latest['Volume']),
            "high": round(latest['High'], 2),
            "low": round(latest['Low'], 2),
            "open": round(latest['Open'], 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for 10 seconds
        await cache.set(cache_key, market_data, ttl=10)
        
        return MarketDataResponse(**market_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@router.post("/batch", response_model=List[MarketDataResponse])
async def get_batch_market_data(
    request: BatchMarketDataRequest,
    cache: TradingCache = Depends(get_cache)
):
    """Get market data for multiple symbols"""
    try:
        results = []
        
        for symbol in request.symbols:
            try:
                cache_key = f"market_data:{symbol}"
                cached_data = await cache.get(cache_key)
                
                if cached_data:
                    results.append(MarketDataResponse(**cached_data))
                    continue
                
                # Fetch individual symbol data
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period="1d", interval="1m")
                
                if not history.empty:
                    latest = history.iloc[-1]
                    previous_close = info.get('previousClose', latest['Close'])
                    
                    market_data = {
                        "symbol": symbol,
                        "price": round(latest['Close'], 2),
                        "change": round(latest['Close'] - previous_close, 2),
                        "change_percent": round(((latest['Close'] - previous_close) / previous_close) * 100, 2),
                        "volume": int(latest['Volume']),
                        "high": round(latest['High'], 2),
                        "low": round(latest['Low'], 2),
                        "open": round(latest['Open'], 2),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Cache for 10 seconds
                    await cache.set(cache_key, market_data, ttl=10)
                    results.append(MarketDataResponse(**market_data))
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching batch market data: {str(e)}")

@router.get("/historical/{symbol}", response_model=HistoricalDataResponse)
async def get_historical_data(
    symbol: str,
    period: str = Query("1mo", description="Period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="Interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo"),
    data_fetcher: TimeSeriesDataFetcher = Depends(get_data_fetcher),
    cache: TradingCache = Depends(get_cache)
):
    """Get historical market data"""
    try:
        cache_key = f"historical:{symbol}:{period}:{interval}"
        cached_data = await cache.get(cache_key)
        
        if cached_data:
            return HistoricalDataResponse(**cached_data)
        
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Convert to list of dictionaries
        data_list = []
        for index, row in history.iterrows():
            data_list.append({
                "timestamp": index.isoformat(),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        response_data = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data_list
        }
        
        # Cache for 5 minutes
        await cache.set(cache_key, response_data, ttl=300)
        
        return HistoricalDataResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")

@router.get("/quote/{symbol}")
async def get_real_time_quote(
    symbol: str,
    cache: TradingCache = Depends(get_cache)
):
    """Get real-time quote for a symbol"""
    try:
        cache_key = f"quote:{symbol}"
        cached_quote = await cache.get(cache_key)
        
        if cached_quote:
            return cached_quote
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        quote = {
            "symbol": symbol,
            "price": info.get('currentPrice', 0),
            "bid": info.get('bid', 0),
            "ask": info.get('ask', 0),
            "bid_size": info.get('bidSize', 0),
            "ask_size": info.get('askSize', 0),
            "last_trade_time": datetime.now().isoformat(),
            "day_high": info.get('dayHigh', 0),
            "day_low": info.get('dayLow', 0),
            "previous_close": info.get('previousClose', 0),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "dividend_yield": info.get('dividendYield', 0)
        }
        
        # Cache for 5 seconds for real-time quotes
        await cache.set(cache_key, quote, ttl=5)
        
        return quote
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote: {str(e)}")

@router.get("/metrics")
async def get_market_metrics():
    """Get overall market metrics and performance"""
    try:
        # Fetch major indices
        indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow, NASDAQ, Russell 2000
        metrics = {}
        
        for index in indices:
            ticker = yf.Ticker(index)
            history = ticker.history(period="1d")
            
            if not history.empty:
                latest = history.iloc[-1]
                previous = history.iloc[-2] if len(history) > 1 else latest
                
                metrics[index] = {
                    "price": round(latest['Close'], 2),
                    "change": round(latest['Close'] - previous['Close'], 2),
                    "change_percent": round(((latest['Close'] - previous['Close']) / previous['Close']) * 100, 2),
                    "volume": int(latest['Volume'])
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "indices": metrics,
            "status": "active" if datetime.now().weekday() < 5 else "closed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market metrics: {str(e)}") 