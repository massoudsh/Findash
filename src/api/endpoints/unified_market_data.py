"""
Unified Market Data API Endpoints
Consolidates functionality from:
- professional_market_data.py: Professional-grade quotes, historical, technical indicators
- market_data_workflow.py: Workflow management, database integration, async tasks
- real_market_data.py: Simple real-time data fetching
- simple_real_data.py: Simplified endpoints with caching

This unified service provides:
- Real-time quotes and market data
- Historical price data (from APIs and database)
- Technical analysis indicators
- Batch operations and async task management
- Database persistence and caching
- Multiple data source fallbacks
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import yfinance as yf
import pandas as pd
import numpy as np
import time

from src.database.postgres_connection import get_db
from src.core.security import get_current_active_user, TokenData, rate_limit_dependency
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace
from src.core.config import get_settings
from src.services.market_data_service import market_data_service
from src.data_processing.market_data_tasks import (
    fetch_single_market_data,
    fetch_multiple_market_data,
    fetch_watchlist_market_data,
    update_portfolio_symbols
)
from src.data_processing.free_data_sources import get_real_market_data, MarketData

logger = logging.getLogger(__name__)
settings = get_settings()

# Unified router with single prefix
router = APIRouter(prefix="/api/market-data", tags=["Unified Market Data"])

# Cache manager
cache_manager = CacheManager()

# Simple in-memory cache for simple endpoints
data_cache = {}
cache_ttl = 60  # 1 minute cache

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class FetchMarketDataRequest(BaseModel):
    """Request to fetch market data"""
    symbols: List[str] = Field(..., description="List of symbols to fetch")
    force_refresh: bool = Field(False, description="Force refresh even if recent data exists")

class MarketQuote(BaseModel):
    """Real-time market quote"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    timestamp: str

class MarketDataResponse(BaseModel):
    """Market data response"""
    symbol: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    timestamp: str
    source: str

class HistoricalPrice(BaseModel):
    """Historical price data point"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    symbol: str
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    timestamp: str

# ============================================
# HELPER FUNCTIONS
# ============================================

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

def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> TechnicalIndicators:
    """Calculate technical indicators from price data"""
    try:
        # Simple Moving Averages
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        
        # Exponential Moving Averages
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 14 else None
        
        # MACD
        macd_line = ema_12 - ema_26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().iloc[-1]
        
        # Bollinger Bands
        sma_20_series = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        bollinger_upper = (sma_20_series + (std_20 * 2)).iloc[-1] if len(df) >= 20 else None
        bollinger_lower = (sma_20_series - (std_20 * 2)).iloc[-1] if len(df) >= 20 else None
        
        return TechnicalIndicators(
            symbol=symbol,
            sma_20=float(sma_20) if sma_20 and not pd.isna(sma_20) else None,
            sma_50=float(sma_50) if sma_50 and not pd.isna(sma_50) else None,
            ema_12=float(ema_12) if not pd.isna(ema_12) else None,
            ema_26=float(ema_26) if not pd.isna(ema_26) else None,
            rsi=float(rsi) if rsi and not pd.isna(rsi) else None,
            macd=float(macd_line) if not pd.isna(macd_line) else None,
            macd_signal=float(macd_signal) if not pd.isna(macd_signal) else None,
            bollinger_upper=float(bollinger_upper) if bollinger_upper and not pd.isna(bollinger_upper) else None,
            bollinger_lower=float(bollinger_lower) if bollinger_lower and not pd.isna(bollinger_lower) else None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        return TechnicalIndicators(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

# ============================================
# UNIFIED ENDPOINTS
# ============================================

@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_real_time_quote(
    symbol: str,
    current_user: TokenData = Depends(get_current_active_user),
    _: bool = Depends(rate_limit_dependency)
):
    """
    Get real-time market quote for a symbol
    Uses Yahoo Finance API for professional-grade market data
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch real-time data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price data
        hist = ticker.history(period="2d", interval="1m")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(info.get('previousClose', hist['Close'].iloc[-2]))
        
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        quote = MarketQuote(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=int(hist['Volume'].iloc[-1]),
            bid=info.get('bid'),
            ask=info.get('ask'),
            bid_size=info.get('bidSize'),
            ask_size=info.get('askSize'),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Real-time quote fetched for {symbol}: ${current_price}")
        return quote
        
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote for {symbol}")

@router.get("/fetch/{symbol}", response_model=MarketDataResponse)
@cache_manager.cache_response(namespace=CacheNamespace.MARKET_DATA, key_prefix="symbol", ttl=60)
async def fetch_market_data(
    symbol: str,
    force_refresh: bool = Query(False, description="Force refresh"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Fetch and store market data for a single symbol
    Integrates with database and caching
    """
    try:
        logger.info(f"Fetching market data for {symbol}")
        
        # Fetch and store data
        market_data = await market_data_service.fetch_and_store(
            symbol, db, force_refresh
        )
        
        if not market_data:
            raise HTTPException(
                status_code=404,
                detail=f"No market data available for {symbol}"
            )
        
        return MarketDataResponse(
            symbol=market_data.symbol,
            price=float(market_data.price),
            open=float(market_data.open),
            high=float(market_data.high),
            low=float(market_data.low),
            volume=market_data.volume,
            timestamp=market_data.time.isoformat(),
            source=market_data.exchange or "unknown"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch/batch")
async def fetch_batch_market_data(
    request: FetchMarketDataRequest,
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Fetch market data for multiple symbols
    """
    try:
        if len(request.symbols) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 symbols allowed per request"
            )
        
        logger.info(f"Fetching market data for {len(request.symbols)} symbols")
        
        # Fetch and store data
        results = await market_data_service.fetch_multiple_and_store(
            request.symbols, db, max_concurrent=5
        )
        
        response_data = {}
        for symbol, data in results.items():
            response_data[symbol] = {
                "symbol": data.symbol,
                "price": float(data.price),
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "volume": data.volume,
                "timestamp": data.time.isoformat(),
                "source": data.exchange or "unknown"
            }
        
        return {
            "status": "success",
            "symbols_requested": len(request.symbols),
            "symbols_fetched": len(results),
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching batch market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{symbol}", response_model=List[HistoricalPrice])
async def get_historical_data(
    symbol: str,
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval"),
    source: str = Query("api", description="Data source: 'api' or 'database'"),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Get historical price data for a symbol
    Supports both API fetching and database retrieval
    """
    try:
        symbol = symbol.upper().strip()
        
        if source == "database":
            # Get from database via market_data_service
            # This would require db session, simplified for now
            raise HTTPException(status_code=501, detail="Database source not yet implemented in unified endpoint")
        
        # Fetch from API
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Convert to our format
        historical_data = []
        for date, row in hist.iterrows():
            historical_data.append(HistoricalPrice(
                date=date.strftime("%Y-%m-%d"),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adjusted_close=float(row['Close'])
            ))
        
        logger.info(f"Historical data fetched for {symbol}: {len(historical_data)} points")
        return historical_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {symbol}")

@router.get("/technical/{symbol}", response_model=TechnicalIndicators)
async def get_technical_analysis(
    symbol: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Get technical analysis indicators for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch price data for calculations
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo", interval="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(hist, symbol)
        
        logger.info(f"Technical indicators calculated for {symbol}")
        return indicators
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate technical analysis for {symbol}")

@router.post("/fetch/async")
async def fetch_market_data_async(
    request: FetchMarketDataRequest,
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit)
):
    """
    Trigger async Celery task to fetch market data
    Returns task ID for tracking
    """
    try:
        if len(request.symbols) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 symbols allowed per request"
            )
        
        # Trigger Celery task
        if len(request.symbols) == 1:
            task = fetch_single_market_data.delay(
                request.symbols[0],
                request.force_refresh
            )
        else:
            task = fetch_multiple_market_data.delay(
                request.symbols,
                max_concurrent=5
            )
        
        return {
            "status": "accepted",
            "task_id": task.id,
            "symbols": request.symbols,
            "message": "Market data fetch task queued"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing market data fetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/real-time")
async def get_real_time_data(
    symbols: str = Query(..., description="Comma-separated list of symbols (e.g., AAPL,MSFT,BTC-USD)"),
    current_user: TokenData = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get real-time market data for multiple symbols using free APIs
    Unified endpoint combining real_market_data and simple_real_data functionality
    """
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        # Fetch real data using free_data_sources
        market_data = await get_real_market_data(symbol_list)
        
        # Convert to response format
        response_data = {}
        for symbol, data in market_data.items():
            response_data[symbol] = {
                "symbol": data.symbol,
                "price": data.price,
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "volume": data.volume,
                "change": data.change,
                "change_percent": data.change_percent,
                "timestamp": data.timestamp.isoformat(),
                "source": data.source
            }
        
        return {
            "status": "success",
            "data": response_data,
            "timestamp": datetime.now().isoformat(),
            "symbols_requested": len(symbol_list),
            "symbols_returned": len(response_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@router.get("/sources/status")
async def get_data_sources_status(
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit)
):
    """
    Get status of all free data sources
    """
    try:
        status = market_data_service.get_source_status()
        
        return {
            "status": "success",
            "sources": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting source status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

