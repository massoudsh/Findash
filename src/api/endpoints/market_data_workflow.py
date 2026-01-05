"""
DEPRECATED: This file has been integrated into unified_market_data.py

This module is kept for backward compatibility only.
All functionality is now available in:
- src.api.endpoints.unified_market_data

Please update imports to use:
    from src.api.endpoints.unified_market_data import router as market_data_router
    
This file will be removed in a future version.

Market Data Workflow API Endpoints
Provides endpoints for managing real-time market data fetching from free APIs
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.postgres_connection import get_db
from src.core.security import get_current_active_user, TokenData
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace
from src.services.market_data_service import market_data_service
from src.data_processing.market_data_tasks import (
    fetch_single_market_data,
    fetch_multiple_market_data,
    fetch_watchlist_market_data,
    update_portfolio_symbols
)
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market-data/workflow", tags=["Market Data Workflow"])

# Cache manager
cache_manager = CacheManager()


class FetchMarketDataRequest(BaseModel):
    """Request to fetch market data"""
    symbols: List[str] = Field(..., description="List of symbols to fetch")
    force_refresh: bool = Field(False, description="Force refresh even if recent data exists")


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
    Fetch real-time market data for a single symbol from free APIs
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


@router.get("/latest/{symbol}")
async def get_latest_market_data(
    symbol: str,
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get latest market data for a symbol from database
    """
    try:
        market_data = market_data_service.get_latest_data(symbol, db)
        
        if not market_data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {symbol}"
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
        logger.error(f"Error getting latest data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical/{symbol}")
async def get_historical_market_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get historical market data for a symbol
    """
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        historical = market_data_service.get_historical_data(
            symbol, db, start, end, limit
        )
        
        return {
            "status": "success",
            "symbol": symbol,
            "count": len(historical),
            "data": [
                {
                    "timestamp": data.time.isoformat(),
                    "price": float(data.price),
                    "open": float(data.open),
                    "high": float(data.high),
                    "low": float(data.low),
                    "volume": data.volume,
                    "source": data.exchange or "unknown"
                }
                for data in historical
            ]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/watchlist/refresh")
async def refresh_watchlist(
    symbols: Optional[List[str]] = Body(None, description="Optional custom watchlist"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit)
):
    """
    Trigger refresh of watchlist market data
    """
    try:
        task = fetch_watchlist_market_data.delay(symbols)
        
        return {
            "status": "accepted",
            "task_id": task.id,
            "message": "Watchlist refresh queued"
        }
        
    except Exception as e:
        logger.error(f"Error queuing watchlist refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/refresh")
async def refresh_portfolio_symbols(
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit)
):
    """
    Trigger refresh of all portfolio symbols
    """
    try:
        task = update_portfolio_symbols.delay()
        
        return {
            "status": "accepted",
            "task_id": task.id,
            "message": "Portfolio symbols refresh queued"
        }
        
    except Exception as e:
        logger.error(f"Error queuing portfolio refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

