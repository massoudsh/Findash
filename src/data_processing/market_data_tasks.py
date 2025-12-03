"""
Celery Tasks for Market Data Fetching
Scheduled tasks to fetch and store real market data from free APIs
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from celery import Task

from src.core.celery_app import celery_app
from src.database.postgres_connection import SessionLocal
from src.services.market_data_service import market_data_service
from src.data_processing.enhanced_free_sources import get_enhanced_market_data

logger = logging.getLogger(__name__)


@celery_app.task(name='market_data.fetch_single', bind=True)
def fetch_single_market_data(self: Task, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch market data for a single symbol
    
    Args:
        symbol: Stock/crypto symbol
        force_refresh: Force refresh even if recent data exists
        
    Returns:
        Dict with status and data
    """
    db = SessionLocal()
    try:
        logger.info(f"📊 Fetching market data for {symbol}")
        
        # Use asyncio to run async function
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            market_data = loop.run_until_complete(
                market_data_service.fetch_and_store(symbol, db, force_refresh)
            )
            
            if market_data:
                return {
                    "status": "success",
                    "symbol": symbol,
                    "data": {
                        "price": float(market_data.price),
                        "open": float(market_data.open),
                        "high": float(market_data.high),
                        "low": float(market_data.low),
                        "volume": market_data.volume,
                        "timestamp": market_data.time.isoformat(),
                        "source": market_data.exchange
                    }
                }
            else:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "message": "No data available"
                }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in fetch_single_market_data for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(name='market_data.fetch_multiple', bind=True)
def fetch_multiple_market_data(
    self: Task,
    symbols: List[str],
    max_concurrent: int = 5
) -> Dict[str, Any]:
    """
    Fetch market data for multiple symbols
    
    Args:
        symbols: List of symbols to fetch
        max_concurrent: Maximum concurrent API calls
        
    Returns:
        Dict with status and results
    """
    db = SessionLocal()
    try:
        logger.info(f"📊 Fetching market data for {len(symbols)} symbols")
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                market_data_service.fetch_multiple_and_store(
                    symbols, db, max_concurrent
                )
            )
            
            return {
                "status": "success",
                "symbols_requested": len(symbols),
                "symbols_fetched": len(results),
                "results": {
                    symbol: {
                        "price": float(data.price),
                        "open": float(data.open),
                        "high": float(data.high),
                        "low": float(data.low),
                        "volume": data.volume,
                        "timestamp": data.time.isoformat(),
                        "source": data.exchange
                    }
                    for symbol, data in results.items()
                }
            }
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in fetch_multiple_market_data: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(name='market_data.fetch_watchlist', bind=True)
def fetch_watchlist_market_data(self: Task, watchlist_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fetch market data for default watchlist symbols
    
    Args:
        watchlist_symbols: Optional custom watchlist, defaults to popular symbols
        
    Returns:
        Dict with status and results
    """
    # Default watchlist if not provided
    if not watchlist_symbols:
        watchlist_symbols = [
            # Tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            # Crypto
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD',
            # ETFs
            'SPY', 'QQQ', 'DIA',
            # Commodities
            'GLD', 'SLV'
        ]
    
    return fetch_multiple_market_data.delay(watchlist_symbols).get(timeout=300)


@celery_app.task(name='market_data.update_portfolio_symbols', bind=True)
def update_portfolio_symbols(self: Task) -> Dict[str, Any]:
    """
    Fetch market data for all symbols in active portfolios
    """
    db = SessionLocal()
    try:
        from src.database.models import Position
        
        # Get all unique symbols from active positions
        active_symbols = db.query(Position.symbol).filter(
            Position.is_active == True
        ).distinct().all()
        
        symbols = [symbol[0] for symbol in active_symbols]
        
        if not symbols:
            logger.info("No active positions to update")
            return {
                "status": "success",
                "message": "No active positions",
                "symbols_fetched": 0
            }
        
        logger.info(f"Updating {len(symbols)} portfolio symbols")
        return fetch_multiple_market_data.delay(symbols).get(timeout=300)
        
    except Exception as e:
        logger.error(f"Error in update_portfolio_symbols: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()


@celery_app.task(name='market_data.cleanup_old_data', bind=True)
def cleanup_old_market_data(self: Task, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old market data beyond retention period
    
    Args:
        days_to_keep: Number of days of data to keep
    """
    db = SessionLocal()
    try:
        from src.database.models import MarketData
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        deleted = db.query(MarketData).filter(
            MarketData.time < cutoff_date
        ).delete()
        
        db.commit()
        
        logger.info(f"Cleaned up {deleted} old market data records")
        
        return {
            "status": "success",
            "deleted_records": deleted,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_market_data: {e}")
        db.rollback()
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()

