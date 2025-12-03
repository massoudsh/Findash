"""
Unified Market Data Service
Manages real-time market data fetching from multiple free APIs with intelligent fallbacks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.database.postgres_connection import get_db
from src.database.models import MarketData
from src.data_processing.enhanced_free_sources import (
    enhanced_data_aggregator,
    get_enhanced_market_data,
    get_single_market_data,
    MarketDataPoint
)

logger = logging.getLogger(__name__)


class MarketDataService:
    """Unified service for managing market data from free APIs"""
    
    def __init__(self):
        self.aggregator = enhanced_data_aggregator
        self.cache_ttl = 60  # 1 minute cache
    
    async def fetch_and_store(
        self,
        symbol: str,
        db: Session,
        force_refresh: bool = False
    ) -> Optional[MarketData]:
        """
        Fetch market data from free APIs and store in database
        
        Args:
            symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
            db: Database session
            force_refresh: Force fetch even if recent data exists
            
        Returns:
            MarketData object if successful, None otherwise
        """
        try:
            # Check if we have recent data (within cache TTL)
            if not force_refresh:
                recent_data = self._get_recent_data(symbol, db)
                if recent_data:
                    logger.debug(f"Using cached data for {symbol}")
                    return recent_data
            
            # Fetch from free APIs
            logger.info(f"Fetching fresh data for {symbol}")
            data_point = await get_single_market_data(symbol)
            
            if not data_point or data_point.price <= 0:
                logger.warning(f"No valid data returned for {symbol}")
                return None
            
            # Store in database
            market_data = self._store_market_data(symbol, data_point, db)
            logger.info(f"✅ Stored market data for {symbol} from {data_point.source}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching/storing data for {symbol}: {e}")
            return None
    
    async def fetch_multiple_and_store(
        self,
        symbols: List[str],
        db: Session,
        max_concurrent: int = 5
    ) -> Dict[str, MarketData]:
        """
        Fetch market data for multiple symbols concurrently
        
        Args:
            symbols: List of symbols to fetch
            db: Database session
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Dictionary mapping symbol to MarketData
        """
        results = {}
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(symbol):
            async with semaphore:
                return await self.fetch_and_store(symbol, db)
        
        # Create tasks
        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        fetched_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for symbol, result in zip(symbols, fetched_data):
            if isinstance(result, MarketData) and result:
                results[symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
        
        logger.info(f"Successfully fetched {len(results)}/{len(symbols)} symbols")
        return results
    
    def _get_recent_data(
        self,
        symbol: str,
        db: Session,
        max_age_seconds: int = 60
    ) -> Optional[MarketData]:
        """Get recent market data from database if available"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            
            recent = db.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.time >= cutoff_time
                )
            ).order_by(MarketData.time.desc()).first()
            
            return recent
        except Exception as e:
            logger.debug(f"Error checking recent data: {e}")
            return None
    
    def _store_market_data(
        self,
        symbol: str,
        data_point: MarketDataPoint,
        db: Session
    ) -> MarketData:
        """Store market data point in database"""
        try:
            # Check if record exists for this timestamp
            existing = db.query(MarketData).filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.time == data_point.timestamp
                )
            ).first()
            
            if existing:
                # Update existing record
                existing.price = data_point.price
                existing.open = data_point.open
                existing.high = data_point.high
                existing.low = data_point.low
                existing.volume = data_point.volume
                existing.exchange = data_point.source
                db.commit()
                return existing
            
            # Create new record
            market_data = MarketData(
                time=data_point.timestamp,
                symbol=symbol,
                price=data_point.price,
                open=data_point.open,
                high=data_point.high,
                low=data_point.low,
                volume=data_point.volume,
                exchange=data_point.source
            )
            
            db.add(market_data)
            db.commit()
            db.refresh(market_data)
            
            return market_data
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing market data: {e}")
            raise
    
    def get_historical_data(
        self,
        symbol: str,
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketData]:
        """Get historical market data from database"""
        try:
            query = db.query(MarketData).filter(MarketData.symbol == symbol)
            
            if start_date:
                query = query.filter(MarketData.time >= start_date)
            if end_date:
                query = query.filter(MarketData.time <= end_date)
            
            return query.order_by(MarketData.time.desc()).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def get_latest_data(
        self,
        symbol: str,
        db: Session
    ) -> Optional[MarketData]:
        """Get latest market data for a symbol"""
        try:
            return db.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(MarketData.time.desc()).first()
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return None
    
    def get_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        return self.aggregator.get_source_status()


# Global service instance
market_data_service = MarketDataService()

