"""
System Initialization for Hybrid Architecture
Handles startup and initialization of FastAPI performance components
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Use lazy imports to avoid circular dependencies
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class SystemInitializer:
    """System initializer for hybrid architecture FastAPI service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache: Optional[Any] = None  # TradingCache type - lazy loaded
        self.intelligence_orchestrator: Optional[Any] = None  # IntelligenceOrchestrator type - lazy loaded
        self.websocket_manager: Optional[Any] = None  # WebSocketManager type - lazy loaded
        self.is_initialized = False
        
    async def initialize_performance_components(self):
        """Initialize performance-critical components for FastAPI service"""
        try:
            logger.info("🚀 Initializing FastAPI performance components...")
            
            # Lazy import to avoid circular dependencies
            from src.database.postgres_connection import init_db_connection
            from src.core.cache import TradingCache, initialize_cache
            from src.realtime.websockets import WebSocketManager
            from src.core.intelligence_orchestrator import IntelligenceOrchestrator
            
            # Initialize database connection
            logger.info("📊 Initializing database connection...")
            init_db_connection()
            
            # Initialize cache
            logger.info("🏎️ Initializing Redis cache...")
            await initialize_cache()
            self.cache = TradingCache()
            
            # Initialize WebSocket manager
            logger.info("🔌 Initializing WebSocket manager...")
            self.websocket_manager = WebSocketManager()
            if hasattr(self.websocket_manager, 'initialize'):
                await self.websocket_manager.initialize()
            
            # Initialize Intelligence Orchestrator for ML/AI operations
            logger.info("🧠 Initializing Intelligence Orchestrator...")
            self.intelligence_orchestrator = IntelligenceOrchestrator(self.cache)
            await self.intelligence_orchestrator.initialize_agents()
            
            self.is_initialized = True
            logger.info("✅ FastAPI performance components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FastAPI components: {e}")
            raise
    
    async def start_market_data_streams(self):
        """Start real-time market data streams"""
        try:
            logger.info("📈 Starting market data streams...")
            
            # Start background tasks for real-time data processing
            await self._start_price_feed_streams()
            await self._start_news_sentiment_streams()
            
            logger.info("✅ Market data streams started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start market data streams: {e}")
    
    async def _start_price_feed_streams(self):
        """Start real-time price feed streams"""
        # Implementation for real-time price feeds
        # This would connect to data providers like Alpha Vantage, Yahoo Finance, etc.
        popular_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "BTC-USD", "ETH-USD"]
        
        async def price_stream_worker():
            while True:
                try:
                    for symbol in popular_symbols:
                        # Simulate price updates (in production, connect to real data feeds)
                        await self._update_symbol_price(symbol)
                    await asyncio.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error(f"Price stream error: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
        
        # Start the background task
        asyncio.create_task(price_stream_worker())
    
    async def _start_news_sentiment_streams(self):
        """Start news and sentiment analysis streams"""
        async def news_sentiment_worker():
            while True:
                try:
                    # Process news sentiment for market analysis
                    await self._process_market_sentiment()
                    await asyncio.sleep(300)  # Update every 5 minutes
                except Exception as e:
                    logger.error(f"News sentiment error: {e}")
                    await asyncio.sleep(600)  # Wait before retrying
        
        # Start the background task
        asyncio.create_task(news_sentiment_worker())
    
    async def _update_symbol_price(self, symbol: str):
        """Update symbol price in cache and broadcast via WebSocket"""
        try:
            # In production, fetch real price data
            # For now, simulate price update
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            price_data = {
                "symbol": symbol,
                "price": info.get('currentPrice', 0),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Cache the price data
            if self.cache:
                await self.cache.set(f"price:{symbol}", price_data, ttl=30)
            
            # Broadcast to WebSocket clients
            if self.websocket_manager and hasattr(self.websocket_manager, 'broadcast_price_update'):
                await self.websocket_manager.broadcast_price_update(price_data)
            
        except Exception as e:
            logger.debug(f"Error updating price for {symbol}: {e}")
    
    async def _process_market_sentiment(self):
        """Process market sentiment analysis"""
        try:
            if self.intelligence_orchestrator:
                # Run sentiment analysis for major market indicators
                sentiment_data = await self.intelligence_orchestrator.analyze_market_sentiment()
                
                # Cache sentiment data
                if self.cache:
                    await self.cache.set("market_sentiment", sentiment_data, ttl=300)
                
                # Broadcast sentiment updates
                if self.websocket_manager and hasattr(self.websocket_manager, 'broadcast_sentiment_update'):
                    await self.websocket_manager.broadcast_sentiment_update(sentiment_data)
                
        except Exception as e:
            logger.debug(f"Error processing market sentiment: {e}")
    
    async def get_component_status(self) -> Dict[str, Any]:
        """Get status of all initialized components"""
        status = {
            "initialized": self.is_initialized,
            "cache": self.cache is not None,
            "websocket_manager": self.websocket_manager is not None,
            "intelligence_orchestrator": self.intelligence_orchestrator is not None,
            "active_connections": len(getattr(self.websocket_manager, 'active_connections', {}))
        }
        
        return status
    
    async def cleanup(self):
        """Cleanup all initialized components"""
        try:
            logger.info("🧹 Cleaning up FastAPI components...")
            
            if self.intelligence_orchestrator:
                await self.intelligence_orchestrator.cleanup()
            
            if self.cache and hasattr(self.cache, 'close'):
                await self.cache.close()
            
            if self.websocket_manager and hasattr(self.websocket_manager, 'disconnect_all'):
                await self.websocket_manager.disconnect_all()
            
            logger.info("✅ FastAPI cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


# Global initializer instance
_system_initializer: Optional[SystemInitializer] = None

def get_system_initializer() -> SystemInitializer:
    """Get global system initializer instance"""
    global _system_initializer
    if _system_initializer is None:
        _system_initializer = SystemInitializer()
    return _system_initializer 