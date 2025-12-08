"""
Unified Event Bus - Central communication hub for all modules
Provides a single interface for all inter-module communication via pub/sub
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from src.realtime.unified_pubsub import (
    get_unified_pubsub,
    PubSubChannel,
    PubSubMessage
)

logger = logging.getLogger(__name__)

class UnifiedEventBus:
    """
    Unified Event Bus for module communication
    All modules should use this for inter-module communication
    """
    
    def __init__(self):
        self.pubsub = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the event bus"""
        if not self._initialized:
            self.pubsub = await get_unified_pubsub()
            self._initialized = True
            logger.info("✅ Unified Event Bus initialized")
    
    async def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        channel: Optional[str] = None,
        source: str = "system",
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Publish an event to the event bus
        
        Args:
            event_type: Type of event (e.g., "price_update", "trade_executed")
            data: Event payload
            channel: Optional specific channel (defaults to event_type)
            source: Source module/service name
            correlation_id: Optional correlation ID
            user_id: Optional user ID
        """
        if not self._initialized:
            await self.initialize()
        
        channel = channel or event_type
        return await self.pubsub.publish(
            channel=channel,
            event_type=event_type,
            data=data,
            source=source,
            correlation_id=correlation_id,
            user_id=user_id
        )
    
    async def subscribe(
        self,
        channel: str,
        handler: Callable[[PubSubMessage], Awaitable[None]]
    ):
        """Subscribe to a channel"""
        if not self._initialized:
            await self.initialize()
        
        await self.pubsub.subscribe(channel, handler)
    
    async def unsubscribe(self, channel: str, handler: Callable):
        """Unsubscribe from a channel"""
        if not self._initialized:
            await self.initialize()
        
        await self.pubsub.unsubscribe(channel, handler)
    
    # Convenience methods for common events
    
    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        change: float,
        volume: float,
        **kwargs
    ):
        """Publish market data update"""
        return await self.publish_event(
            event_type="price_update",
            data={
                "symbol": symbol,
                "price": price,
                "change": change,
                "volume": volume,
                **kwargs
            },
            channel=PubSubChannel.MARKET_DATA_SYMBOL.value.format(symbol=symbol),
            source="market_data_processor"
        )
    
    async def publish_trade(
        self,
        portfolio_id: str,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: float,
        **kwargs
    ):
        """Publish trade execution"""
        return await self.publish_event(
            event_type="trade_executed",
            data={
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price,
                "total": quantity * price,
                **kwargs
            },
            channel=PubSubChannel.TRADES_PORTFOLIO.value.format(portfolio_id=portfolio_id),
            source="execution_manager",
            user_id=portfolio_id
        )
    
    async def publish_portfolio_update(
        self,
        portfolio_id: str,
        total_value: float,
        cash_balance: float,
        positions: list,
        **kwargs
    ):
        """Publish portfolio update"""
        return await self.publish_event(
            event_type="portfolio_update",
            data={
                "total_value": total_value,
                "cash_balance": cash_balance,
                "positions": positions,
                **kwargs
            },
            channel=PubSubChannel.PORTFOLIO.value.format(portfolio_id=portfolio_id),
            source="portfolio_manager",
            user_id=portfolio_id
        )
    
    async def publish_agent_result(
        self,
        agent_id: str,
        pipeline_id: str,
        result: Dict[str, Any],
        status: str = "completed"
    ):
        """Publish AI agent result"""
        return await self.publish_event(
            event_type="agent_result",
            data={
                "agent_id": agent_id,
                "pipeline_id": pipeline_id,
                "result": result,
                "status": status
            },
            channel=PubSubChannel.AGENT_RESULT.value.format(agent_id=agent_id),
            source="intelligence_orchestrator"
        )
    
    async def publish_risk_alert(
        self,
        portfolio_id: str,
        alert_type: str,
        message: str,
        severity: str = "warning",
        **kwargs
    ):
        """Publish risk alert"""
        return await self.publish_event(
            event_type="risk_alert",
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                **kwargs
            },
            channel=PubSubChannel.RISK_ALERTS.value,
            source="risk_manager",
            user_id=portfolio_id
        )

# Global event bus instance
_event_bus: Optional[UnifiedEventBus] = None

async def get_event_bus() -> UnifiedEventBus:
    """Get or create global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = UnifiedEventBus()
        await _event_bus.initialize()
    return _event_bus

