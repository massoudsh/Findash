"""
Unified Pub/Sub System - Bridges Redis, Kafka, and WebSockets
Provides seamless real-time communication across all platform modules
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from src.realtime.websockets import WebSocketManager

logger = logging.getLogger(__name__)

class PubSubChannel(str, Enum):
    """Unified channel names for all platform events"""
    # Market Data Channels
    MARKET_DATA = "market_data"
    MARKET_DATA_SYMBOL = "market_data:{symbol}"  # Use format() for symbol
    PRICE_UPDATES = "price_updates"
    ORDERBOOK = "orderbook:{symbol}"
    
    # Trading Channels
    TRADES = "trades"
    TRADES_PORTFOLIO = "trades:{portfolio_id}"
    ORDERS = "orders"
    ORDER_STATUS = "order_status:{order_id}"
    
    # Portfolio Channels
    PORTFOLIO = "portfolio:{portfolio_id}"
    POSITIONS = "positions:{portfolio_id}"
    PORTFOLIO_VALUE = "portfolio_value:{portfolio_id}"
    
    # AI Agent Channels
    AGENT_STATUS = "agent_status"
    AGENT_RESULT = "agent_result:{agent_id}"
    PIPELINE_UPDATE = "pipeline:{pipeline_id}"
    
    # Risk & Compliance
    RISK_ALERTS = "risk_alerts"
    COMPLIANCE_EVENTS = "compliance_events"
    
    # System Channels
    SYSTEM_HEALTH = "system_health"
    METRICS = "metrics"
    NOTIFICATIONS = "notifications:{user_id}"
    
    # Sentiment & Alternative Data
    SENTIMENT = "sentiment"
    SENTIMENT_SYMBOL = "sentiment:{symbol}"
    NEWS = "news"
    SOCIAL = "social"

@dataclass
class PubSubMessage:
    """Standardized pub/sub message format"""
    channel: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PubSubMessage":
        return cls(
            channel=data["channel"],
            event_type=data["event_type"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            correlation_id=data.get("correlation_id"),
            user_id=data.get("user_id")
        )

class UnifiedPubSubManager:
    """
    Unified Pub/Sub Manager that bridges Redis, Kafka, and WebSockets
    Provides a single interface for all real-time communication
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        websocket_manager: Optional[WebSocketManager] = None
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.websocket_manager = websocket_manager
        self.subscriptions: Dict[str, Set[Callable]] = {}
        self.running = False
        self._listener_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize Redis connection and pub/sub"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            self.running = True
            logger.info("✅ Unified Pub/Sub Manager initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pub/Sub Manager: {e}")
            return False
    
    async def publish(
        self,
        channel: str,
        event_type: str,
        data: Dict[str, Any],
        source: str = "system",
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Publish message to channel (Redis + WebSocket broadcast)
        
        Args:
            channel: Channel name (can use format strings like "market_data:{symbol}")
            event_type: Type of event (e.g., "price_update", "trade_executed")
            data: Event payload
            source: Source service/module name
            correlation_id: Optional correlation ID for tracking
            user_id: Optional user ID for user-specific channels
        """
        try:
            # Create standardized message
            message = PubSubMessage(
                channel=channel,
                event_type=event_type,
                data=data,
                timestamp=datetime.utcnow(),
                source=source,
                correlation_id=correlation_id,
                user_id=user_id
            )
            
            # Publish to Redis
            if self.redis_client:
                await self.redis_client.publish(
                    channel,
                    json.dumps(message.to_dict())
                )
            
            # Broadcast to WebSocket subscribers
            if self.websocket_manager:
                await self.websocket_manager.broadcast_to_channel(
                    channel,
                    {
                        "type": event_type,
                        "channel": channel,
                        **message.to_dict()
                    }
                )
            
            logger.debug(f"📤 Published {event_type} to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish to {channel}: {e}")
            return False
    
    async def subscribe(
        self,
        channel: str,
        handler: Callable[[PubSubMessage], Awaitable[None]]
    ):
        """
        Subscribe to a channel with a handler function
        
        Args:
            channel: Channel name to subscribe to
            handler: Async function that receives PubSubMessage
        """
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
            # Subscribe to Redis channel
            if self.pubsub:
                await self.pubsub.subscribe(channel)
                logger.info(f"📡 Subscribed to Redis channel: {channel}")
        
        self.subscriptions[channel].add(handler)
        
        # Start listener if not running
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())
    
    async def unsubscribe(self, channel: str, handler: Callable):
        """Unsubscribe handler from channel"""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(handler)
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]
                if self.pubsub:
                    await self.pubsub.unsubscribe(channel)
    
    async def _listen(self):
        """Internal listener that processes Redis messages"""
        logger.info("🎧 Starting unified pub/sub listener...")
        while self.running:
            try:
                if not self.pubsub:
                    await asyncio.sleep(1)
                    continue
                
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                
                if message and message.get("type") == "message":
                    channel = message.get("channel")
                    data_str = message.get("data")
                    
                    if channel and data_str:
                        try:
                            data = json.loads(data_str)
                            pubsub_msg = PubSubMessage.from_dict(data)
                            
                            # Call all handlers for this channel
                            if channel in self.subscriptions:
                                for handler in self.subscriptions[channel]:
                                    try:
                                        await handler(pubsub_msg)
                                    except Exception as e:
                                        logger.error(
                                            f"❌ Handler error for {channel}: {e}",
                                            exc_info=True
                                        )
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Invalid JSON in message from {channel}: {e}")
                        except Exception as e:
                            logger.error(f"❌ Error processing message from {channel}: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Error in pub/sub listener: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        change: float,
        volume: float,
        **kwargs
    ):
        """Convenience method for publishing market data"""
        channel = PubSubChannel.MARKET_DATA_SYMBOL.value.format(symbol=symbol)
        return await self.publish(
            channel=channel,
            event_type="price_update",
            data={
                "symbol": symbol,
                "price": price,
                "change": change,
                "change_percent": (change / (price - change)) * 100 if change != 0 else 0,
                "volume": volume,
                **kwargs
            },
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
        """Convenience method for publishing trade events"""
        # Publish to portfolio-specific channel
        portfolio_channel = PubSubChannel.TRADES_PORTFOLIO.value.format(
            portfolio_id=portfolio_id
        )
        await self.publish(
            channel=portfolio_channel,
            event_type="trade_executed",
            data={
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price,
                "total": quantity * price,
                **kwargs
            },
            source="execution_manager",
            user_id=portfolio_id
        )
        
        # Also publish to general trades channel
        await self.publish(
            channel=PubSubChannel.TRADES.value,
            event_type="trade_executed",
            data={
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price,
                **kwargs
            },
            source="execution_manager"
        )
    
    async def publish_portfolio_update(
        self,
        portfolio_id: str,
        total_value: float,
        cash_balance: float,
        positions: List[Dict],
        **kwargs
    ):
        """Convenience method for publishing portfolio updates"""
        channel = PubSubChannel.PORTFOLIO.value.format(portfolio_id=portfolio_id)
        return await self.publish(
            channel=channel,
            event_type="portfolio_update",
            data={
                "total_value": total_value,
                "cash_balance": cash_balance,
                "positions": positions,
                **kwargs
            },
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
        """Convenience method for publishing AI agent results"""
        channel = PubSubChannel.AGENT_RESULT.value.format(agent_id=agent_id)
        return await self.publish(
            channel=channel,
            event_type="agent_result",
            data={
                "agent_id": agent_id,
                "pipeline_id": pipeline_id,
                "result": result,
                "status": status
            },
            source="intelligence_orchestrator"
        )
    
    async def publish_sentiment(
        self,
        symbol: str,
        sentiment: float,
        sources: Dict[str, float],
        **kwargs
    ):
        """Convenience method for publishing sentiment data"""
        channel = PubSubChannel.SENTIMENT_SYMBOL.value.format(symbol=symbol)
        return await self.publish(
            channel=channel,
            event_type="sentiment_update",
            data={
                "symbol": symbol,
                "sentiment": sentiment,
                "sources": sources,
                **kwargs
            },
            source="sentiment_analyzer"
        )
    
    async def close(self):
        """Close connections and cleanup"""
        self.running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("🔌 Unified Pub/Sub Manager closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "running": self.running,
            "subscribed_channels": list(self.subscriptions.keys()),
            "total_subscriptions": sum(len(handlers) for handlers in self.subscriptions.values()),
            "redis_connected": self.redis_client is not None,
            "websocket_manager": self.websocket_manager is not None
        }

# Global singleton instance
_unified_pubsub: Optional[UnifiedPubSubManager] = None

async def get_unified_pubsub() -> UnifiedPubSubManager:
    """Get or create global unified pub/sub instance"""
    global _unified_pubsub
    if _unified_pubsub is None:
        from src.realtime.websockets import websocket_manager
        _unified_pubsub = UnifiedPubSubManager(
            websocket_manager=websocket_manager
        )
        await _unified_pubsub.initialize()
    return _unified_pubsub

