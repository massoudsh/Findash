"""
WebSocket Bridge - Connects Redis Pub/Sub to WebSocket clients
Bridges backend pub/sub events to frontend WebSocket connections
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from src.realtime.websockets import WebSocketManager
from src.realtime.unified_pubsub import (
    UnifiedPubSubManager,
    PubSubChannel,
    PubSubMessage,
    get_unified_pubsub
)

logger = logging.getLogger(__name__)

class WebSocketBridge:
    """
    Bridges Redis Pub/Sub channels to WebSocket clients
    Automatically forwards pub/sub messages to subscribed WebSocket clients
    """
    
    def __init__(
        self,
        websocket_manager: WebSocketManager,
        pubsub_manager: UnifiedPubSubManager
    ):
        self.websocket_manager = websocket_manager
        self.pubsub_manager = pubsub_manager
        self.client_subscriptions: Dict[str, set] = {}  # client_id -> set of channels
        self.running = False
    
    async def initialize(self):
        """Initialize the bridge and subscribe to common channels"""
        self.running = True
        
        # Subscribe to all market data channels
        await self.pubsub_manager.subscribe(
            PubSubChannel.MARKET_DATA.value,
            self._forward_to_websockets
        )
        
        # Subscribe to trades channel
        await self.pubsub_manager.subscribe(
            PubSubChannel.TRADES.value,
            self._forward_to_websockets
        )
        
        # Subscribe to system health
        await self.pubsub_manager.subscribe(
            PubSubChannel.SYSTEM_HEALTH.value,
            self._forward_to_websockets
        )
        
        # Subscribe to sentiment
        await self.pubsub_manager.subscribe(
            PubSubChannel.SENTIMENT.value,
            self._forward_to_websockets
        )
        
        logger.info("🌉 WebSocket Bridge initialized")
    
    async def _forward_to_websockets(self, message: PubSubMessage):
        """Forward pub/sub message to WebSocket subscribers"""
        try:
            # Broadcast to all WebSocket clients subscribed to this channel
            await self.websocket_manager.broadcast_to_channel(
                message.channel,
                {
                    "type": message.event_type,
                    "channel": message.channel,
                    "data": message.data,
                    "timestamp": message.timestamp.isoformat(),
                    "source": message.source
                }
            )
        except Exception as e:
            logger.error(f"❌ Failed to forward message to WebSockets: {e}")
    
    async def handle_client_subscribe(
        self,
        client_id: str,
        channel: str
    ) -> bool:
        """Handle client subscription request"""
        if client_id not in self.client_subscriptions:
            self.client_subscriptions[client_id] = set()
        
        self.client_subscriptions[client_id].add(channel)
        
        # Subscribe to pub/sub channel if not already subscribed
        # (The bridge handles common channels, but we can add dynamic ones)
        
        logger.info(f"👤 Client {client_id} subscribed to {channel}")
        return True
    
    async def handle_client_unsubscribe(
        self,
        client_id: str,
        channel: str
    ) -> bool:
        """Handle client unsubscription request"""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].discard(channel)
            logger.info(f"👤 Client {client_id} unsubscribed from {channel}")
        return True
    
    async def handle_client_disconnect(self, client_id: str):
        """Clean up when client disconnects"""
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
            logger.info(f"👤 Client {client_id} disconnected from bridge")
    
    def get_client_subscriptions(self, client_id: str) -> set:
        """Get all channels a client is subscribed to"""
        return self.client_subscriptions.get(client_id, set())

# Global bridge instance
_websocket_bridge: Optional[WebSocketBridge] = None

async def get_websocket_bridge() -> WebSocketBridge:
    """Get or create global WebSocket bridge instance"""
    global _websocket_bridge
    if _websocket_bridge is None:
        from src.realtime.websockets import websocket_manager
        pubsub = await get_unified_pubsub()
        _websocket_bridge = WebSocketBridge(
            websocket_manager=websocket_manager,
            pubsub_manager=pubsub
        )
        await _websocket_bridge.initialize()
    return _websocket_bridge

