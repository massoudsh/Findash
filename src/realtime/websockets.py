"""
WebSocket Manager for Real-time Communication
Handles WebSocket connections, subscriptions, and real-time data streaming
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection with metadata"""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    subscriptions: Set[str]
    last_ping: Optional[datetime] = None

class WebSocketManager:
    """
    Manages WebSocket connections and handles real-time communication
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        self.stats = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "subscriptions": {}
        }
    
    async def initialize(self):
        """Initialize the WebSocket manager (async setup if needed)"""
        logger.info("WebSocket manager initialized")
        return True
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        connection = WebSocketConnection(
            websocket=websocket,
            client_id=client_id,
            connected_at=datetime.utcnow(),
            subscriptions=set()
        )
        
        self.active_connections[client_id] = connection
        self.stats["total_connections"] += 1
        
        logger.info(f"WebSocket connection established for client {client_id}")
        
        # Send connection confirmation
        await self.send_to_client(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            
            # Remove from all subscriptions
            for channel in connection.subscriptions:
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(client_id)
                    if not self.subscriptions[channel]:
                        del self.subscriptions[channel]
            
            # Close the connection
            try:
                await connection.websocket.close()
            except Exception:
                pass  # Connection might already be closed
            
            del self.active_connections[client_id]
            logger.info(f"WebSocket connection closed for client {client_id}")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client"""
        if client_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent client {client_id}")
            return False
        
        try:
            connection = self.active_connections[client_id]
            await connection.websocket.send_text(json.dumps(message))
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to client {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return 0
        
        message["timestamp"] = datetime.utcnow().isoformat()
        message_json = json.dumps(message)
        
        sent_count = 0
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            try:
                await connection.websocket.send_text(message_json)
                sent_count += 1
                self.stats["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
        
        logger.debug(f"Broadcasted message to {sent_count} clients")
        return sent_count
    
    async def subscribe_client(self, client_id: str, channel: str):
        """Subscribe a client to a specific channel"""
        if client_id not in self.active_connections:
            return False
        
        # Add to client's subscriptions
        self.active_connections[client_id].subscriptions.add(channel)
        
        # Add to channel subscriptions
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        self.subscriptions[channel].add(client_id)
        
        # Update stats
        if channel not in self.stats["subscriptions"]:
            self.stats["subscriptions"][channel] = 0
        self.stats["subscriptions"][channel] += 1
        
        logger.info(f"Client {client_id} subscribed to channel {channel}")
        
        # Send subscription confirmation
        await self.send_to_client(client_id, {
            "type": "subscription_confirmed",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    async def unsubscribe_client(self, client_id: str, channel: str):
        """Unsubscribe a client from a specific channel"""
        if client_id not in self.active_connections:
            return False
        
        # Remove from client's subscriptions
        self.active_connections[client_id].subscriptions.discard(channel)
        
        # Remove from channel subscriptions
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(client_id)
            if not self.subscriptions[channel]:
                del self.subscriptions[channel]
        
        # Update stats
        if channel in self.stats["subscriptions"]:
            self.stats["subscriptions"][channel] -= 1
            if self.stats["subscriptions"][channel] <= 0:
                del self.stats["subscriptions"][channel]
        
        logger.info(f"Client {client_id} unsubscribed from channel {channel}")
        
        # Send unsubscription confirmation
        await self.send_to_client(client_id, {
            "type": "unsubscription_confirmed",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return True
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast a message to all subscribers of a specific channel"""
        if channel not in self.subscriptions:
            logger.debug(f"No subscribers for channel {channel}")
            return 0
        
        message["channel"] = channel
        message["timestamp"] = datetime.utcnow().isoformat()
        message_json = json.dumps(message)
        
        sent_count = 0
        disconnected_clients = []
        
        for client_id in self.subscriptions[channel].copy():
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].websocket.send_text(message_json)
                    sent_count += 1
                    self.stats["messages_sent"] += 1
                except Exception as e:
                    logger.error(f"Failed to send to client {client_id} on channel {channel}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
        
        logger.debug(f"Broadcasted message to {sent_count} subscribers of channel {channel}")
        return sent_count
    
    async def handle_message(self, client_id: str, message: str):
        """Handle incoming message from a client"""
        if client_id not in self.active_connections:
            return
        
        try:
            data = json.loads(message)
            self.stats["messages_received"] += 1
            
            message_type = data.get("type", "unknown")
            
            if message_type == "subscribe":
                channel = data.get("channel")
                if channel:
                    await self.subscribe_client(client_id, channel)
            
            elif message_type == "unsubscribe":
                channel = data.get("channel") 
                if channel:
                    await self.unsubscribe_client(client_id, channel)
            
            elif message_type == "ping":
                # Update last ping time
                self.active_connections[client_id].last_ping = datetime.utcnow()
                # Send pong response
                await self.send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "get_subscriptions":
                # Send current subscriptions
                subscriptions = list(self.active_connections[client_id].subscriptions)
                await self.send_to_client(client_id, {
                    "type": "subscriptions",
                    "subscriptions": subscriptions,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                logger.warning(f"Unknown message type '{message_type}' from client {client_id}")
                await self.send_to_client(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received from client {client_id}")
            await self.send_to_client(client_id, {
                "type": "error",
                "message": "Invalid JSON format",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")
    
    async def send_market_data(self, symbol: str, data: Dict[str, Any]):
        """Send market data to subscribers"""
        channel = f"market_data:{symbol}"
        message = {
            "type": "market_data",
            "symbol": symbol,
            "data": data
        }
        return await self.broadcast_to_channel(channel, message)
    
    async def send_portfolio_update(self, portfolio_id: str, data: Dict[str, Any]):
        """Send portfolio update to subscribers"""
        channel = f"portfolio:{portfolio_id}"
        message = {
            "type": "portfolio_update",
            "portfolio_id": portfolio_id,
            "data": data
        }
        return await self.broadcast_to_channel(channel, message)
    
    async def send_alert(self, user_id: str, alert_data: Dict[str, Any]):
        """Send alert to a specific user"""
        channel = f"alerts:{user_id}"
        message = {
            "type": "alert",
            "data": alert_data
        }
        return await self.broadcast_to_channel(channel, message)
    
    async def send_trade_update(self, portfolio_id: str, trade_data: Dict[str, Any]):
        """Send trade execution update"""
        channel = f"trades:{portfolio_id}"
        message = {
            "type": "trade_update",
            "data": trade_data
        }
        return await self.broadcast_to_channel(channel, message)
    
    def get_connection_count(self) -> int:
        """Get current number of active connections"""
        return len(self.active_connections)
    
    def get_channel_subscribers(self, channel: str) -> int:
        """Get number of subscribers for a channel"""
        return len(self.subscriptions.get(channel, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection and messaging statistics"""
        return {
            **self.stats,
            "active_connections": len(self.active_connections),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "active_channels": len(self.subscriptions)
        }
    
    async def cleanup_stale_connections(self, timeout_minutes: int = 30):
        """Clean up connections that haven't pinged recently"""
        timeout_threshold = datetime.utcnow().timestamp() - (timeout_minutes * 60)
        stale_clients = []
        
        for client_id, connection in self.active_connections.items():
            last_ping = connection.last_ping or connection.connected_at
            if last_ping.timestamp() < timeout_threshold:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            logger.info(f"Cleaning up stale connection for client {client_id}")
            await self.disconnect(client_id)
        
        return len(stale_clients)
    
    async def broadcast_price_update(self, price_data: Dict[str, Any]):
        """Broadcast price update to all clients subscribed to price feeds"""
        symbol = price_data.get("symbol", "UNKNOWN")
        message = {
            "type": "price_update",
            "data": price_data
        }
        
        # Broadcast to price channel subscribers
        await self.broadcast_to_channel(f"prices:{symbol}", message)
        await self.broadcast_to_channel("prices:all", message)
        
        logger.debug(f"Broadcasted price update for {symbol}")
    
    async def broadcast_sentiment_update(self, sentiment_data: Dict[str, Any]):
        """Broadcast sentiment analysis update to subscribers"""
        message = {
            "type": "sentiment_update",
            "data": sentiment_data
        }
        
        # Broadcast to sentiment channel subscribers
        await self.broadcast_to_channel("sentiment", message)
        
        logger.debug("Broadcasted sentiment update")
    
    async def disconnect_all(self):
        """Disconnect all active connections"""
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            await self.disconnect(client_id)
        
        logger.info(f"Disconnected all {len(client_ids)} WebSocket connections")

# Global WebSocket manager instance
websocket_manager = WebSocketManager() 