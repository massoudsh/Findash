"""
WebSocket API Routes for FastAPI service
Handles WebSocket connection management, subscriptions, and real-time messaging
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json
import asyncio

# Use consolidated WebSocket implementation
from src.realtime.websockets import WebSocketManager

# Create global websocket manager instance
websocket_manager = WebSocketManager()

router = APIRouter()

# Pydantic models
class ConnectionInfo(BaseModel):
    client_id: str
    connected_at: str
    subscriptions: List[str]
    status: str

class SubscriptionRequest(BaseModel):
    client_id: str
    channels: List[str]
    symbols: Optional[List[str]] = None

class MessageRequest(BaseModel):
    channel: str
    message: Dict[str, Any]
    target_clients: Optional[List[str]] = None

class WebSocketStats(BaseModel):
    total_connections: int
    active_connections: int
    total_messages_sent: int
    total_messages_received: int
    active_subscriptions: Dict[str, int]
    uptime: str

@router.get("/connections", response_model=List[ConnectionInfo])
async def get_active_connections():
    """Get list of all active WebSocket connections"""
    
    connections = []
    
    # Get connections from websocket manager
    if hasattr(websocket_manager, 'active_connections'):
        for client_id, connection in websocket_manager.active_connections.items():
            connections.append(ConnectionInfo(
                client_id=client_id,
                connected_at=getattr(connection, 'connected_at', datetime.utcnow().isoformat()),
                subscriptions=getattr(connection, 'subscriptions', []),
                status="connected"
            ))
    
    return connections

@router.get("/stats", response_model=WebSocketStats)
async def get_websocket_stats():
    """Get WebSocket service statistics"""
    
    # Get stats from websocket manager
    stats = getattr(websocket_manager, 'stats', {})
    active_connections = len(getattr(websocket_manager, 'active_connections', {}))
    
    return WebSocketStats(
        total_connections=stats.get('total_connections', 0),
        active_connections=active_connections,
        total_messages_sent=stats.get('messages_sent', 0),
        total_messages_received=stats.get('messages_received', 0),
        active_subscriptions=stats.get('subscriptions', {}),
        uptime="active"
    )

@router.post("/subscribe", response_model=Dict[str, str])
async def manage_subscriptions(request: SubscriptionRequest):
    """Manage WebSocket subscriptions for a client"""
    
    if not hasattr(websocket_manager, 'active_connections'):
        raise HTTPException(status_code=404, detail="WebSocket manager not initialized")
    
    if request.client_id not in websocket_manager.active_connections:
        raise HTTPException(status_code=404, detail=f"Client {request.client_id} not connected")
    
    try:
        # In production, this would manage actual subscriptions
        # For now, just acknowledge the request
        
        subscription_message = {
            "type": "subscription_update",
            "channels": request.channels,
            "symbols": request.symbols or [],
            "status": "active",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send confirmation to the specific client
        connection = websocket_manager.active_connections[request.client_id]
        await connection.send_text(json.dumps(subscription_message))
        
        return {
            "status": "success",
            "message": f"Subscriptions updated for client {request.client_id}",
            "channels": request.channels
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription management failed: {str(e)}")

@router.post("/broadcast", response_model=Dict[str, str])
async def broadcast_message(request: MessageRequest):
    """Broadcast a message to WebSocket clients"""
    
    if not hasattr(websocket_manager, 'active_connections'):
        raise HTTPException(status_code=500, detail="WebSocket manager not initialized")
    
    try:
        message = {
            "channel": request.channel,
            "data": request.message,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "broadcast"
        }
        
        message_json = json.dumps(message)
        sent_count = 0
        
        if request.target_clients:
            # Send to specific clients
            for client_id in request.target_clients:
                if client_id in websocket_manager.active_connections:
                    try:
                        await websocket_manager.active_connections[client_id].send_text(message_json)
                        sent_count += 1
                    except Exception:
                        continue  # Skip failed sends
        else:
            # Broadcast to all connected clients
            for client_id, connection in websocket_manager.active_connections.items():
                try:
                    await connection.send_text(message_json)
                    sent_count += 1
                except Exception:
                    continue  # Skip failed sends
        
        return {
            "status": "success",
            "message": f"Message broadcasted to {sent_count} clients",
            "channel": request.channel
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")

@router.delete("/connections/{client_id}", response_model=Dict[str, str])
async def disconnect_client(client_id: str):
    """Forcefully disconnect a specific WebSocket client"""
    
    if not hasattr(websocket_manager, 'active_connections'):
        raise HTTPException(status_code=404, detail="WebSocket manager not initialized")
    
    if client_id not in websocket_manager.active_connections:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    try:
        # Disconnect the client
        await websocket_manager.disconnect(client_id)
        
        return {
            "status": "success",
            "message": f"Client {client_id} disconnected",
            "client_id": client_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")

@router.get("/channels", response_model=List[Dict[str, Any]])
async def get_available_channels():
    """Get list of available WebSocket channels"""
    
    return [
        {
            "channel": "market_data",
            "description": "Real-time market data updates",
            "subscription_format": "market_data:{symbol}",
            "example": "market_data:AAPL"
        },
        {
            "channel": "portfolio",
            "description": "Portfolio value and position updates",
            "subscription_format": "portfolio:{portfolio_id}",
            "example": "portfolio:12345"
        },
        {
            "channel": "alerts",
            "description": "Price alerts and notifications",
            "subscription_format": "alerts:{user_id}",
            "example": "alerts:user123"
        },
        {
            "channel": "news",
            "description": "Real-time news and sentiment updates",
            "subscription_format": "news:{symbol}",
            "example": "news:TSLA"
        },
        {
            "channel": "trades",
            "description": "Trade execution confirmations",
            "subscription_format": "trades:{portfolio_id}",
            "example": "trades:12345"
        },
        {
            "channel": "system",
            "description": "System announcements and maintenance alerts",
            "subscription_format": "system",
            "example": "system"
        }
    ]

@router.post("/test-connection", response_model=Dict[str, str])
async def test_websocket_connection():
    """Test WebSocket functionality"""
    
    try:
        # Test basic WebSocket manager functionality
        if not hasattr(websocket_manager, 'active_connections'):
            return {
                "status": "warning",
                "message": "WebSocket manager not fully initialized",
                "recommendation": "Restart the service"
            }
        
        active_count = len(websocket_manager.active_connections)
        
        # Send a test message to all connected clients
        test_message = {
            "type": "test",
            "message": "WebSocket connection test",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if active_count > 0:
            await websocket_manager.broadcast_to_all(test_message)
        
        return {
            "status": "success",
            "message": f"WebSocket test completed. {active_count} active connections",
            "active_connections": active_count
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"WebSocket test failed: {str(e)}",
            "recommendation": "Check WebSocket manager configuration"
        }

@router.get("/health", response_model=Dict[str, Any])
async def websocket_health():
    """Health check for WebSocket service"""
    
    try:
        active_connections = len(getattr(websocket_manager, 'active_connections', {}))
        stats = getattr(websocket_manager, 'stats', {})
        
        return {
            "status": "healthy",
            "service": "websocket",
            "timestamp": datetime.utcnow().isoformat(),
            "active_connections": active_connections,
            "total_messages": stats.get('messages_sent', 0) + stats.get('messages_received', 0),
            "websocket_manager": "initialized" if hasattr(websocket_manager, 'active_connections') else "not_initialized",
            "supported_channels": 6
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "websocket",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "active_connections": 0
        }

# WebSocket endpoint handler
@router.websocket("/connect/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket connection endpoint"""
    
    await websocket_manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "client_id": client_id,
            "server_time": datetime.utcnow().isoformat(),
            "available_channels": ["market_data", "portfolio", "alerts", "news", "trades", "system"]
        }
        await websocket.send_text(json.dumps(welcome_message))
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(client_id, data)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.disconnect(client_id) 