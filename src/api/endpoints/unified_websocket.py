"""
Unified WebSocket API Endpoints
Consolidates functionality from:
- websocket_realtime.py: Agent status and wallet transaction updates
- websocket.py: Trading data, market updates, and notifications
- realtime.py: Unified real-time API with pub/sub integration
- routes/websocket.py: WebSocket connection management REST endpoints

This unified service provides:
- Real-time market data streaming
- Agent status updates
- Wallet transaction notifications
- Portfolio updates
- Trade execution notifications
- System notifications and alerts
- Connection management and statistics
"""

import logging
import json
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.realtime.websockets import WebSocketManager
from src.core.security import verify_token, TokenData
from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Unified router
router = APIRouter(prefix="/api/ws", tags=["Unified WebSocket"])

security = HTTPBearer()

# Global websocket manager instance (injected from main_refactored.py)
websocket_manager: Optional[WebSocketManager] = None

def set_websocket_manager(manager: WebSocketManager):
    """Set the websocket manager instance"""
    global websocket_manager
    websocket_manager = manager

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ConnectionInfo(BaseModel):
    client_id: str
    connected_at: str
    subscriptions: List[str]
    status: str

class WebSocketStats(BaseModel):
    total_connections: int
    active_connections: int
    total_messages_sent: int
    total_messages_received: int
    active_subscriptions: Dict[str, int]

# ============================================
# HELPER FUNCTIONS
# ============================================

async def get_user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenData]:
    """Extract user from WebSocket token"""
    if not credentials:
        return None
    try:
        return verify_token(credentials.credentials)
    except Exception:
        return None

# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@router.websocket("/trading")
async def websocket_trading(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    Main WebSocket endpoint for trading data
    
    Available channels:
    - market_data: General market data updates
    - market_data.{SYMBOL}: Symbol-specific market data
    - trades.{SYMBOL}: Symbol-specific trade executions
    - portfolio.{USER_ID}: User-specific portfolio updates
    - risk_alerts: Risk management alerts
    - news: News and market alerts
    """
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"trading_{uuid.uuid4()}"
    
    try:
        # Accept connection
        await websocket_manager.connect(websocket, client_id)
        logger.info(f"WebSocket trading client connected: {client_id}")
        
        # Authentication (optional for public data)
        user_id = None
        if token:
            try:
                user = verify_token(token)
                user_id = user.user_id if hasattr(user, 'user_id') else None
            except Exception as e:
                logger.warning(f"WebSocket authentication failed for {client_id}: {e}")
                # Continue without authentication for public data
        
        # Send initial connection confirmation
        await websocket_manager.send_to_client(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "channels_available": [
                "market_data",
                "market_data.{SYMBOL}",
                "trades.{SYMBOL}",
                "portfolio.{USER_ID}",
                "risk_alerts",
                "news"
            ],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Main message loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "subscribe":
                    channel = message.get("channel")
                    if channel:
                        await websocket_manager.subscribe_client(client_id, channel)
                
                elif msg_type == "unsubscribe":
                    channel = message.get("channel")
                    if channel:
                        await websocket_manager.unsubscribe_client(client_id, channel)
                
                elif msg_type == "ping":
                    await websocket_manager.send_to_client(client_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except json.JSONDecodeError:
                await websocket_manager.send_to_client(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                await websocket_manager.send_to_client(client_id, {
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket trading client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error for {client_id}: {e}")
    finally:
        if websocket_manager and client_id:
            await websocket_manager.disconnect(client_id)

@router.websocket("/agents/status")
async def websocket_agent_status(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for real-time agent status updates"""
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"agent_status_{datetime.utcnow().timestamp()}"
    
    try:
        # Accept connection
        await websocket_manager.connect(websocket, client_id)
        
        # Subscribe to agent status channel
        await websocket_manager.subscribe_client(client_id, "agent_status")
        
        # Send initial status
        await websocket_manager.send_to_client(client_id, {
            "type": "agent_status_update",
            "message": "Connected to agent status stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket_manager.send_to_client(client_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in agent status WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if websocket_manager and client_id:
            await websocket_manager.disconnect(client_id)

@router.websocket("/wallet/transactions")
async def websocket_wallet_transactions(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint for real-time wallet transaction updates"""
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"wallet_txns_{datetime.utcnow().timestamp()}"
    
    try:
        # Verify token if provided
        user = None
        if token:
            user = verify_token(token)
            if not user:
                await websocket.close(code=1008, reason="Invalid token")
                return
        
        # Accept connection
        await websocket_manager.connect(websocket, client_id)
        
        # Subscribe to wallet transactions channel (user-specific if authenticated)
        user_id = user.user_id if user and hasattr(user, 'user_id') else None
        channel = f"wallet_transactions_{user_id}" if user_id else "wallet_transactions"
        await websocket_manager.subscribe_client(client_id, channel)
        
        # Send initial confirmation
        await websocket_manager.send_to_client(client_id, {
            "type": "wallet_transaction_update",
            "message": "Connected to wallet transaction stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket_manager.send_to_client(client_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in wallet transactions WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if websocket_manager and client_id:
            await websocket_manager.disconnect(client_id)

@router.websocket("/market-data")
async def websocket_market_data(websocket: WebSocket):
    """Dedicated endpoint for market data streaming"""
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"market_data_{uuid.uuid4()}"
    
    try:
        await websocket_manager.connect(websocket, client_id)
        
        # Auto-subscribe to market data channels
        await websocket_manager.subscribe_client(client_id, "market_data")
        await websocket_manager.subscribe_client(client_id, "price_updates")
        
        await websocket_manager.send_to_client(client_id, {
            "type": "subscription_confirmed",
            "channels": ["market_data", "price_updates"],
            "message": "Subscribed to market data stream"
        })
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket_manager and client_id:
            await websocket_manager.disconnect(client_id)

@router.websocket("/portfolio/{portfolio_id}")
async def websocket_portfolio(
    websocket: WebSocket,
    portfolio_id: str
):
    """Dedicated endpoint for portfolio updates"""
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"portfolio_{portfolio_id}_{uuid.uuid4()}"
    
    try:
        await websocket_manager.connect(websocket, client_id)
        
        # Auto-subscribe to portfolio channels
        portfolio_channel = f"portfolio_{portfolio_id}"
        trades_channel = f"trades_portfolio_{portfolio_id}"
        
        await websocket_manager.subscribe_client(client_id, portfolio_channel)
        await websocket_manager.subscribe_client(client_id, trades_channel)
        
        await websocket_manager.send_to_client(client_id, {
            "type": "subscription_confirmed",
            "channels": [portfolio_channel, trades_channel],
            "portfolio_id": portfolio_id
        })
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        if websocket_manager and client_id:
            await websocket_manager.disconnect(client_id)

# ============================================
# REST ENDPOINTS FOR WEBSOCKET MANAGEMENT
# ============================================

@router.get("/stats", response_model=WebSocketStats)
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    stats = getattr(websocket_manager, 'stats', {})
    active_connections = len(getattr(websocket_manager, 'active_connections', {}))
    subscriptions = getattr(websocket_manager, 'subscriptions', {})
    
    return WebSocketStats(
        total_connections=stats.get('total_connections', 0),
        active_connections=active_connections,
        total_messages_sent=stats.get('messages_sent', 0),
        total_messages_received=stats.get('messages_received', 0),
        active_subscriptions={channel: len(clients) for channel, clients in subscriptions.items()}
    )

@router.get("/connections", response_model=List[ConnectionInfo])
async def get_active_connections():
    """Get list of all active WebSocket connections"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    connections = []
    active_conns = getattr(websocket_manager, 'active_connections', {})
    
    for client_id, connection in active_conns.items():
        subscriptions = getattr(connection, 'subscriptions', set())
        connected_at = getattr(connection, 'connected_at', datetime.utcnow())
        
        connections.append(ConnectionInfo(
            client_id=client_id,
            connected_at=connected_at.isoformat() if isinstance(connected_at, datetime) else str(connected_at),
            subscriptions=list(subscriptions),
            status="connected"
        ))
    
    return connections

@router.get("/channels", response_model=List[Dict[str, Any]])
async def get_available_channels():
    """Get list of available WebSocket channels"""
    return [
        {
            "channel": "market_data",
            "description": "Real-time market data updates",
            "subscription_format": "market_data or market_data:{symbol}",
            "example": "market_data:AAPL"
        },
        {
            "channel": "portfolio",
            "description": "Portfolio value and position updates",
            "subscription_format": "portfolio:{portfolio_id}",
            "example": "portfolio:12345"
        },
        {
            "channel": "agent_status",
            "description": "AI agent status and decision updates",
            "subscription_format": "agent_status",
            "example": "agent_status"
        },
        {
            "channel": "wallet_transactions",
            "description": "Wallet transaction updates (user-specific)",
            "subscription_format": "wallet_transactions or wallet_transactions:{user_id}",
            "example": "wallet_transactions:123"
        },
        {
            "channel": "trades",
            "description": "Trade execution confirmations",
            "subscription_format": "trades:{symbol} or trades_portfolio:{portfolio_id}",
            "example": "trades:AAPL"
        },
        {
            "channel": "risk_alerts",
            "description": "Risk management alerts",
            "subscription_format": "risk_alerts",
            "example": "risk_alerts"
        },
        {
            "channel": "news",
            "description": "Real-time news and sentiment updates",
            "subscription_format": "news or news:{symbol}",
            "example": "news:TSLA"
        }
    ]

@router.post("/broadcast/market-data")
async def broadcast_market_data(
    market_data: Dict[str, Any],
    current_user: TokenData = Depends(get_user_from_token)
):
    """
    Broadcast market data to WebSocket clients
    (Admin endpoint for testing/integration)
    """
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    try:
        await websocket_manager.broadcast_to_channel("market_data", {
            "type": "market_data",
            "data": market_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": "Market data broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting market data: {e}")
        raise HTTPException(status_code=500, detail=f"Error broadcasting market data: {str(e)}")

@router.post("/broadcast/trade")
async def broadcast_trade_execution(
    trade_data: Dict[str, Any],
    current_user: TokenData = Depends(get_user_from_token)
):
    """
    Broadcast trade execution to WebSocket clients
    (Admin endpoint for testing/integration)
    """
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    try:
        symbol = trade_data.get("symbol", "unknown")
        await websocket_manager.broadcast_to_channel(f"trades_{symbol}", {
            "type": "trade_executed",
            "data": trade_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": "Trade execution broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting trade execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error broadcasting trade execution: {str(e)}")

@router.post("/send/risk-alert")
async def send_risk_alert(
    user_id: str,
    alert_data: Dict[str, Any],
    current_user: TokenData = Depends(get_user_from_token)
):
    """
    Send risk alert to specific user via WebSocket
    (Admin endpoint for testing/integration)
    """
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    try:
        await websocket_manager.broadcast_to_channel("risk_alerts", {
            "type": "risk_alert",
            "user_id": user_id,
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"Risk alert sent for user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error sending risk alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending risk alert: {str(e)}")

@router.post("/notify/user/{user_id}")
async def notify_user(
    user_id: str,
    notification: Dict[str, Any],
    current_user: TokenData = Depends(get_user_from_token)
):
    """Send notification to specific user via WebSocket"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    try:
        # Find all connections for this user (would need user tracking in production)
        # For now, broadcast to a user-specific channel
        channel = f"notifications_{user_id}"
        await websocket_manager.broadcast_to_channel(channel, {
            "type": "notification",
            "user_id": user_id,
            "data": notification,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"Notification sent to user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error sending user notification: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending notification: {str(e)}")

@router.delete("/connections/{client_id}")
async def disconnect_client(
    client_id: str,
    current_user: TokenData = Depends(get_user_from_token)
):
    """Forcefully disconnect a specific WebSocket client"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    
    if client_id not in getattr(websocket_manager, 'active_connections', {}):
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    
    try:
        await websocket_manager.disconnect(client_id)
        return {
            "status": "success",
            "message": f"Client {client_id} disconnected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disconnect failed: {str(e)}")

@router.get("/health")
async def websocket_health():
    """Health check for WebSocket service"""
    try:
        if not websocket_manager:
            return {
                "status": "unhealthy",
                "service": "websocket",
                "message": "WebSocket manager not initialized",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        active_connections = len(getattr(websocket_manager, 'active_connections', {}))
        stats = getattr(websocket_manager, 'stats', {})
        
        return {
            "status": "healthy",
            "service": "websocket",
            "timestamp": datetime.utcnow().isoformat(),
            "active_connections": active_connections,
            "total_messages": stats.get('messages_sent', 0) + stats.get('messages_received', 0),
            "websocket_manager": "initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "websocket",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "active_connections": 0
        }

# ============================================
# HELPER FUNCTIONS FOR BROADCASTING
# ============================================

async def broadcast_agent_status_update(agent_id: str, status_data: dict):
    """Broadcast agent status update to all subscribed clients"""
    if websocket_manager:
        await websocket_manager.broadcast_to_channel("agent_status", {
            "type": "agent_status_update",
            "agent_id": agent_id,
            "data": status_data,
            "timestamp": datetime.utcnow().isoformat()
        })

async def broadcast_wallet_transaction_update(user_id: int, transaction_data: dict):
    """Broadcast wallet transaction update to user's subscribed clients"""
    if websocket_manager:
        await websocket_manager.broadcast_to_channel(f"wallet_transactions_{user_id}", {
            "type": "wallet_transaction_update",
            "user_id": user_id,
            "data": transaction_data,
            "timestamp": datetime.utcnow().isoformat()
        })

