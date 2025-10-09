"""
WebSocket endpoints for Quantum Trading Matrix™
Real-time trading data, market updates, and notifications
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import Optional, Dict, Any
import json

from src.core.websocket import websocket_manager, WebSocketMessage, MessageType, MarketData, TradeExecution
from src.core.security import verify_token
from src.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/trading")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
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
    client_id = None
    
    try:
        # Accept connection
        client_id = await websocket_manager.connect(websocket)
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Authentication (optional for public data)
        user_id = None
        if token:
            try:
                payload = verify_token(token)
                user_id = payload.get("sub")
                if user_id:
                    # Get user data (simplified)
                    user_data = {"user_id": user_id, "permissions": ["trade", "view_portfolio"]}
                    await websocket_manager.authenticate_connection(client_id, user_id, user_data)
            except Exception as e:
                logger.warning(f"WebSocket authentication failed for {client_id}: {e}")
                # Continue without authentication for public data
        
        # Send initial market data
        await _send_initial_data(client_id)
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                await websocket_manager.handle_message(client_id, data)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                # Send error message to client
                error_msg = WebSocketMessage(
                    MessageType.ERROR,
                    {"message": f"Error processing message: {str(e)}"}
                )
                await websocket_manager.connections[client_id].send_message(error_msg)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket connection error for {client_id}: {e}")
    finally:
        if client_id:
            await websocket_manager.disconnect(client_id)


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return websocket_manager.get_connection_stats()


@router.post("/broadcast/market-data")
async def broadcast_market_data(market_data: Dict[str, Any], token: str = Depends(security)):
    """
    Broadcast market data to WebSocket clients
    (Admin endpoint for testing/integration)
    """
    try:
        # Verify admin token
        payload = verify_token(token.credentials)
        # Add admin permission check here if needed
        
        # Create market data object
        data = MarketData(**market_data)
        await websocket_manager.broadcast_market_data(data)
        
        return {"status": "success", "message": "Market data broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error broadcasting market data: {str(e)}"
        )


@router.post("/broadcast/trade")
async def broadcast_trade_execution(trade_data: Dict[str, Any], token: str = Depends(security)):
    """
    Broadcast trade execution to WebSocket clients
    (Admin endpoint for testing/integration)
    """
    try:
        # Verify admin token
        payload = verify_token(token.credentials)
        
        # Create trade execution object
        trade = TradeExecution(**trade_data)
        await websocket_manager.broadcast_trade_execution(trade)
        
        return {"status": "success", "message": "Trade execution broadcasted"}
        
    except Exception as e:
        logger.error(f"Error broadcasting trade execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error broadcasting trade execution: {str(e)}"
        )


@router.post("/send/risk-alert")
async def send_risk_alert(
    user_id: str, 
    alert_data: Dict[str, Any], 
    token: str = Depends(security)
):
    """
    Send risk alert to specific user via WebSocket
    (Admin endpoint for testing/integration)
    """
    try:
        # Verify admin token
        payload = verify_token(token.credentials)
        
        # Send risk alert
        sent = await websocket_manager.send_risk_alert(user_id, alert_data)
        
        return {
            "status": "success" if sent else "user_not_connected",
            "message": f"Risk alert {'sent' if sent else 'queued'} for user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error sending risk alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error sending risk alert: {str(e)}"
        )


@router.post("/notify/user/{user_id}")
async def notify_user(
    user_id: str,
    notification: Dict[str, Any],
    token: str = Depends(security)
):
    """
    Send notification to specific user via WebSocket
    """
    try:
        # Verify token
        payload = verify_token(token.credentials)
        
        # Send notification
        message = WebSocketMessage(MessageType.SYSTEM_NOTIFICATION, notification)
        sent = await websocket_manager.send_to_user(user_id, message)
        
        return {
            "status": "success" if sent else "user_not_connected",
            "message": f"Notification {'sent' if sent else 'queued'} for user {user_id}"
        }
        
    except Exception as e:
        logger.error(f"Error sending user notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error sending notification: {str(e)}"
        )


async def _send_initial_data(client_id: str):
    """Send initial data to newly connected client"""
    try:
        connection = websocket_manager.connections[client_id]
        
        # Send available channels
        channels_msg = WebSocketMessage(
            MessageType.SYSTEM_NOTIFICATION,
            {
                "message": "Available channels",
                "channels": [
                    "market_data - General market updates",
                    "market_data.{SYMBOL} - Symbol-specific updates",
                    "trades.{SYMBOL} - Trade executions for symbol", 
                    "portfolio.{USER_ID} - Portfolio updates (auth required)",
                    "risk_alerts - Risk management alerts (auth required)",
                    "news - News and market alerts"
                ],
                "usage": {
                    "subscribe": {"type": "subscribe", "data": {"channel": "market_data"}},
                    "unsubscribe": {"type": "unsubscribe", "data": {"channel": "market_data"}},
                    "heartbeat": {"type": "heartbeat", "data": {}}
                }
            }
        )
        await connection.send_message(channels_msg)
        
        # Send cached market data if available
        if websocket_manager.market_data_cache:
            for symbol, market_data in list(websocket_manager.market_data_cache.items())[:5]:  # Limit to 5
                message = WebSocketMessage(MessageType.MARKET_DATA, market_data.__dict__)
                await connection.send_message(message)
        
    except Exception as e:
        logger.error(f"Error sending initial data to {client_id}: {e}")


# WebSocket client example (for documentation)
WEBSOCKET_CLIENT_EXAMPLE = """
// JavaScript WebSocket client example
const ws = new WebSocket('ws://localhost:8000/ws/trading?token=YOUR_JWT_TOKEN');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
    
    // Subscribe to market data
    ws.send(JSON.stringify({
        type: 'subscribe',
        data: { channel: 'market_data' }
    }));
    
    // Subscribe to specific symbol
    ws.send(JSON.stringify({
        type: 'subscribe', 
        data: { channel: 'market_data.BTC-USD' }
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
    
    switch(message.type) {
        case 'market_data':
            updateMarketDisplay(message.data);
            break;
        case 'trade_executed':
            showTradeNotification(message.data);
            break;
        case 'risk_alert':
            showRiskAlert(message.data);
            break;
        case 'error':
            console.error('WebSocket error:', message.data);
            break;
    }
};

ws.onclose = function(event) {
    console.log('WebSocket connection closed');
};

// Send heartbeat every 30 seconds
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'heartbeat',
            data: {}
        }));
    }
}, 30000);
""" 