"""
DEPRECATED: This file has been integrated into unified_websocket.py

This module is kept for backward compatibility only.
All functionality is now available in:
- src.api.endpoints.unified_websocket

Please update imports to use:
    from src.api.endpoints.unified_websocket import router as websocket_router
    
This file will be removed in a future version.

Unified Real-time API Endpoints
Provides WebSocket connections with pub/sub integration
"""

import json
import uuid
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from src.realtime.websockets import websocket_manager
from src.realtime.unified_pubsub import get_unified_pubsub, PubSubChannel
from src.realtime.websocket_bridge import get_websocket_bridge

logger = logging.getLogger(__name__)
router = APIRouter()

@router.on_event("startup")
async def startup_event():
    """Initialize unified pub/sub and WebSocket bridge on startup"""
    try:
        # Initialize unified pub/sub
        pubsub = await get_unified_pubsub()
        
        # Initialize WebSocket bridge
        bridge = await get_websocket_bridge()
        
        logger.info("✅ Real-time API initialized with unified pub/sub")
    except Exception as e:
        logger.error(f"❌ Failed to initialize real-time API: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        pubsub = await get_unified_pubsub()
        await pubsub.close()
        logger.info("🔌 Real-time API shut down")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(default=None)
):
    """
    Main WebSocket endpoint for real-time updates
    
    Supports:
    - Market data subscriptions
    - Portfolio updates
    - Trade notifications
    - AI agent results
    - System health updates
    """
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid.uuid4())
    
    await websocket_manager.connect(websocket, client_id)
    bridge = await get_websocket_bridge()
    
    logger.info(f"🔌 WebSocket client connected: {client_id}")
    
    try:
        # Send connection confirmation
        await websocket_manager.send_to_client(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "channels_available": [
                "market_data",
                "trades",
                "portfolio",
                "sentiment",
                "system_health"
            ]
        })
        
        # Main message loop
        while True:
            try:
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                
                msg_type = message.get("type")
                
                if msg_type == "subscribe":
                    channel = message.get("channel")
                    if channel:
                        await websocket_manager.subscribe_client(client_id, channel)
                        await bridge.handle_client_subscribe(client_id, channel)
                
                elif msg_type == "unsubscribe":
                    channel = message.get("channel")
                    if channel:
                        await websocket_manager.unsubscribe_client(client_id, channel)
                        await bridge.handle_client_unsubscribe(client_id, channel)
                
                elif msg_type == "ping":
                    await websocket_manager.send_to_client(client_id, {
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                
                elif msg_type == "get_subscriptions":
                    subscriptions = list(
                        websocket_manager.active_connections.get(
                            client_id, {}
                        ).subscriptions if client_id in websocket_manager.active_connections
                        else []
                    )
                    await websocket_manager.send_to_client(client_id, {
                        "type": "subscriptions",
                        "subscriptions": subscriptions
                    })
                
            except json.JSONDecodeError:
                await websocket_manager.send_to_client(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"❌ Error handling WebSocket message: {e}")
                await websocket_manager.send_to_client(client_id, {
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client disconnected: {client_id}")
    finally:
        await websocket_manager.disconnect(client_id)
        await bridge.handle_client_disconnect(client_id)

@router.websocket("/ws/market-data")
async def websocket_market_data_endpoint(websocket: WebSocket):
    """Dedicated endpoint for market data streaming"""
    client_id = f"market_data_{uuid.uuid4()}"
    await websocket_manager.connect(websocket, client_id)
    
    # Auto-subscribe to market data channels
    await websocket_manager.subscribe_client(client_id, PubSubChannel.MARKET_DATA.value)
    await websocket_manager.subscribe_client(client_id, PubSubChannel.PRICE_UPDATES.value)
    
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        pass
    finally:
        await websocket_manager.disconnect(client_id)

@router.websocket("/ws/portfolio/{portfolio_id}")
async def websocket_portfolio_endpoint(
    websocket: WebSocket,
    portfolio_id: str
):
    """Dedicated endpoint for portfolio updates"""
    client_id = f"portfolio_{portfolio_id}_{uuid.uuid4()}"
    await websocket_manager.connect(websocket, client_id)
    
    # Auto-subscribe to portfolio channels
    portfolio_channel = PubSubChannel.PORTFOLIO.value.format(portfolio_id=portfolio_id)
    trades_channel = PubSubChannel.TRADES_PORTFOLIO.value.format(portfolio_id=portfolio_id)
    
    await websocket_manager.subscribe_client(client_id, portfolio_channel)
    await websocket_manager.subscribe_client(client_id, trades_channel)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await websocket_manager.disconnect(client_id) 