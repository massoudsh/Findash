"""
WebSocket Endpoints for Real-time Updates
Agent status and wallet transaction updates
"""

import logging
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.realtime.websockets import WebSocketManager
from src.core.security import verify_token, TokenData
from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/ws", tags=["WebSocket Real-time"])
security = HTTPBearer()

# Use global websocket manager from main_refactored.py
# This will be injected at runtime
websocket_manager: Optional[WebSocketManager] = None

def set_websocket_manager(manager: WebSocketManager):
    """Set the websocket manager instance"""
    global websocket_manager
    websocket_manager = manager

async def get_user_from_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenData]:
    """Extract user from WebSocket token"""
    if not credentials:
        return None
    try:
        return verify_token(credentials.credentials)
    except Exception:
        return None

@router.websocket("/agents/status")
async def websocket_agent_status(websocket: WebSocket, token: Optional[str] = Query(None)):
    """WebSocket endpoint for real-time agent status updates"""
    if not websocket_manager:
        await websocket.close(code=1008, reason="WebSocket manager not initialized")
        return
    
    client_id = f"agent_status_{datetime.utcnow().timestamp()}"
    
    try:
        # Accept connection
        await websocket_manager.connect(websocket, client_id)
        
        # Subscribe to agent status channel
        await websocket_manager.subscribe(client_id, "agent_status")
        
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
        await websocket_manager.disconnect(client_id)

@router.websocket("/wallet/transactions")
async def websocket_wallet_transactions(websocket: WebSocket, token: Optional[str] = Query(None)):
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
        channel = f"wallet_transactions_{user.user_id}" if user else "wallet_transactions"
        await websocket_manager.subscribe(client_id, channel)
        
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
        await websocket_manager.disconnect(client_id)

# Helper function to broadcast agent status updates
async def broadcast_agent_status_update(agent_id: str, status_data: dict):
    """Broadcast agent status update to all subscribed clients"""
    if websocket_manager:
        await websocket_manager.broadcast("agent_status", {
            "type": "agent_status_update",
            "agent_id": agent_id,
            "data": status_data,
            "timestamp": datetime.utcnow().isoformat()
        })

# Helper function to broadcast wallet transaction updates
async def broadcast_wallet_transaction_update(user_id: int, transaction_data: dict):
    """Broadcast wallet transaction update to user's subscribed clients"""
    if websocket_manager:
        await websocket_manager.broadcast(f"wallet_transactions_{user_id}", {
            "type": "wallet_transaction_update",
            "user_id": user_id,
            "data": transaction_data,
            "timestamp": datetime.utcnow().isoformat()
        })

