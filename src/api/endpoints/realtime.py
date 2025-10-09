from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.realtime.websockets import manager
from src.realtime.redis_pubsub import redis_pubsub
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# The channel we will listen to for dashboard updates
DASHBOARD_CHANNEL = "dashboard_updates"

@router.lifespan("startup")
async def startup_event():
    """On startup, subscribe to the Redis channel and start the listener."""
    await redis_pubsub.subscribe(DASHBOARD_CHANNEL)
    asyncio.create_task(redis_pubsub.listen(manager.broadcast))
    logger.info("Realtime dashboard router started up.")

@router.lifespan("shutdown")
async def shutdown_event():
    """On shutdown, close the Redis connection."""
    await redis_pubsub.close()
    logger.info("Realtime dashboard router shut down.")

@router.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """
    The main WebSocket endpoint for the live dashboard.
    Manages the connection and lifecycle of a client.
    """
    await manager.connect(websocket)
    logger.info(f"Client connected: {websocket.client.host}")
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client disconnected: {websocket.client.host}") 