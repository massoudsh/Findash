"""
Real-time Processing API Routes for FastAPI service
Handles real-time market data streams, alerts, and live updates
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import json

from src.core.cache import TradingCache
from src.realtime.websockets import WebSocketManager
from src.database.postgres_connection import get_db
from src.database.crud import create_alert_rule, get_alert_rules_by_user, get_alert_rule, update_alert_rule, delete_alert_rule
from sqlalchemy.orm import Session

# Create websocket manager instance for realtime routes
websocket_manager = WebSocketManager()

router = APIRouter()

# Pydantic models
class RealTimeDataResponse(BaseModel):
    timestamp: str
    symbol: str
    price: float
    volume: int
    change: float
    change_percent: float
    bid: Optional[float] = None
    ask: Optional[float] = None

class AlertRule(BaseModel):
    symbol: str
    condition: str  # "above", "below", "percent_change"
    threshold: float
    is_active: bool = True

class AlertResponse(BaseModel):
    id: str
    symbol: str
    message: str
    triggered_at: str
    price: float

class AlertRuleCreate(BaseModel):
    name: str
    description: str = ''
    category: str
    metric: str
    operator: str
    threshold: float
    duration: str
    notifications: dict
    severity: str = 'medium'
    enabled: bool = True

class AlertRuleOut(BaseModel):
    id: int
    user_id: int
    name: str
    description: str
    category: str
    metric: str
    operator: str
    threshold: float
    duration: str
    notifications: dict
    severity: str
    enabled: bool
    created_at: str
    updated_at: str

    class Config:
        orm_mode = True

@router.get("/stream", response_model=Dict[str, Any])
async def get_realtime_stream_info():
    """Get information about available real-time data streams"""
    return {
        "status": "active",
        "active_streams": ["market_data", "news", "sentiment"],
        "connected_clients": len(websocket_manager.active_connections) if hasattr(websocket_manager, 'active_connections') else 0,
        "supported_symbols": ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "BTC-USD", "ETH-USD"],
        "update_frequency": "real-time",
        "websocket_endpoint": "/ws/{client_id}"
    }

@router.get("/current/{symbol}", response_model=RealTimeDataResponse)
async def get_current_price(symbol: str):
    """Get current real-time price for a symbol"""
    try:
        # In production, this would fetch from real-time data provider
        # For now, simulate real-time data
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', current_price)
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close > 0 else 0
        
        return RealTimeDataResponse(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            price=current_price,
            volume=info.get('volume', 0),
            change=change,
            change_percent=round(change_percent, 2),
            bid=info.get('bid'),
            ask=info.get('ask')
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or data unavailable")

@router.get("/batch", response_model=List[RealTimeDataResponse])
async def get_batch_prices(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Get real-time prices for multiple symbols"""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    if len(symbol_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed per request")
    
    results = []
    for symbol in symbol_list:
        try:
            # Get current price for each symbol
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice', 0)
            previous_close = info.get('previousClose', current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close > 0 else 0
            
            results.append(RealTimeDataResponse(
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                price=current_price,
                volume=info.get('volume', 0),
                change=change,
                change_percent=round(change_percent, 2),
                bid=info.get('bid'),
                ask=info.get('ask')
            ))
        except Exception:
            # Skip symbols that fail
            continue
    
    return results

@router.post("/alerts", response_model=AlertRuleOut)
async def create_alert(alert: AlertRuleCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new alert rule for a user"""
    rule = create_alert_rule(db, user_id, alert.dict())
    return rule

@router.get("/alerts", response_model=List[AlertRuleOut])
async def list_alerts(user_id: int, db: Session = Depends(get_db)):
    """List all alert rules for a user"""
    return get_alert_rules_by_user(db, user_id)

@router.put("/alerts/{alert_id}", response_model=AlertRuleOut)
async def update_alert(alert_id: int, alert: AlertRuleCreate, db: Session = Depends(get_db)):
    """Update an alert rule"""
    rule = update_alert_rule(db, alert_id, alert.dict())
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return rule

@router.delete("/alerts/{alert_id}", response_model=dict)
async def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    """Delete an alert rule"""
    success = delete_alert_rule(db, alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"status": "deleted"}

@router.get("/market-status", response_model=Dict[str, Any])
async def get_market_status():
    """Get current market status"""
    return {
        "us_market": {
            "status": "open",  # open, closed, pre_market, after_hours
            "next_open": "2024-12-02T09:30:00Z",
            "next_close": "2024-12-02T16:00:00Z"
        },
        "crypto_market": {
            "status": "open",  # Always open
            "24h_active": True
        },
        "server_time": datetime.utcnow().isoformat(),
        "timezone": "UTC"
    }

@router.post("/broadcast", response_model=Dict[str, str])
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all connected WebSocket clients"""
    try:
        await websocket_manager.broadcast_to_all(message)
        return {
            "status": "success",
            "message": "Broadcast sent to all connected clients",
            "client_count": len(websocket_manager.active_connections) if hasattr(websocket_manager, 'active_connections') else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def realtime_health():
    """Health check for real-time processing service"""
    return {
        "status": "healthy",
        "service": "realtime",
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": len(websocket_manager.active_connections) if hasattr(websocket_manager, 'active_connections') else 0,
        "data_sources": ["yfinance", "websocket"],
        "uptime": "active"
    } 