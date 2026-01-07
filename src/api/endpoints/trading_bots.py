"""
Trading Bots API Endpoints
Provides endpoints for creating, managing, and monitoring automated trading bots
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from src.core.security import get_current_active_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading-bots", tags=["Trading Bots"])


# Request/Response Models
class TradingBotCreate(BaseModel):
    name: str
    strategy: str
    symbol: Optional[str] = None
    parameters: Optional[dict] = None


class TradingBotResponse(BaseModel):
    id: str
    name: str
    strategy: str
    status: str
    symbol: Optional[str] = None
    parameters: Optional[dict] = None
    performance: dict
    created_at: str
    updated_at: str


class TradingBotPerformance(BaseModel):
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


# Mock data storage (in production, use database)
bots_db = {}


@router.get("/", response_model=List[TradingBotResponse])
async def get_trading_bots(
    # current_user: TokenData = Depends(get_current_active_user)  # Temporarily disabled for testing
):
    """Get all trading bots for the current user"""
    user_bots = [bot for bot in bots_db.values() if bot.get("user_id") == current_user.user_id]
    return user_bots


@router.post("/", response_model=TradingBotResponse)
async def create_trading_bot(
    bot_data: TradingBotCreate,
    # current_user: TokenData = Depends(get_current_active_user)  # Temporarily disabled for testing
):
    """Create a new trading bot"""
    bot_id = f"bot_{len(bots_db) + 1}"
    
    new_bot = {
        "id": bot_id,
        "name": bot_data.name,
        "strategy": bot_data.strategy,
        "status": "paused",
        "symbol": bot_data.symbol,
        "parameters": bot_data.parameters or {},
        "performance": {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
        },
        "user_id": "default",  # current_user.user_id if current_user else "default",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    
    bots_db[bot_id] = new_bot
    logger.info(f"Created trading bot {bot_id} for user {current_user.user_id}")
    
    return new_bot


@router.get("/{bot_id}", response_model=TradingBotResponse)
async def get_trading_bot(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get a specific trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return bot


@router.post("/{bot_id}/start")
async def start_trading_bot(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Start a trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    bot["status"] = "active"
    bot["updated_at"] = datetime.now().isoformat()
    logger.info(f"Started trading bot {bot_id}")
    
    return {"message": "Trading bot started", "bot_id": bot_id, "status": "active"}


@router.post("/{bot_id}/pause")
async def pause_trading_bot(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Pause a trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    bot["status"] = "paused"
    bot["updated_at"] = datetime.now().isoformat()
    logger.info(f"Paused trading bot {bot_id}")
    
    return {"message": "Trading bot paused", "bot_id": bot_id, "status": "paused"}


@router.post("/{bot_id}/stop")
async def stop_trading_bot(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Stop a trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    bot["status"] = "stopped"
    bot["updated_at"] = datetime.now().isoformat()
    logger.info(f"Stopped trading bot {bot_id}")
    
    return {"message": "Trading bot stopped", "bot_id": bot_id, "status": "stopped"}


@router.delete("/{bot_id}")
async def delete_trading_bot(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Delete a trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del bots_db[bot_id]
    logger.info(f"Deleted trading bot {bot_id}")
    
    return {"message": "Trading bot deleted", "bot_id": bot_id}


@router.get("/{bot_id}/performance", response_model=TradingBotPerformance)
async def get_bot_performance(
    bot_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get performance metrics for a trading bot"""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    
    bot = bots_db[bot_id]
    if bot.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return bot["performance"]

