"""
Trading Bots API Endpoints
Provides endpoints for creating, managing, and monitoring automated trading bots
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from src.core.security import get_current_active_user, get_optional_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading-bots", tags=["Trading Bots"])


# Request/Response Models (Phase 2: aligned with frontend Trading Bots UI)
class RiskConfig(BaseModel):
    max_position_pct: float = 5.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0


class TradingBotCreate(BaseModel):
    name: str
    strategy: str
    execution_mode: Optional[str] = Field(default="paper", alias="executionMode")  # "paper" | "live"
    symbols: Optional[List[str]] = None
    agent_sources: Optional[List[str]] = Field(default=None, alias="agentSources")
    risk: Optional[RiskConfig] = None
    symbol: Optional[str] = None  # legacy
    parameters: Optional[dict] = None

    class Config:
        populate_by_name = True


class TradingBotUpdate(BaseModel):
    name: Optional[str] = None
    execution_mode: Optional[str] = Field(default=None, alias="executionMode")
    symbols: Optional[List[str]] = None
    agent_sources: Optional[List[str]] = Field(default=None, alias="agentSources")
    risk: Optional[RiskConfig] = None
    parameters: Optional[dict] = None

    class Config:
        populate_by_name = True


class TradingBotResponse(BaseModel):
    id: str
    name: str
    strategy: str
    status: str
    execution_mode: Optional[str] = "paper"
    symbols: Optional[List[str]] = None
    agent_sources: Optional[List[str]] = None
    risk: Optional[dict] = None
    symbol: Optional[str] = None
    parameters: Optional[dict] = None
    performance: dict
    created_at: str
    updated_at: str
    last_signal_at: Optional[str] = None


class TradingBotPerformance(BaseModel):
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


# Mock data storage (in production, use database)
bots_db = {}


def _normalize_bot(bot: dict) -> dict:
    """Ensure Phase 2 fields exist for response compatibility."""
    out = dict(bot)
    out.setdefault("execution_mode", "paper")
    out.setdefault("symbols", [bot.get("symbol")] if bot.get("symbol") else [])
    out.setdefault("agent_sources", [])
    _r = RiskConfig()
    out.setdefault("risk", getattr(_r, "model_dump", _r.dict)())
    out.setdefault("last_signal_at", None)
    return out


def _user_id(current_user: Optional[TokenData]) -> str:
    return (current_user.user_id if current_user else "default")


@router.get("/", response_model=List[TradingBotResponse])
async def get_trading_bots(
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Get all trading bots for the current user (or default when unauthenticated)."""
    user_bots = [_normalize_bot(bot) for bot in bots_db.values() if bot.get("user_id") == _user_id(current_user)]
    return user_bots


@router.post("/", response_model=TradingBotResponse)
async def create_trading_bot(
    bot_data: TradingBotCreate,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Create a new trading bot (owned by current user or default when unauthenticated)."""
    bot_id = f"bot_{len(bots_db) + 1}"
    uid = _user_id(current_user)
    _risk = bot_data.risk or RiskConfig()
    risk = _risk.model_dump() if hasattr(_risk, "model_dump") else _risk.dict()
    symbols = bot_data.symbols or ([bot_data.symbol] if bot_data.symbol else [])
    new_bot = {
        "id": bot_id,
        "name": bot_data.name,
        "strategy": bot_data.strategy,
        "status": "stopped",
        "execution_mode": bot_data.execution_mode or "paper",
        "symbols": symbols,
        "agent_sources": bot_data.agent_sources or [],
        "risk": risk,
        "symbol": bot_data.symbol,
        "parameters": bot_data.parameters or {},
        "performance": {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0},
        "user_id": uid,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "last_signal_at": None,
    }
    
    bots_db[bot_id] = new_bot
    logger.info(f"Created trading bot {bot_id} for user {uid}")
    
    return new_bot


@router.patch("/{bot_id}", response_model=TradingBotResponse)
async def update_trading_bot(
    bot_id: str,
    payload: TradingBotUpdate,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Update bot config (Phase 2: execution_mode, symbols, agent_sources, risk)."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    if payload.name is not None:
        bot["name"] = payload.name
    if payload.execution_mode is not None:
        bot["execution_mode"] = payload.execution_mode
    if payload.symbols is not None:
        bot["symbols"] = payload.symbols
    if payload.agent_sources is not None:
        bot["agent_sources"] = payload.agent_sources
    if payload.risk is not None:
        _r = payload.risk
        bot["risk"] = _r.model_dump() if hasattr(_r, "model_dump") else _r.dict()
    if payload.parameters is not None:
        bot["parameters"] = payload.parameters
    bot["updated_at"] = datetime.now().isoformat()
    return _normalize_bot(bot)


@router.get("/{bot_id}", response_model=TradingBotResponse)
async def get_trading_bot(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Get a specific trading bot."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    return _normalize_bot(bot)


@router.post("/{bot_id}/start")
async def start_trading_bot(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Start a trading bot. Execution uses bot's execution_mode (paper/live) when execution layer is connected."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    bot["status"] = "active"
    bot["updated_at"] = datetime.now().isoformat()
    execution_mode = bot.get("execution_mode") or "paper"
    logger.info(f"Started trading bot {bot_id} (execution_mode={execution_mode})")
    return {"message": "Trading bot started", "bot_id": bot_id, "status": "active", "execution_mode": execution_mode}


@router.post("/{bot_id}/pause")
async def pause_trading_bot(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Pause a trading bot."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    bot["status"] = "paused"
    bot["updated_at"] = datetime.now().isoformat()
    logger.info(f"Paused trading bot {bot_id}")
    return {"message": "Trading bot paused", "bot_id": bot_id, "status": "paused"}


@router.post("/{bot_id}/stop")
async def stop_trading_bot(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Stop a trading bot."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    bot["status"] = "stopped"
    bot["updated_at"] = datetime.now().isoformat()
    logger.info(f"Stopped trading bot {bot_id}")
    return {"message": "Trading bot stopped", "bot_id": bot_id, "status": "stopped"}


@router.delete("/{bot_id}")
async def delete_trading_bot(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Delete a trading bot."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    del bots_db[bot_id]
    logger.info(f"Deleted trading bot {bot_id}")
    return {"message": "Trading bot deleted", "bot_id": bot_id}


@router.get("/{bot_id}/performance", response_model=TradingBotPerformance)
async def get_bot_performance(
    bot_id: str,
    current_user: Optional[TokenData] = Depends(get_optional_user),
):
    """Get performance metrics for a trading bot."""
    if bot_id not in bots_db:
        raise HTTPException(status_code=404, detail="Trading bot not found")
    bot = bots_db[bot_id]
    if bot.get("user_id") != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Access denied")
    return bot["performance"]

