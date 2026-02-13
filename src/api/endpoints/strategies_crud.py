"""
Strategies CRUD API
Create and list user trading strategies (in-memory store; can be replaced with DB).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Strategies"])


class StrategyParameters(BaseModel):
    risk_budget: Optional[float] = None
    max_drawdown_limit: Optional[float] = None
    target_sharpe: Optional[float] = None
    rebalance_frequency: Optional[str] = None
    time_horizon: Optional[str] = None
    rsi_period: Optional[int] = None
    rsi_oversold: Optional[int] = None
    rsi_overbought: Optional[int] = None
    bb_period: Optional[int] = None
    bb_stddev: Optional[float] = None
    momentum_lookback: Optional[int] = None
    momentum_threshold: Optional[float] = None
    volatility_threshold: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_confidence: Optional[float] = None
    volatility_spread_threshold: Optional[float] = None
    threshold: Optional[float] = None


class StrategyCreate(BaseModel):
    name: str
    description: str
    strategy_type: str
    symbols: List[str]
    initial_capital: Optional[float] = None
    parameters: Optional[dict] = None
    is_active: bool = True


class StrategyResponse(BaseModel):
    id: int
    name: str
    description: str
    strategy_type: str
    is_active: bool
    created_at: str
    symbols: Optional[List[str]] = None
    initial_capital: Optional[float] = None
    parameters: Optional[dict] = None


# In-memory store (use DB in production)
_strategies_store: List[dict] = []
_next_id = 1


def _next_id_gen() -> int:
    global _next_id
    n = _next_id
    _next_id += 1
    return n


@router.get("/", response_model=List[StrategyResponse])
async def list_strategies():
    """List all strategies."""
    return _strategies_store


@router.post("/", response_model=StrategyResponse)
async def create_strategy(body: StrategyCreate):
    """Create a new strategy."""
    global _strategies_store
    sid = _next_id_gen()
    now = datetime.utcnow().isoformat() + "Z"
    strategy = {
        "id": sid,
        "name": body.name,
        "description": body.description,
        "strategy_type": body.strategy_type,
        "is_active": body.is_active,
        "created_at": now,
        "symbols": body.symbols,
        "initial_capital": body.initial_capital,
        "parameters": body.parameters or {},
    }
    _strategies_store.append(strategy)
    logger.info("Created strategy id=%s name=%s", sid, body.name)
    return strategy


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: int):
    """Get a strategy by id."""
    for s in _strategies_store:
        if s["id"] == strategy_id:
            return s
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(strategy_id: int, body: StrategyCreate):
    """Update a strategy."""
    for i, s in enumerate(_strategies_store):
        if s["id"] == strategy_id:
            _strategies_store[i] = {
                "id": strategy_id,
                "name": body.name,
                "description": body.description,
                "strategy_type": body.strategy_type,
                "is_active": body.is_active,
                "created_at": s["created_at"],
                "symbols": body.symbols,
                "initial_capital": body.initial_capital,
                "parameters": body.parameters or {},
            }
            return _strategies_store[i]
    raise HTTPException(status_code=404, detail="Strategy not found")


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: int):
    """Delete a strategy."""
    global _strategies_store
    for i, s in enumerate(_strategies_store):
        if s["id"] == strategy_id:
            _strategies_store.pop(i)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Strategy not found")
