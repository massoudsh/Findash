"""
Backtesting API Endpoints
Provides endpoints for running strategy backtests and analyzing historical performance
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, date
import logging

from src.core.security import get_current_active_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtesting", tags=["Backtesting"])


# Request/Response Models
class BacktestRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: Optional[float] = 100000.0
    parameters: Optional[dict] = None


class BacktestResult(BaseModel):
    backtest_id: str
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float
    annualized_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: str
    profit_factor: Optional[float] = None
    created_at: str


class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: str


# Mock storage for backtest results
backtests_db = {}


@router.post("/run", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Run a backtest for a given strategy and symbol"""
    backtest_id = f"backtest_{len(backtests_db) + 1}"
    
    # Simulate backtest execution (in production, this would run actual backtesting logic)
    # For now, return mock results
    result = {
        "backtest_id": backtest_id,
        "strategy": request.strategy,
        "symbol": request.symbol,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "total_return": 15.5,  # Mock value
        "annualized_return": 18.2,
        "sharpe_ratio": 1.85,
        "max_drawdown": -8.2,
        "win_rate": 0.62,
        "total_trades": 145,
        "avg_trade_duration": "2.5 days",
        "profit_factor": 1.45,
        "created_at": datetime.now().isoformat(),
    }
    
    backtests_db[backtest_id] = {
        **result,
        "user_id": current_user.user_id,
        "initial_capital": request.initial_capital,
        "parameters": request.parameters or {},
    }
    
    logger.info(f"Ran backtest {backtest_id} for user {current_user.user_id}")
    
    return result


@router.get("/results", response_model=list[BacktestResult])
async def get_backtest_results(
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get all backtest results for the current user"""
    user_backtests = [
        {k: v for k, v in bt.items() if k != "user_id" and k != "initial_capital" and k != "parameters"}
        for bt in backtests_db.values()
        if bt.get("user_id") == current_user.user_id
    ]
    
    # Apply filters
    if strategy:
        user_backtests = [bt for bt in user_backtests if bt.get("strategy") == strategy]
    if symbol:
        user_backtests = [bt for bt in user_backtests if bt.get("symbol") == symbol]
    
    return user_backtests


@router.get("/results/{backtest_id}", response_model=BacktestResult)
async def get_backtest_result(
    backtest_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get a specific backtest result"""
    if backtest_id not in backtests_db:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    
    backtest = backtests_db[backtest_id]
    if backtest.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    result = {k: v for k, v in backtest.items() if k != "user_id" and k != "initial_capital" and k != "parameters"}
    return result


@router.delete("/results/{backtest_id}")
async def delete_backtest_result(
    backtest_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Delete a backtest result"""
    if backtest_id not in backtests_db:
        raise HTTPException(status_code=404, detail="Backtest result not found")
    
    backtest = backtests_db[backtest_id]
    if backtest.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    del backtests_db[backtest_id]
    logger.info(f"Deleted backtest {backtest_id}")
    
    return {"message": "Backtest result deleted", "backtest_id": backtest_id}


@router.get("/strategies")
async def get_available_strategies():
    """Get list of available backtesting strategies"""
    return {
        "strategies": [
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Trades based on price momentum indicators",
            },
            {
                "id": "mean_reversion",
                "name": "Mean Reversion",
                "description": "Trades when price deviates from mean",
            },
            {
                "id": "arbitrage",
                "name": "Arbitrage",
                "description": "Exploits price differences across markets",
            },
            {
                "id": "scalping",
                "name": "Scalping",
                "description": "High-frequency short-term trading",
            },
        ]
    }

