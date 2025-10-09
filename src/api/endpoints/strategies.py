from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from src.strategies.tasks import run_strategy_backtest

router = APIRouter()

class BacktestRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    initial_capital: float = 100000.0
    strategy_params: Dict[str, Any] = None

@router.post("/run_backtest", status_code=202)
def trigger_backtest(request: BacktestRequest):
    """
    Triggers an asynchronous backtest for a given trading strategy.
    """
    task = run_strategy_backtest.delay(
        strategy_name=request.strategy_name,
        symbols=request.symbols,
        initial_capital=request.initial_capital,
        strategy_params=request.strategy_params
    )
    return {"message": "Strategy backtest started.", "task_id": task.id} 