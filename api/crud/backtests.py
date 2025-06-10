from sqlalchemy.orm import Session
from typing import Dict, Any

from database.models import BacktestResult, User

def create_backtest_result(
    db: Session, 
    user: User, 
    backtest_data: Dict[str, Any],
    request_params: Dict[str, Any]
) -> BacktestResult:
    """
    Create and save a new backtest result to the database.

    Args:
        db: The database session.
        user: The user who ran the backtest.
        backtest_data: The results from the backtest engine (equity curve, metrics, etc.).
        request_params: The original parameters sent to the backtest API.

    Returns:
        The newly created BacktestResult object.
    """
    
    metrics = backtest_data.get("metrics", {})
    
    db_backtest = BacktestResult(
        user_id=user.id,
        strategy_name=backtest_data.get("strategy_name", "Unknown"),
        parameters=request_params,
        
        # Metrics
        total_return=metrics.get("total_return"),
        annual_return=metrics.get("annual_return"),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        max_drawdown=metrics.get("max_drawdown"),
        win_rate=metrics.get("winning_days"),
        
        # Full data
        equity_curve=backtest_data.get("equity_curve"),
        drawdown_curve=backtest_data.get("drawdown")
    )
    
    db.add(db_backtest)
    db.commit()
    db.refresh(db_backtest)
    
    return db_backtest 