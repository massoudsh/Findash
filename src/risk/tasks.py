import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

from src.core.celery_app import celery_app
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher
from src.risk.risk_manager import calculate_var_helper, calculate_sharpe_ratio_helper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Schemas ---
@dataclass
class PositionRiskInfo:
    symbol: str
    position_size: float
    entry_price: float
    var_95: float

# --- Risk Calculation Logic ---
# Use helper functions from unified RiskManager
calculate_var = calculate_var_helper
calculate_sharpe_ratio = calculate_sharpe_ratio_helper

# --- Celery Task ---
@celery_app.task(name="risk.evaluate_trade_risk")
def evaluate_trade_risk(symbol: str, entry_price: float, stop_loss: float, take_profit: float) -> Dict:
    """
    Evaluates the risk of a potential trade.
    """
    logger.info(f"Evaluating trade risk for {symbol}")
    try:
        data_fetcher = TimeSeriesDataFetcher()
        # Fetch 1 year of data for risk calculation
        data = data_fetcher.fetch_data_for_symbol(symbol, days=365)
        if data.empty:
            return {"status": "error", "message": "Insufficient data for risk evaluation."}
        
        returns = data['Close'].pct_change().dropna()

        if (entry_price - stop_loss) <= 0:
            risk_reward_ratio = float('inf')
        else:
            risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)

        metrics = {
            'risk_reward_ratio': risk_reward_ratio,
            'volatility': returns.std() * np.sqrt(252),
            'var_95': calculate_var(returns),
            'sharpe_ratio': calculate_sharpe_ratio(returns)
        }
        
        return {"status": "success", "symbol": symbol, "metrics": metrics}

    except Exception as e:
        logger.error(f"Error during trade risk evaluation for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)} 