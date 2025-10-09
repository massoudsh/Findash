import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

from src.core.celery_app import celery_app
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher

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
def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

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