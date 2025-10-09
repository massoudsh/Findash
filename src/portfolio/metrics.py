import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PortfolioMetricsCalculator:
    """
    A utility class to calculate various portfolio risk and performance metrics.
    """
    
    @staticmethod
    def calculate_all_metrics(
        returns: pd.DataFrame, 
        weights: np.ndarray,
        risk_free_rate: float = 0.02,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculates a comprehensive set of risk and performance metrics.
        """
        if returns.empty or len(weights) != len(returns.columns):
            logger.warning("Invalid input for metric calculation. Returning empty dict.")
            return {}

        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate metrics
        mean_return = portfolio_returns.mean()
        annualized_return = mean_return * 252
        volatility = portfolio_returns.std()
        annualized_volatility = volatility * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, alpha * 100)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        metrics = {
            "mean_return": mean_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "value_at_risk_95": var_95,
            "cvar_95": cvar_95,
        }
        
        # Round all metric values for cleaner output
        return {k: round(v, 4) for k, v in metrics.items()} 