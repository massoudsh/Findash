import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.metrics import PortfolioMetricsCalculator

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    def __init__(self, risk_free_rate: float = 0.02, **kwargs):
        self.risk_free_rate = risk_free_rate
        self.prices: pd.DataFrame = None
        self.returns: pd.DataFrame = None

    def load_data(self, prices: pd.DataFrame):
        self.prices = prices
        self.returns = self.prices.pct_change().dropna()
        logger.info(f"Data loaded for strategy. Shape: {self.prices.shape}")

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on the strategy's logic.
        Should return a Series of weights.
        """
        pass

    def backtest(self, initial_capital: float = 100000.0) -> Dict[str, Any]:
        logger.info(f"Starting backtest for {self.__class__.__name__}...")
        
        weights = self.generate_signals()
        
        # Align returns with weights
        aligned_returns = self.returns.loc[weights.index]
        
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate full portfolio metrics
        metrics = PortfolioMetricsCalculator.calculate_all_metrics(
            returns=aligned_returns,
            weights=weights.values, # This is an approximation; weights change
            risk_free_rate=self.risk_free_rate
        )
        
        logger.info(f"Backtest complete. Final portfolio value: ${cumulative_returns.iloc[-1] * initial_capital:.2f}")

        return {
            "final_capital": cumulative_returns.iloc[-1] * initial_capital,
            "cumulative_returns": cumulative_returns.to_dict(),
            "metrics": metrics
        }

    def get_parameters(self) -> Dict:
        """
        Returns the parameters of the strategy.
        This can be overridden by subclasses to expose their specific parameters.
        """
        return {} 