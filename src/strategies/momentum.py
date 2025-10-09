import pandas as pd
import numpy as np
import logging
from typing import List

from .base import BaseStrategy
from src.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    A strategy that selects assets based on momentum and then optimizes
    the portfolio using a specified method (e.g., HRP).
    """
    def __init__(self, 
                 lookback_period: int = 60, 
                 top_percentile: float = 0.7,
                 optimization_method: str = 'HRP',
                 **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.top_percentile = top_percentile
        self.optimization_method = optimization_method

    def _calculate_momentum(self) -> pd.Series:
        """Calculates momentum for all assets."""
        return self.returns.rolling(window=self.lookback_period).mean()

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates portfolio weights based on momentum and optimization.
        This implementation generates a single set of weights for the entire period.
        A more advanced version would rebalance periodically.
        """
        logger.info("Generating signals for Momentum Strategy...")
        momentum_scores = self._calculate_momentum().iloc[-1] # Use latest scores
        
        # Select top assets based on momentum
        high_momentum_assets = momentum_scores[
            momentum_scores > momentum_scores.quantile(1 - self.top_percentile)
        ]
        
        if high_momentum_assets.empty:
            logger.warning("No assets met the momentum criteria.")
            # Return equal weights for all assets as a fallback
            num_assets = len(self.prices.columns)
            return pd.DataFrame(1/num_assets, index=self.returns.index, columns=self.prices.columns)

        selected_prices = self.prices[high_momentum_assets.index]
        
        # Optimize the selected portfolio
        optimizer = PortfolioOptimizer(prices_data=selected_prices)
        results = optimizer.run_optimization(method=self.optimization_method)
        
        # Create a full weight series, with zero weight for non-selected assets
        final_weights = pd.Series(0.0, index=self.prices.columns)
        final_weights.update(pd.Series(results['weights']))
        
        # Broadcast these weights across the entire backtest period
        weights_df = pd.DataFrame(final_weights, index=self.returns.index, columns=self.prices.columns)
        
        return weights_df 