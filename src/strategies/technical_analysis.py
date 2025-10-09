import pandas as pd
import ta
import numpy as np
import logging
from typing import Dict, List, Any
from .base import BaseStrategy
from .risk_aware import RiskAwareStrategy

logger = logging.getLogger(__name__)

class RsiStrategy(BaseStrategy):
    """
    A momentum strategy based on the Relative Strength Index (RSI).
    Generates a BUY signal when RSI is below a threshold (oversold) and
    a SELL signal when it is above a threshold (overbought).
    """
    def __init__(self, rsi_period: int = 14, oversold_threshold: int = 30, overbought_threshold: int = 70):
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates RSI and generates BUY/SELL signals.
        Args:
            data (pd.DataFrame): DataFrame with a 'close' column.
        Returns:
            pd.DataFrame: Original DataFrame with a new 'signal' column.
        """
        if data is None:
            data = self.prices
        
        df = data.copy()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        
        df['signal'] = 'HOLD'
        df.loc[df['rsi'] < self.oversold_threshold, 'signal'] = 'BUY'
        df.loc[df['rsi'] > self.overbought_threshold, 'signal'] = 'SELL'
        
        return df

    def get_parameters(self) -> Dict:
        return {
            "rsi_period": self.rsi_period,
            "oversold_threshold": self.oversold_threshold,
            "overbought_threshold": self.overbought_threshold
        }

class MeanReversionStrategy(BaseStrategy):
    """
    A mean-reversion strategy based on Bollinger Bands.
    Generates a BUY signal when the price crosses below the lower band and
    a SELL signal when it crosses above the upper band.
    """
    def __init__(self, bbands_period: int = 20, bbands_stddev: int = 2):
        super().__init__()
        self.bbands_period = bbands_period
        self.bbands_stddev = bbands_stddev

    def generate_signals(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates Bollinger Bands and generates BUY/SELL signals.
        Args:
            data (pd.DataFrame): DataFrame with a 'close' column.
        Returns:
            pd.DataFrame: Original DataFrame with a new 'signal' column.
        """
        if data is None:
            data = self.prices
            
        df = data.copy()
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=self.bbands_period, window_dev=self.bbands_stddev)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        df['signal'] = 'HOLD'
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 'BUY'
        df.loc[df['close'] > df['bb_upper'], 'signal'] = 'SELL'

        return df

    def get_parameters(self) -> Dict:
        return {
            "bbands_period": self.bbands_period,
            "bbands_stddev": self.bbands_stddev
        }

class TechnicalAnalysisStrategy(BaseStrategy):
    """
    Combined technical analysis strategy using multiple indicators
    """
    def __init__(self, rsi_period: int = 14, bb_period: int = 20, bb_stddev: int = 2):
        super().__init__()
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_stddev = bb_stddev
    
    def generate_signals(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate signals using RSI and Bollinger Bands"""
        if data is None:
            data = self.prices
            
        df = data.copy()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=self.bb_period, window_dev=self.bb_stddev)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Combined signals
        df['signal'] = 'HOLD'
        # Strong buy: RSI oversold and price below lower BB
        df.loc[(df['rsi'] < 30) & (df['close'] < df['bb_lower']), 'signal'] = 'BUY'
        # Strong sell: RSI overbought and price above upper BB
        df.loc[(df['rsi'] > 70) & (df['close'] > df['bb_upper']), 'signal'] = 'SELL'
        
        return df
    
    def get_parameters(self) -> Dict:
        return {
            "rsi_period": self.rsi_period,
            "bb_period": self.bb_period,
            "bb_stddev": self.bb_stddev
        }

class MomentumRiskAwareStrategy(RiskAwareStrategy):
    """
    A risk-aware strategy that first selects a universe of assets based on
    momentum, and then optimizes the portfolio of those selected assets
    according to the risk constraints defined in the base class.
    """
    def __init__(
        self,
        lookback_period: int = 90,
        momentum_percentile: float = 0.75,
        **kwargs
    ):
        """
        Args:
            lookback_period (int): The number of days to look back for momentum calculation.
            momentum_percentile (float): The percentile of top-performing assets to select.
            **kwargs: Arguments to be passed to the RiskAwareStrategy base class.
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.momentum_percentile = momentum_percentile
        self.selected_assets: List[str] = []

    def _calculate_momentum(self) -> pd.Series:
        """Calculates the momentum for each asset over the lookback period."""
        if len(self.prices) < self.lookback_period:
            return pd.Series(dtype=float)
        
        # Using ROC (Rate of Change) as the momentum indicator
        momentum = self.prices.pct_change(self.lookback_period).iloc[-1]
        return momentum.dropna()

    def generate_signals(self, current_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Generates target portfolio weights by selecting for momentum and optimizing for risk.
        """
        if not self.should_rebalance(current_date):
            return {"status": "HOLD"}

        logger.info(f"Rebalancing started for {current_date.date()}...")

        # 1. Calculate momentum and select asset universe
        momentum_scores = self._calculate_momentum()
        if momentum_scores.empty:
            return {"status": "INSUFFICIENT_DATA"}
        
        percentile_threshold = momentum_scores.quantile(self.momentum_percentile)
        self.selected_assets = momentum_scores[momentum_scores >= percentile_threshold].index.tolist()
        
        if not self.selected_assets:
            return {"status": "NO_ASSETS_SELECTED"}
            
        logger.info(f"Selected {len(self.selected_assets)} assets based on momentum.")
        
        # 2. Optimize portfolio for the selected assets
        returns_subset = self.returns[self.selected_assets]
        
        # We need a reference to the optimizer from the base class.
        # This uses the placeholder for now.
        optimized_results = self.optimizer.run_synthetic_optimization(
            returns_subset, 
            custom_parameters={'target': 'max_sharpe'} # or 'risk_budget'
        )
        
        # Create a full weight vector including zero weights for non-selected assets
        final_weights = pd.Series(0.0, index=self.returns.columns)
        final_weights[self.selected_assets] = optimized_results['weights']
        
        self.last_rebalance_date = current_date

        return {
            "status": "REBALANCE",
            "weights": final_weights.to_dict(),
            "selected_assets": self.selected_assets
        } 