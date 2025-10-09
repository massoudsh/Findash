import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    A vectorized backtester to evaluate the performance of trading signals.
    """
    def __init__(self, 
                 time_series_data: pd.DataFrame, 
                 signals: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        self.data = time_series_data
        self.signals = signals
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results: pd.DataFrame = None

    def run(self):
        """
        Runs the backtest.
        """
        if self.signals.empty:
            logger.warning("Signals are empty. Cannot run backtest.")
            return

        # Align data and signals
        self.data, self.signals = self.data.align(self.signals, join='inner', axis=0)
        
        # Calculate positions based on signals (1 = long, -1 = short, 0 = neutral)
        positions = self.signals['signal'].shift(1).fillna(0) # Shift to avoid lookahead bias

        # Calculate daily market returns
        market_returns = self.data['Close'].pct_change()

        # Calculate strategy returns
        strategy_returns = market_returns * positions

        # Calculate portfolio value
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['market_returns'] = market_returns
        portfolio['strategy_returns'] = strategy_returns
        portfolio['cumulative_market_returns'] = (1 + market_returns).cumprod()
        portfolio['cumulative_strategy_returns'] = (1 + strategy_returns).cumprod()
        
        self.results = portfolio
        logger.info("Backtest run completed.")

    def get_performance_summary(self) -> dict:
        """
        Returns a summary of the backtest performance.
        """
        if self.results is None:
            raise ValueError("Backtest has not been run yet. Call .run() first.")

        total_return = self.results['cumulative_strategy_returns'].iloc[-1] - 1
        annualized_return = total_return * (252 / len(self.results))
        annualized_volatility = self.results['strategy_returns'].std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0

        return {
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "annualized_volatility_pct": annualized_volatility * 100,
            "sharpe_ratio": sharpe_ratio,
        } 