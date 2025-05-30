import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Add the root directory to path to allow imports across modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from Risk Management module
from Modules.M6.Risk.Management.portfolio_metrics import PortfolioMetrics
from Modules.M6.Risk.Management.portfolio_optimization import PortfolioOptimizer

# Import from core modules
from core.logging_config import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)

class RiskAwareStrategy:
    """
    Base class for strategies that incorporate risk management.
    
    This class provides a framework for developing trading strategies
    that actively consider portfolio risk metrics in their decision-making.
    """
    
    def __init__(
        self,
        risk_budget: float = 0.05,        # Maximum allowed CVaR
        max_drawdown_limit: float = 0.15, # Maximum allowed drawdown
        target_sharpe: float = 1.0,       # Minimum target Sharpe ratio
        rebalance_frequency: str = 'M',   # 'D' for daily, 'W' for weekly, 'M' for monthly
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the risk-aware strategy.
        
        Args:
            risk_budget: Maximum risk allowed (CVaR)
            max_drawdown_limit: Maximum allowed drawdown
            target_sharpe: Target Sharpe ratio
            rebalance_frequency: How often to rebalance
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_budget = risk_budget
        self.max_drawdown_limit = max_drawdown_limit
        self.target_sharpe = target_sharpe
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        
        self.current_weights = None
        self.current_positions = {}
        self.portfolio_history = []
        self.last_rebalance_date = None
        
        # Initialize risk tools
        self.portfolio_metrics = PortfolioMetrics()
        self.optimizer = None  # Will be initialized when needed
        
    def load_data(
        self, 
        prices: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        additional_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> None:
        """
        Load market data for strategy execution.
        
        Args:
            prices: Historical price data
            factors: Factor data for risk models
            additional_data: Any additional data needed
        """
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.factors = factors
        self.additional_data = additional_data or {}
        
        logger.info(f"Loaded price data with shape {prices.shape}")
        if factors is not None:
            logger.info(f"Loaded factor data with shape {factors.shape}")
            
    def check_risk_compliance(self, weights: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if portfolio weights comply with risk limits.
        
        Args:
            weights: Portfolio weights to check
            
        Returns:
            Tuple of (is_compliant, risk_metrics)
        """
        # Calculate risk metrics
        risk_metrics = PortfolioMetrics.calculate_risk_metrics(
            self.returns, weights, self.risk_free_rate
        )
        
        # Check compliance with risk limits
        is_compliant = (
            risk_metrics['cvar'] <= self.risk_budget and
            risk_metrics['maximum_drawdown'] <= self.max_drawdown_limit and
            risk_metrics['sharpe'] >= self.target_sharpe
        )
        
        if not is_compliant:
            logger.warning(f"Portfolio not compliant with risk limits: CVaR={risk_metrics['cvar']:.4f}, "
                          f"MaxDD={risk_metrics['maximum_drawdown']:.4f}, Sharpe={risk_metrics['sharpe']:.4f}")
        
        return is_compliant, risk_metrics
        
    def optimize_portfolio(
        self,
        optimization_method: str = 'risk_budget',
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio weights based on risk parameters.
        
        Args:
            optimization_method: Method for optimization
            constraints: Additional constraints
            
        Returns:
            Optimized weights array
        """
        if self.optimizer is None:
            # Initialize the optimizer if it doesn't exist yet
            from Modules.M6.Risk.Management.portfolio_optimization import PortfolioOptimizer
            self.optimizer = PortfolioOptimizer()
        
        # Define default constraints if none provided
        if constraints is None:
            constraints = {
                'max_weight': 0.2,  # No asset can be more than 20% of portfolio
                'min_weight': 0.0,  # Allow assets to have zero weight
            }
        
        if optimization_method == 'risk_budget':
            # Optimize to meet risk budget
            custom_params = {
                'target': 'min_risk',
                'constraints': {
                    'cvar_limit': self.risk_budget,
                    'max_weight': constraints['max_weight'],
                    'min_weight': constraints['min_weight']
                }
            }
            
            # Run optimization
            results = self.optimizer.run_synthetic_optimization(
                self.returns, custom_parameters=custom_params
            )
            
            weights = results['weights']
            
        elif optimization_method == 'max_sharpe':
            # Optimize for maximum Sharpe ratio
            custom_params = {
                'target': 'max_sharpe',
                'constraints': {
                    'max_weight': constraints['max_weight'],
                    'min_weight': constraints['min_weight']
                }
            }
            
            # Run optimization
            results = self.optimizer.run_synthetic_optimization(
                self.returns, custom_parameters=custom_params
            )
            
            weights = results['weights']
            
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Validate risk compliance
        is_compliant, metrics = self.check_risk_compliance(weights)
        
        if not is_compliant and 'relax_constraints' not in constraints:
            logger.warning("Optimized portfolio does not meet risk requirements. "
                          "Consider relaxing constraints.")
        
        return weights
    
    def calculate_position_sizes(
        self, 
        capital: float,
        weights: np.ndarray,
        prices: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate target position sizes based on weights and capital.
        
        Args:
            capital: Available capital
            weights: Target portfolio weights
            prices: Current market prices
            
        Returns:
            Dictionary of position details by asset
        """
        latest_prices = prices.iloc[-1]
        
        positions = {}
        for i, asset in enumerate(self.returns.columns):
            if weights[i] > 0:
                # Calculate capital allocation
                allocation = capital * weights[i]
                
                # Calculate number of shares (rounded down)
                price = latest_prices[asset]
                shares = int(allocation / price)
                
                # Calculate actual allocation
                actual_allocation = shares * price
                
                positions[asset] = {
                    'weight': weights[i],
                    'target_allocation': allocation,
                    'actual_allocation': actual_allocation,
                    'shares': shares,
                    'price': price
                }
        
        return positions
    
    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """
        Determine if portfolio should be rebalanced based on schedule.
        
        Args:
            current_date: Current trading date
            
        Returns:
            Boolean indicating whether to rebalance
        """
        if self.last_rebalance_date is None:
            return True
        
        if self.rebalance_frequency == 'D':
            # Daily rebalancing
            return current_date.date() > self.last_rebalance_date.date()
        elif self.rebalance_frequency == 'W':
            # Weekly rebalancing
            return current_date.week != self.last_rebalance_date.week
        elif self.rebalance_frequency == 'M':
            # Monthly rebalancing
            return (current_date.year != self.last_rebalance_date.year or 
                    current_date.month != self.last_rebalance_date.month)
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")
    
    def execute_trades(
        self, 
        positions: Dict[str, Dict[str, float]],
        current_positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute trades to achieve target positions.
        
        This method would integrate with your trading module (M8).
        
        Args:
            positions: Target positions dictionary
            current_positions: Current positions dictionary
            
        Returns:
            Dictionary of trade results
        """
        # This is a placeholder that would integrate with your trading system
        # For example, calling an API from your paper trading module
        
        # Integration point with M8 | Paper Trading
        # from Modules.M8.Paper.Trading.execution import execute_order
        
        trades = {}
        for asset in set(list(positions.keys()) + list(current_positions.keys())):
            if asset in positions and asset in current_positions:
                # Asset is in both target and current - may need adjustment
                target_shares = positions[asset]['shares']
                current_shares = current_positions[asset]['shares']
                
                if target_shares != current_shares:
                    # Need to adjust position
                    adjustment = target_shares - current_shares
                    trades[asset] = {
                        'action': 'BUY' if adjustment > 0 else 'SELL',
                        'shares': abs(adjustment),
                        'target_shares': target_shares,
                        'current_shares': current_shares
                    }
            
            elif asset in positions:
                # New position to establish
                trades[asset] = {
                    'action': 'BUY',
                    'shares': positions[asset]['shares'],
                    'target_shares': positions[asset]['shares'],
                    'current_shares': 0
                }
                
            else:
                # Position to liquidate
                trades[asset] = {
                    'action': 'SELL',
                    'shares': current_positions[asset]['shares'],
                    'target_shares': 0,
                    'current_shares': current_positions[asset]['shares']
                }
        
        # In a real implementation, execute these trades via your trading API
        logger.info(f"Generated {len(trades)} trade orders")
        
        return trades
    
    def update_portfolio_metrics(
        self,
        positions: Dict[str, Dict[str, float]],
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Update portfolio metrics and track performance.
        
        Args:
            positions: Current positions dictionary
            current_date: Current trading date
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Extract weights from positions
        weights = np.zeros(len(self.returns.columns))
        total_value = sum(pos['actual_allocation'] for pos in positions.values())
        
        for i, asset in enumerate(self.returns.columns):
            if asset in positions:
                weights[i] = positions[asset]['actual_allocation'] / total_value
        
        # Calculate risk metrics
        metrics = PortfolioMetrics.calculate_risk_metrics(
            self.returns.loc[:current_date], weights, self.risk_free_rate
        )
        
        # Store in history
        self.portfolio_history.append({
            'date': current_date,
            'weights': weights,
            'positions': positions,
            'metrics': metrics
        })
        
        return metrics
    
    def backtest(
        self,
        initial_capital: float = 1000000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        optimization_method: str = 'risk_budget'
    ) -> pd.DataFrame:
        """
        Backtest the risk-aware strategy.
        
        Args:
            initial_capital: Starting capital
            start_date: Backtest start date
            end_date: Backtest end date
            optimization_method: Portfolio optimization method
            
        Returns:
            DataFrame of backtest results
        """
        # Set date range for backtest
        if start_date is None:
            start_date = self.prices.index[100]  # Allow for enough history
        if end_date is None:
            end_date = self.prices.index[-1]
            
        date_range = self.prices.loc[start_date:end_date].index
        
        # Initialize backtest state
        capital = initial_capital
        self.current_positions = {}
        self.portfolio_history = []
        self.last_rebalance_date = None
        
        # Run backtest
        for current_date in date_range:
            logger.debug(f"Processing date: {current_date}")
            
            # Check if we should rebalance
            if self.should_rebalance(current_date):
                logger.info(f"Rebalancing portfolio on {current_date}")
                
                # Optimize portfolio
                weights = self.optimize_portfolio(optimization_method)
                
                # Calculate position sizes
                positions = self.calculate_position_sizes(
                    capital, weights, self.prices.loc[:current_date]
                )
                
                # Execute trades
                trades = self.execute_trades(positions, self.current_positions)
                
                # Update current positions
                self.current_positions = positions
                self.last_rebalance_date = current_date
            
            # Update portfolio metrics
            metrics = self.update_portfolio_metrics(self.current_positions, current_date)
            
            # Update capital based on performance
            if len(self.portfolio_history) >= 2:
                prev_metrics = self.portfolio_history[-2]['metrics']
                daily_return = metrics['mean'] - prev_metrics['mean']
                capital *= (1 + daily_return)
        
        # Compile results
        results = pd.DataFrame([{
            'date': entry['date'],
            'portfolio_value': sum(pos['actual_allocation'] for pos in entry['positions'].values()),
            'sharpe': entry['metrics']['sharpe'],
            'volatility': entry['metrics']['volatility'],
            'cvar': entry['metrics']['cvar'],
            'max_drawdown': entry['metrics']['maximum_drawdown']
        } for entry in self.portfolio_history])
        
        return results
    
    def generate_report(self, output_format: str = 'html') -> Any:
        """
        Generate a comprehensive strategy report.
        
        Args:
            output_format: Format for the report ('html', 'pdf', or 'dataframe')
            
        Returns:
            Report in the specified format
        """
        if not self.portfolio_history:
            raise ValueError("No strategy history available. Run backtest first.")
            
        # Extract final weights and metrics
        final_entry = self.portfolio_history[-1]
        weights = final_entry['weights']
        metrics = final_entry['metrics']
        
        # Generate a risk report using the portfolio metrics module
        report = PortfolioMetrics.generate_risk_report(
            metrics,
            weights=weights,
            asset_names=self.returns.columns.tolist()
        )
        
        # Add performance metrics
        performance_data = pd.DataFrame([{
            'date': entry['date'],
            'portfolio_value': sum(pos['actual_allocation'] for pos in entry['positions'].values())
        } for entry in self.portfolio_history])
        
        # Calculate returns
        performance_data['return'] = performance_data['portfolio_value'].pct_change()
        
        # Calculate cumulative return
        initial_value = performance_data['portfolio_value'].iloc[0]
        final_value = performance_data['portfolio_value'].iloc[-1]
        cumulative_return = (final_value / initial_value) - 1
        
        # Add to report
        report = pd.concat([
            report,
            pd.DataFrame([{
                'Category': 'Performance Metrics',
                'Metric': 'Cumulative Return',
                'Value': f"{cumulative_return:.2%}"
            }, {
                'Category': 'Performance Metrics',
                'Metric': 'Trading Period',
                'Value': f"{performance_data['date'].iloc[0].strftime('%Y-%m-%d')} to {performance_data['date'].iloc[-1].strftime('%Y-%m-%d')}"
            }])
        ])
        
        # Return in requested format
        if output_format == 'dataframe':
            return report
        elif output_format == 'html':
            return report.to_html()
        elif output_format == 'pdf':
            # Implementation would depend on your PDF generation library
            return report
        else:
            raise ValueError(f"Unknown output format: {output_format}")


class MomentumRiskAwareStrategy(RiskAwareStrategy):
    """
    A momentum strategy that incorporates risk management.
    
    This is an example of extending the base RiskAwareStrategy.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,  # Trading days for momentum calculation
        momentum_percentile: float = 0.7,  # Select top 70% momentum assets
        **kwargs
    ):
        """
        Initialize the momentum risk-aware strategy.
        
        Args:
            lookback_period: Days to look back for momentum
            momentum_percentile: Percentile cutoff for selection
            **kwargs: Additional arguments for RiskAwareStrategy
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.momentum_percentile = momentum_percentile
        
    def calculate_momentum_scores(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for assets.
        
        Args:
            prices: Historical price data
            
        Returns:
            Series of momentum scores by asset
        """
        # Calculate returns over lookback period
        returns = prices.pct_change(self.lookback_period)
        
        # Get latest momentum scores
        latest_returns = returns.iloc[-1]
        
        return latest_returns
    
    def select_momentum_assets(self, prices: pd.DataFrame) -> List[str]:
        """
        Select assets based on momentum.
        
        Args:
            prices: Historical price data
            
        Returns:
            List of selected assets
        """
        # Calculate momentum scores
        momentum_scores = self.calculate_momentum_scores(prices)
        
        # Sort by momentum
        sorted_assets = momentum_scores.sort_values(ascending=False)
        
        # Select top percentile
        cutoff_index = int(len(sorted_assets) * self.momentum_percentile)
        selected_assets = sorted_assets.index[:cutoff_index]
        
        return list(selected_assets)
    
    def optimize_portfolio(
        self,
        optimization_method: str = 'risk_budget',
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Optimize portfolio with momentum filter.
        
        Args:
            optimization_method: Method for optimization
            constraints: Additional constraints
            
        Returns:
            Optimized weights array
        """
        # Select assets based on momentum
        momentum_assets = self.select_momentum_assets(self.prices)
        
        # Create a filtered returns DataFrame
        momentum_returns = self.returns[momentum_assets]
        
        # Set up constraints to only include momentum assets
        if constraints is None:
            constraints = {}
            
        # Create a mask for asset selection
        asset_mask = {asset: asset in momentum_assets for asset in self.returns.columns}
        constraints['asset_mask'] = asset_mask
        
        # Call the parent method with modified constraints
        return super().optimize_portfolio(optimization_method, constraints)


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Load sample data
    from skfolio.datasets import load_sp500_dataset
    prices = load_sp500_dataset()
    
    # Create strategy
    strategy = MomentumRiskAwareStrategy(
        risk_budget=0.05,
        max_drawdown_limit=0.15,
        target_sharpe=1.0,
        rebalance_frequency='M',
        lookback_period=60,
        momentum_percentile=0.7
    )
    
    # Load data
    strategy.load_data(prices)
    
    # Run backtest
    results = strategy.backtest(
        initial_capital=1000000,
        start_date='2019-01-01',
        end_date='2021-12-31'
    )
    
    # Generate report
    report = strategy.generate_report()
    
    # Print summary
    print("Backtest Results Summary:")
    print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Sharpe Ratio: {results['sharpe'].iloc[-1]:.4f}")
    print(f"Max Drawdown: {results['max_drawdown'].iloc[-1]:.2%}")
    print(f"CVaR: {results['cvar'].iloc[-1]:.2%}") 