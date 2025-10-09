import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
import cvxpy as cp
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using cvxpy and scipy.
    """
    
    def __init__(self, 
                 prices_data: pd.DataFrame,
                 risk_free_rate: float = 0.02,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        self.prices = prices_data
        self.returns = self._prices_to_returns(self.prices)
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.assets = self.returns.columns.tolist()
        
    def _prices_to_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert prices to returns"""
        return prices.pct_change().dropna()
        
    def run_optimization(self, 
                         method: str = 'HRP', 
                         objective: str = 'sharpe_ratio') -> Dict:
        """
        Runs the specified portfolio optimization method.
        """
        logger.info(f"Running optimization with method: {method} and objective: {objective}")
        
        if method == 'HRP':
            return self._optimize_hrp()
        elif method == 'Mean-Variance':
            return self._optimize_mean_variance(objective)
        elif method == 'Equal-Weight':
            return self._optimize_equal_weight()
        else:
            raise ValueError(f"Optimization method '{method}' not supported.")

    def _optimize_hrp(self) -> Dict:
        """Performs Hierarchical Risk Parity optimization (simplified version)."""
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # For simplicity, use inverse volatility weighting as HRP proxy
        volatilities = self.returns.std()
        inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
        
        weights = inv_vol_weights.values
        
        return self._format_results(weights)
        
    def _optimize_mean_variance(self, objective_str: str) -> Dict:
        """Performs Mean-Variance optimization using cvxpy."""
        n_assets = len(self.assets)
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.min_weight,
            weights <= self.max_weight
        ]
        
        if objective_str == 'sharpe_ratio':
            # Maximize Sharpe ratio (approximate as maximize return - risk penalty)
            portfolio_return = mean_returns.T @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
            
        elif objective_str == 'min_volatility':
            # Minimize volatility
            portfolio_risk = cp.quad_form(weights, cov_matrix)
            objective = cp.Minimize(portfolio_risk)
            
        else:
            raise ValueError(f"Objective '{objective_str}' not supported.")
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if weights.value is not None:
                return self._format_results(weights.value)
            else:
                logger.warning("Optimization failed, using equal weights")
                return self._optimize_equal_weight()
                
        except Exception as e:
            logger.error(f"Optimization error: {e}, using equal weights")
            return self._optimize_equal_weight()

    def _optimize_equal_weight(self) -> Dict:
        """Equal weight optimization."""
        n_assets = len(self.assets)
        weights = np.ones(n_assets) / n_assets
        return self._format_results(weights)

    def _format_results(self, weights: np.ndarray) -> Dict:
        """Formats the optimization results into a dictionary."""
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        expected_return = np.dot(weights, mean_returns) * 252  # Annualized
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio
        excess_return = expected_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        return {
            "weights": dict(zip(self.assets, weights)),
            "metrics": {
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            }
        } 