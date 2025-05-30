import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Import skfolio for portfolio optimization
import skfolio
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.population import HierarchicalRiskParity, HierarchicalEqualRiskContribution
from skfolio.preprocessing import prices_to_returns
from skfolio.datasets import load_sp500_dataset
from skfolio.metrics import sharpe_ratio, maximum_drawdown, conditional_drawdown_at_risk
from skfolio.plotting import plot_efficient_frontier, plot_dendrogram
from skfolio.model_selection import WalkForward, cross_val_predict

# Import risk management module from our codebase
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from M6___Risk_Management.risk_manager import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using skfolio library.
    
    This class implements portfolio optimization techniques including:
    - Mean-Variance optimization (Markowitz)
    - Hierarchical Risk Parity (HRP)
    - Hierarchical Equal Risk Contribution (HERC)
    - Max Sharpe Ratio optimization
    - Min Volatility optimization
    - Max Diversification optimization
    - Walkforward optimization and backtesting
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 target_return: Optional[float] = None,
                 max_volatility: Optional[float] = None,
                 min_weight: float = 0.01,
                 max_weight: float = 0.25):
        """
        Initialize the Portfolio Optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            max_volatility: Maximum allowed portfolio volatility
            min_weight: Minimum weight per asset (default: 1%)
            max_weight: Maximum weight per asset (default: 25%)
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_volatility = max_volatility
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # For storing results
        self.optimized_weights = None
        self.optimization_results = None
        self.assets = None
        self.returns = None
        self.covariance = None
        self.efficient_frontier = None
        
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(self, prices: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Prepare price data for optimization.
        
        Args:
            prices: DataFrame with datetime index and price columns for each asset
            freq: Frequency for returns calculation ('D' for daily, 'M' for monthly)
            
        Returns:
            DataFrame with returns
        """
        # Ensure data is sorted by date
        prices = prices.sort_index()
        
        # Convert prices to returns
        returns = prices_to_returns(prices, freq=freq)
        
        # Store assets and returns
        self.assets = returns.columns.tolist()
        self.returns = returns
        
        # Calculate asset covariance matrix
        self.covariance = returns.cov()
        
        logger.info(f"Prepared data for {len(self.assets)} assets with {len(returns)} periods")
        return returns
    
    def optimize_mean_variance(self, objective: str = 'sharpe_ratio') -> Dict:
        """
        Perform mean-variance optimization (Markowitz).
        
        Args:
            objective: Objective function ('sharpe_ratio', 'min_volatility', 'max_return', 
                      'min_cvar', 'max_diversification')
            
        Returns:
            Dictionary with optimization results
        """
        if self.returns is None:
            logger.error("No return data available. Call prepare_data() first.")
            return None
        
        # Map string objective to skfolio ObjectiveFunction
        objective_map = {
            'sharpe_ratio': ObjectiveFunction.SHARPE_RATIO,
            'min_volatility': ObjectiveFunction.MIN_VOLATILITY,
            'max_return': ObjectiveFunction.MAX_RETURN,
            'min_cvar': ObjectiveFunction.MIN_CVAR,
            'max_diversification': ObjectiveFunction.MAX_DIVERSIFICATION
        }
        
        if objective not in objective_map:
            logger.error(f"Invalid objective: {objective}. Choose from: {list(objective_map.keys())}")
            return None
        
        # Setup optimization parameters
        opt_params = {
            'objective_function': objective_map[objective],
            'min_weights': self.min_weight,
            'max_weights': self.max_weight,
            'risk_free_rate': self.risk_free_rate
        }
        
        # Add target return constraint if specified
        if self.target_return is not None:
            opt_params['min_return'] = self.target_return
        
        # Add max volatility constraint if specified
        if self.max_volatility is not None:
            opt_params['max_volatility'] = self.max_volatility
        
        try:
            # Create and fit the optimizer
            optimizer = MeanRisk(**opt_params)
            optimizer.fit(self.returns)
            
            # Get weights and metrics
            weights = optimizer.weights_
            expected_return = optimizer.expected_return_
            expected_volatility = optimizer.expected_volatility_
            sharpe = optimizer.sharpe_ratio_
            
            # Store results
            self.optimized_weights = weights
            self.optimization_results = {
                'weights': weights,
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'sharpe_ratio': sharpe,
                'objective': objective,
                'method': 'mean_variance'
            }
            
            logger.info(f"Mean-Variance optimization ({objective}) complete:")
            logger.info(f"Expected Return: {expected_return:.4f}")
            logger.info(f"Expected Volatility: {expected_volatility:.4f}")
            logger.info(f"Sharpe Ratio: {sharpe:.4f}")
            
            return self.optimization_results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
    
    def optimize_hierarchical(self, method: str = 'HRP') -> Dict:
        """
        Perform hierarchical portfolio optimization.
        
        Args:
            method: Hierarchical method ('HRP' for Hierarchical Risk Parity or
                  'HERC' for Hierarchical Equal Risk Contribution)
            
        Returns:
            Dictionary with optimization results
        """
        if self.returns is None:
            logger.error("No return data available. Call prepare_data() first.")
            return None
        
        try:
            if method == 'HRP':
                # Hierarchical Risk Parity
                optimizer = HierarchicalRiskParity(
                    linkage='single',
                    risk_measure='variance'
                )
            elif method == 'HERC':
                # Hierarchical Equal Risk Contribution
                optimizer = HierarchicalEqualRiskContribution(
                    linkage='single',
                    risk_measure='variance'
                )
            else:
                logger.error(f"Invalid method: {method}. Choose from: ['HRP', 'HERC']")
                return None
            
            # Fit the optimizer
            optimizer.fit(self.returns)
            
            # Get weights and metrics
            weights = optimizer.weights_
            expected_return = np.sum(weights * self.returns.mean())
            covariance_matrix = self.returns.cov().values
            expected_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
            sharpe = (expected_return - self.risk_free_rate) / expected_volatility
            
            # Create cluster dendrogram for visualization
            linkage_matrix = optimizer.linkage_matrix_
            
            # Store results
            self.optimized_weights = weights
            self.optimization_results = {
                'weights': weights,
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'sharpe_ratio': sharpe,
                'method': method,
                'linkage_matrix': linkage_matrix
            }
            
            logger.info(f"{method} optimization complete:")
            logger.info(f"Expected Return: {expected_return:.4f}")
            logger.info(f"Expected Volatility: {expected_volatility:.4f}")
            logger.info(f"Sharpe Ratio: {sharpe:.4f}")
            
            return self.optimization_results
            
        except Exception as e:
            logger.error(f"Hierarchical optimization failed: {e}")
            return None
    
    def compute_efficient_frontier(self, points: int = 20) -> pd.DataFrame:
        """
        Compute efficient frontier points.
        
        Args:
            points: Number of points in the efficient frontier
            
        Returns:
            DataFrame with efficient frontier points
        """
        if self.returns is None:
            logger.error("No return data available. Call prepare_data() first.")
            return None
        
        try:
            # Get range of possible returns
            min_vol_port = MeanRisk(
                objective_function=ObjectiveFunction.MIN_VOLATILITY,
                min_weights=self.min_weight,
                max_weights=self.max_weight
            ).fit(self.returns)
            
            max_return_port = MeanRisk(
                objective_function=ObjectiveFunction.MAX_RETURN,
                min_weights=self.min_weight,
                max_weights=self.max_weight
            ).fit(self.returns)
            
            min_return = min_vol_port.expected_return_
            max_return = max_return_port.expected_return_
            
            # Create range of target returns
            target_returns = np.linspace(min_return, max_return, points)
            
            # Compute efficient frontier
            efficient_frontier = []
            for target_return in target_returns:
                port = MeanRisk(
                    objective_function=ObjectiveFunction.MIN_VOLATILITY,
                    min_return=target_return,
                    min_weights=self.min_weight,
                    max_weights=self.max_weight
                ).fit(self.returns)
                
                efficient_frontier.append({
                    'return': port.expected_return_,
                    'volatility': port.expected_volatility_,
                    'sharpe': port.sharpe_ratio_,
                    'weights': port.weights_
                })
            
            # Convert to DataFrame
            ef_df = pd.DataFrame(efficient_frontier)
            self.efficient_frontier = ef_df
            
            logger.info(f"Computed efficient frontier with {points} points")
            return ef_df
            
        except Exception as e:
            logger.error(f"Computing efficient frontier failed: {e}")
            return None
    
    def backtest_walkforward(self, 
                            n_splits: int = 5, 
                            train_size: float = 0.5,
                            method: str = 'mean_variance',
                            objective: str = 'sharpe_ratio') -> Dict:
        """
        Perform walk-forward optimization and backtesting.
        
        Args:
            n_splits: Number of train/test splits
            train_size: Proportion of data for training
            method: Optimization method ('mean_variance', 'HRP', 'HERC')
            objective: Objective function (for mean_variance only)
            
        Returns:
            Dictionary with backtesting results
        """
        if self.returns is None:
            logger.error("No return data available. Call prepare_data() first.")
            return None
        
        try:
            # Create optimizer based on method
            if method == 'mean_variance':
                # Map string objective to skfolio ObjectiveFunction
                objective_map = {
                    'sharpe_ratio': ObjectiveFunction.SHARPE_RATIO,
                    'min_volatility': ObjectiveFunction.MIN_VOLATILITY,
                    'max_return': ObjectiveFunction.MAX_RETURN,
                    'min_cvar': ObjectiveFunction.MIN_CVAR,
                    'max_diversification': ObjectiveFunction.MAX_DIVERSIFICATION
                }
                
                if objective not in objective_map:
                    logger.error(f"Invalid objective: {objective}. Choose from: {list(objective_map.keys())}")
                    return None
                
                optimizer = MeanRisk(
                    objective_function=objective_map[objective],
                    min_weights=self.min_weight,
                    max_weights=self.max_weight,
                    risk_free_rate=self.risk_free_rate
                )
            elif method == 'HRP':
                optimizer = HierarchicalRiskParity(
                    linkage='single',
                    risk_measure='variance'
                )
            elif method == 'HERC':
                optimizer = HierarchicalEqualRiskContribution(
                    linkage='single',
                    risk_measure='variance'
                )
            else:
                logger.error(f"Invalid method: {method}. Choose from: ['mean_variance', 'HRP', 'HERC']")
                return None
            
            # Create walk-forward cross-validator
            cv = WalkForward(
                n_splits=n_splits,
                train_size=train_size,
                gap=1  # 1 period gap between train and test
            )
            
            # Perform walk-forward optimization
            predicted_returns = cross_val_predict(
                model=optimizer,
                X=self.returns,
                cv=cv,
                n_jobs=-1
            )
            
            # Calculate backtest metrics
            backtest_sharpe = sharpe_ratio(predicted_returns)
            backtest_mdd = maximum_drawdown(predicted_returns)
            backtest_cdar = conditional_drawdown_at_risk(predicted_returns, alpha=0.05)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + predicted_returns).cumprod()
            
            # Store results
            backtest_results = {
                'predicted_returns': predicted_returns,
                'cumulative_returns': cumulative_returns,
                'sharpe_ratio': backtest_sharpe,
                'maximum_drawdown': backtest_mdd,
                'conditional_drawdown_at_risk': backtest_cdar,
                'method': method,
                'objective': objective if method == 'mean_variance' else None,
                'n_splits': n_splits,
                'train_size': train_size
            }
            
            logger.info(f"Walk-forward backtesting complete ({method}):")
            logger.info(f"Sharpe Ratio: {backtest_sharpe:.4f}")
            logger.info(f"Maximum Drawdown: {backtest_mdd:.4f}")
            logger.info(f"Conditional Drawdown at Risk (5%): {backtest_cdar:.4f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return None
    
    def plot_optimal_portfolio(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimal portfolio weights and metrics.
        
        Args:
            save_path: Path to save the plot (if None, display only)
        """
        if self.optimized_weights is None:
            logger.error("No optimized weights available. Run optimization first.")
            return
        
        weights = self.optimized_weights
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot weights as pie chart
        plt.subplot(2, 2, 1)
        plt.pie(weights, labels=self.assets, autopct='%1.1f%%')
        plt.title('Optimal Portfolio Weights')
        
        # Plot weights as bar chart
        plt.subplot(2, 2, 2)
        plt.bar(self.assets, weights)
        plt.xticks(rotation=45, ha='right')
        plt.title('Asset Allocation')
        plt.ylabel('Weight')
        
        # If we have efficient frontier data, plot it
        if self.efficient_frontier is not None:
            plt.subplot(2, 2, 3)
            plt.scatter(
                self.efficient_frontier['volatility'], 
                self.efficient_frontier['return'], 
                c=self.efficient_frontier['sharpe'], 
                cmap='viridis'
            )
            
            # Mark the chosen portfolio
            if 'expected_volatility' in self.optimization_results:
                plt.scatter(
                    self.optimization_results['expected_volatility'],
                    self.optimization_results['expected_return'],
                    marker='*',
                    s=200,
                    c='red',
                    label='Selected Portfolio'
                )
            
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Expected Volatility')
            plt.ylabel('Expected Return')
            plt.title('Efficient Frontier')
            plt.legend()
        
        # Plot portfolio metrics
        plt.subplot(2, 2, 4)
        if self.optimization_results:
            metrics = {
                'Expected Return': self.optimization_results.get('expected_return', 0),
                'Expected Volatility': self.optimization_results.get('expected_volatility', 0),
                'Sharpe Ratio': self.optimization_results.get('sharpe_ratio', 0)
            }
            
            plt.bar(metrics.keys(), metrics.values())
            plt.ylabel('Value')
            plt.title('Portfolio Metrics')
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Portfolio plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_hierarchical_clusters(self, save_path: Optional[str] = None) -> None:
        """
        Plot hierarchical clustering dendrogram for HRP/HERC methods.
        
        Args:
            save_path: Path to save the plot (if None, display only)
        """
        if self.optimization_results is None or 'linkage_matrix' not in self.optimization_results:
            logger.error("No hierarchical clustering results available. Run hierarchical optimization first.")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot dendrogram
            linkage_matrix = self.optimization_results['linkage_matrix']
            plot_dendrogram(
                linkage_matrix=linkage_matrix,
                labels=self.assets,
                leaf_rotation=90
            )
            
            plt.title('Hierarchical Clustering of Assets')
            plt.xlabel('Assets')
            plt.ylabel('Distance')
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Dendrogram plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting dendrogram: {e}")
    
    def integrate_with_risk_manager(self, 
                                  capital: float, 
                                  risk_manager: RiskManager,
                                  max_portfolio_risk: Optional[float] = None) -> Dict:
        """
        Integrate optimized portfolio with risk management system.
        
        Args:
            capital: Total portfolio capital
            risk_manager: RiskManager instance
            max_portfolio_risk: Maximum portfolio risk (if None, use risk manager default)
            
        Returns:
            Dictionary with position sizes and risk metrics
        """
        if self.optimized_weights is None:
            logger.error("No optimized weights available. Run optimization first.")
            return None
        
        try:
            # If max_portfolio_risk is provided, update risk manager
            if max_portfolio_risk is not None:
                risk_manager.max_portfolio_risk = max_portfolio_risk
            
            # Calculate position sizes and risk
            positions = {}
            total_risk = 0
            
            for i, asset in enumerate(self.assets):
                weight = self.optimized_weights[i]
                position_capital = capital * weight
                
                # For each asset, calculate stop loss based on volatility
                if self.returns is not None:
                    asset_returns = self.returns[asset]
                    asset_volatility = asset_returns.std() * np.sqrt(252)  # Annualized volatility
                    
                    # Use current price from the last row of returns
                    price_data = self.returns.index[-1]
                    current_price = 100  # Default placeholder
                    
                    # Calculate stop loss (2 standard deviations away)
                    stop_loss = current_price * (1 - 2 * asset_volatility)
                    
                    # Use risk manager to calculate position size
                    position_size = risk_manager.calculate_position_size(
                        capital=position_capital,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        symbol=asset
                    )
                    
                    # Calculate value at risk
                    var_95 = asset_returns.quantile(0.05) * position_size * current_price
                    
                    # Store position information
                    positions[asset] = {
                        'weight': weight,
                        'capital': position_capital,
                        'position_size': position_size,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'var_95': var_95,
                        'volatility': asset_volatility
                    }
                    
                    # Add to total risk
                    total_risk += abs(var_95)
            
            # Calculate portfolio risk metrics
            portfolio_risk = total_risk / capital
            portfolio_metrics = {
                'total_capital': capital,
                'portfolio_risk': portfolio_risk,
                'positions': positions
            }
            
            logger.info(f"Portfolio integration complete:")
            logger.info(f"Total Capital: ${capital:.2f}")
            logger.info(f"Portfolio Risk: {portfolio_risk:.4f}")
            
            return portfolio_metrics
            
        except Exception as e:
            logger.error(f"Risk integration failed: {e}")
            return None
    
    def save_optimization_results(self, filename: str = None) -> str:
        """
        Save optimization results to disk.
        
        Args:
            filename: Filename to save results (if None, generates a default name)
            
        Returns:
            Path to saved file
        """
        if self.optimization_results is None:
            logger.error("No optimization results to save. Run optimization first.")
            return None
        
        if filename is None:
            # Generate filename with timestamp and method
            method = self.optimization_results.get('method', 'optimization')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{method}_{timestamp}.json"
        
        # Create full path
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            # Prepare data for serialization
            save_data = self.optimization_results.copy()
            
            # Convert numpy arrays to lists for JSON serialization
            for key, value in save_data.items():
                if isinstance(value, np.ndarray):
                    save_data[key] = value.tolist()
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            
            logger.info(f"Optimization results saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    @classmethod
    def load_optimization_results(cls, file_path: str) -> Dict:
        """
        Load optimization results from disk.
        
        Args:
            file_path: Path to saved results
            
        Returns:
            Dictionary with optimization results
        """
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Optimization results loaded from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Example: Portfolio optimization with skfolio
    try:
        # Load sample data from skfolio
        prices = load_sp500_dataset()
        
        # Create optimizer
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            min_weight=0.01,
            max_weight=0.2
        )
        
        # Prepare data
        returns = optimizer.prepare_data(prices)
        
        # Run mean-variance optimization with Sharpe ratio objective
        mv_results = optimizer.optimize_mean_variance(objective='sharpe_ratio')
        
        # Compute efficient frontier
        ef = optimizer.compute_efficient_frontier(points=20)
        
        # Plot optimal portfolio
        optimizer.plot_optimal_portfolio(save_path="optimal_portfolio_mv.png")
        
        # Run hierarchical risk parity optimization
        hrp_results = optimizer.optimize_hierarchical(method='HRP')
        
        # Plot hierarchical clusters
        optimizer.plot_hierarchical_clusters(save_path="hierarchical_clusters.png")
        
        # Backtest using walk-forward optimization
        backtest_results = optimizer.backtest_walkforward(
            n_splits=5,
            train_size=0.6,
            method='mean_variance',
            objective='sharpe_ratio'
        )
        
        # Integrate with risk manager
        risk_manager = RiskManager(max_portfolio_risk=0.02)
        portfolio = optimizer.integrate_with_risk_manager(
            capital=100000,
            risk_manager=risk_manager
        )
        
        # Save results
        optimizer.save_optimization_results()
        
        print("Portfolio optimization complete!")
        
    except Exception as e:
        print(f"Error in example: {e}")
        print("Note: This example requires actual data to run properly.") 