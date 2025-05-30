import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from skfolio import RiskMeasure, Portfolio
from skfolio.metrics import (
    mean, volatility, sharpe, maximum_drawdown, 
    cvar, value_at_risk, calmar, sortino
)
from core.logging_config import setup_logging
from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import alpaca_trade_api as tradeapi

# Initialize logger
logger = logging.getLogger(__name__)

class PortfolioMetrics:
    """
    Calculate and visualize portfolio risk metrics.
    
    This class provides utilities for:
    - Computing comprehensive risk metrics
    - Visualizing portfolio performance
    - Generating risk reports
    - Comparing portfolios
    """
    
    @staticmethod
    def calculate_risk_metrics(
        returns: pd.DataFrame, 
        weights: np.ndarray,
        risk_free_rate: float = 0.0,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            returns: Asset returns dataframe
            weights: Portfolio weights
            risk_free_rate: Risk-free rate
            alpha: Confidence level for VaR/CVaR
            
        Returns:
            Dictionary of risk metrics
        """
        # Create portfolio object
        portfolio = Portfolio(returns=returns, weights=weights)
        
        # Calculate metrics
        metrics = {
            "mean": mean(portfolio, risk_free_rate),
            "volatility": volatility(portfolio),
            "sharpe": sharpe(portfolio, risk_free_rate),
            "sortino": sortino(portfolio, risk_free_rate),
            "calmar": calmar(portfolio, risk_free_rate),
            "maximum_drawdown": maximum_drawdown(portfolio),
            "var": value_at_risk(portfolio, alpha=alpha),
            "cvar": cvar(portfolio, alpha=alpha),
            "annual_return": mean(portfolio, risk_free_rate, period=252),
            "annual_volatility": volatility(portfolio, period=252)
        }
        
        return metrics
    
    @staticmethod
    def plot_efficient_frontier(
        efficient_frontier, 
        title: str = "Efficient Frontier",
        highlight_portfolio: bool = True,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot the efficient frontier.
        
        Args:
            efficient_frontier: Efficient frontier object from optimization
            title: Plot title
            highlight_portfolio: Whether to highlight the optimal portfolio
            save_path: Path to save the plot (if None, won't save)
            show_plot: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot the efficient frontier
        ax = efficient_frontier.plot(show=False, assets=True)
        
        # Customize plot
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_weights_distribution(
        weights: np.ndarray,
        asset_names: List[str],
        title: str = "Portfolio Weights Distribution",
        min_weight: float = 0.01,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot the distribution of portfolio weights.
        
        Args:
            weights: Portfolio weights
            asset_names: List of asset names
            title: Plot title
            min_weight: Minimum weight to display (filters out tiny allocations)
            save_path: Path to save the plot (if None, won't save)
            show_plot: Whether to display the plot
        """
        # Create DataFrame with weights
        df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights
        })
        
        # Filter out small weights
        df = df[df['Weight'] >= min_weight]
        
        # Sort by weight
        df = df.sort_values('Weight', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Asset', y='Weight', data=df)
        
        # Add values on top of bars
        for i, v in enumerate(df['Weight']):
            ax.text(i, v + 0.005, f"{v:.2%}", ha='center')
            
        # Customize plot
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_stress_test_results(
        stress_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['mean', 'cvar', 'sharpe'],
        title: str = "Portfolio Stress Test Results",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot the results of stress tests.
        
        Args:
            stress_results: Dictionary of stress test results
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot (if None, won't save)
            show_plot: Whether to display the plot
        """
        # Prepare data
        scenarios = list(stress_results.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [stress_results[scenario][metric] for scenario in scenarios]
            
            # Create bar chart
            ax = axes[i]
            bars = ax.bar(scenarios, values)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + (0.01 * max(values)),
                    f"{height:.4f}",
                    ha='center', va='bottom'
                )
                
            # Customize subplot
            ax.set_title(f"{metric.capitalize()}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(metric.capitalize())
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
        
        # Add main title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def generate_risk_report(
        portfolio_metrics: Dict[str, float],
        stress_results: Optional[Dict[str, Dict[str, float]]] = None,
        weights: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a comprehensive risk report.
        
        Args:
            portfolio_metrics: Dictionary of portfolio metrics
            stress_results: Dictionary of stress test results (optional)
            weights: Portfolio weights (optional)
            asset_names: Asset names (optional)
            
        Returns:
            DataFrame with the risk report
        """
        # Create base report with portfolio metrics
        report_data = []
        
        # Add portfolio metrics
        for metric, value in portfolio_metrics.items():
            report_data.append({
                'Category': 'Portfolio Metrics',
                'Metric': metric.replace('_', ' ').capitalize(),
                'Value': f"{value:.4f}" if isinstance(value, (int, float)) else value
            })
            
        # Add stress test results if provided
        if stress_results:
            for scenario, metrics in stress_results.items():
                for metric, value in metrics.items():
                    report_data.append({
                        'Category': f'Stress Test: {scenario}',
                        'Metric': metric.replace('_', ' ').capitalize(),
                        'Value': f"{value:.4f}" if isinstance(value, (int, float)) else value
                    })
        
        # Add weight allocation if provided
        if weights is not None and asset_names is not None:
            for asset, weight in zip(asset_names, weights):
                if weight >= 0.01:  # Only include weights above 1%
                    report_data.append({
                        'Category': 'Weight Allocation',
                        'Metric': asset,
                        'Value': f"{weight:.2%}"
                    })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df

# Example usage function
def analyze_portfolio_risk(
    portfolio: Portfolio,
    stress_results: Optional[Dict[str, Dict[str, float]]] = None,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
    generate_plots: bool = True,
    save_plots: bool = False,
    plot_dir: str = "./plots"
) -> Dict[str, Any]:
    """
    Run a complete portfolio risk analysis.
    
    Args:
        portfolio: Portfolio object
        stress_results: Dictionary of stress test results (optional)
        risk_free_rate: Risk-free rate for calculations
        alpha: Confidence level for VaR/CVaR
        generate_plots: Whether to generate plots
        save_plots: Whether to save plots
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Calculate risk metrics
        metrics = PortfolioMetrics.calculate_risk_metrics(
            portfolio.returns, portfolio.weights, risk_free_rate, alpha
        )
        
        # Generate risk report
        report = PortfolioMetrics.generate_risk_report(
            metrics, 
            stress_results,
            portfolio.weights,
            portfolio.returns.columns.tolist()
        )
        
        results = {
            "metrics": metrics,
            "report": report
        }
        
        # Generate plots if requested
        if generate_plots:
            if save_plots:
                import os
                os.makedirs(plot_dir, exist_ok=True)
                
            # Plot weights distribution
            PortfolioMetrics.plot_weights_distribution(
                portfolio.weights,
                portfolio.returns.columns.tolist(),
                save_path=f"{plot_dir}/weights.png" if save_plots else None,
                show_plot=True
            )
            
            # Plot stress test results if available
            if stress_results:
                PortfolioMetrics.plot_stress_test_results(
                    stress_results,
                    save_path=f"{plot_dir}/stress_tests.png" if save_plots else None,
                    show_plot=True
                )
            
            results["plots_generated"] = True
            
        return results
    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {str(e)}")
        raise e

# If run directly, execute a simple demo
if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Create a sample portfolio
    from skfolio.datasets import load_sp500_dataset
    
    # Load sample data
    prices = load_sp500_dataset()
    returns = prices.pct_change().dropna()
    
    # Create equal weight portfolio
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    
    # Create portfolio
    portfolio = Portfolio(returns=returns, weights=weights)
    
    # Run analysis
    results = analyze_portfolio_risk(portfolio, generate_plots=True)
    
    # Print report
    print(results["report"]) 