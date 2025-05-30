# your_app/utils/portfolio_utils.py

import pandas as pd
from skportfolio import Portfolio
from skportfolio.evaluation import evaluate_portfolio
from skportfolio.visualization import plot_portfolio_returns

def evaluate_trading_strategy(df):
    """
    Evaluate a trading strategy using Skfolio.
    
    Args:
        df (pd.DataFrame): DataFrame with strategy returns data.
        
    Returns:
        dict: Contains evaluation metrics and generated plots.
    """
    # Assume df has a column 'returns' which is the strategy's daily returns
    portfolio = Portfolio(returns=df['returns'])
    
    # Evaluate portfolio performance
    metrics = evaluate_portfolio(portfolio)
    
    # Generate performance visualization
    plot = plot_portfolio_returns(portfolio)

    return {'metrics': metrics, 'plot': plot}