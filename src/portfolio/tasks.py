import logging
from typing import List, Dict
import numpy as np

from src.core.celery_app import celery_app
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher
from .optimizer import PortfolioOptimizer
from .metrics import PortfolioMetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(name="portfolio.run_optimization")
def run_portfolio_optimization(symbols: List[str], 
                             method: str = 'HRP', 
                             objective: str = 'sharpe_ratio') -> Dict:
    """
    Celery task to run portfolio optimization.

    Args:
        symbols (List[str]): A list of ticker symbols.
        method (str): The optimization method to use ('HRP', 'Mean-Variance').
        objective (str): The objective for Mean-Variance ('sharpe_ratio', 'min_volatility').

    Returns:
        A dictionary containing the optimization results.
    """
    logger.info(f"Starting portfolio optimization for symbols: {symbols}")
    try:
        data_fetcher = TimeSeriesDataFetcher()
        # Use 2 years of daily data for optimization
        prices_data = data_fetcher.fetch_multiple_symbols(symbols, days=365*2)

        if prices_data.empty or prices_data.isnull().values.any():
            logger.error("Fetched data is empty or contains NaNs.")
            return {"status": "error", "message": "Insufficient data for optimization."}

        optimizer = PortfolioOptimizer(prices_data=prices_data)
        results = optimizer.run_optimization(method=method, objective=objective)
        
        # Calculate full metrics suite on the optimized portfolio
        weights_array = np.array(list(results['weights'].values()))
        full_metrics = PortfolioMetricsCalculator.calculate_all_metrics(
            returns=optimizer.returns, 
            weights=weights_array
        )
        results['full_metrics'] = full_metrics
        
        logger.info("Portfolio optimization finished successfully.")
        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Error during portfolio optimization: {e}", exc_info=True)
        return {"status": "error", "message": str(e)} 