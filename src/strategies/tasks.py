import logging
from typing import List, Dict, Any

from src.core.celery_app import celery_app
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher
from src.strategies.momentum import MomentumStrategy

STRATEGY_MAPPING = {
    "momentum": MomentumStrategy,
}

logger = logging.getLogger(__name__)

@celery_app.task(name="strategies.run_backtest")
def run_strategy_backtest(strategy_name: str, 
                        symbols: List[str], 
                        initial_capital: float = 100000.0,
                        strategy_params: Dict[str, Any] = None) -> Dict:
    """
    Celery task to run a backtest for a given strategy.
    """
    logger.info(f"Starting backtest for strategy '{strategy_name}' with symbols {symbols}")
    
    strategy_class = STRATEGY_MAPPING.get(strategy_name.lower())
    if not strategy_class:
        return {"status": "error", "message": f"Strategy '{strategy_name}' not found."}

    try:
        data_fetcher = TimeSeriesDataFetcher()
        prices_data = data_fetcher.fetch_multiple_symbols(symbols, days=365*3) # 3 years of data

        if prices_data.empty:
            return {"status": "error", "message": "Could not fetch data for backtest."}

        strategy = strategy_class(**(strategy_params or {}))
        strategy.load_data(prices_data)
        
        results = strategy.backtest(initial_capital=initial_capital)
        
        return {"status": "success", "strategy": strategy_name, "results": results}

    except Exception as e:
        logger.error(f"Error during backtest for strategy {strategy_name}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)} 