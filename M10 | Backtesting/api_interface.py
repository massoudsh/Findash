import pandas as pd
from typing import List, Dict, Any
# Assuming zipline.py is in the same directory.
# The `.` is crucial for relative imports within a package.
from .zipline import QuantumBacktester

def run_quantum_backtest_for_api(
    tickers: List[str],
    start_date: str,
    end_date: str,
    capital_base: float,
    strategy_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    A wrapper function to run the QuantumBacktester and format results for a JSON API.

    This function instantiates the backtester, runs the simulation, and formats
    the output into a JSON-serializable dictionary for easy consumption by a web frontend.

    Args:
        tickers: List of ticker symbols.
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD).
        capital_base: Initial capital.
        strategy_params: Dictionary of parameters for the strategy.

    Returns:
        A dictionary containing backtest results or an error message.
    """
    try:
        backtester = QuantumBacktester(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            capital_base=capital_base
        )

        # Connect all available components. These methods handle failures gracefully.
        backtester.connect_geo_agent()
        backtester.connect_prophet_forecaster()
        backtester.connect_portfolio_optimizer()
        
        # Set strategy parameters from the API request
        backtester.set_strategy_params(strategy_params)

        # Run the backtest
        perf, metrics = backtester.run_backtest()

        if perf is not None and metrics is not None:
            # Calculate drawdown from performance data
            equity_curve = perf["portfolio_value"]
            drawdown = (equity_curve / equity_curve.cummax() - 1.0)

            # Format results for JSON serialization
            return {
                "equity_curve": equity_curve.tolist(),
                "drawdown": drawdown.tolist(),
                "timestamps": [ts.strftime('%Y-%m-%d') for ts in perf.index],
                "strategy_name": strategy_params.get("name", "QuantumStrategy"),
                "metrics": metrics
            }
        else:
            return {"error": "Backtest execution failed. Check logs for details."}

    except Exception as e:
        # In a real app, you would log the full exception here.
        # import logging
        # logging.error(f"API backtest wrapper failed: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"} 