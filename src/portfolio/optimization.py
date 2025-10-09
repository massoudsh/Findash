import pandas as pd
import numpy as np
from src.database.postgres_connection import get_db
import logging

# Try to import skfolio, fallback to custom implementation if not available
try:
    from skfolio import Portfolio
    from skfolio.optimization import MeanVarianceOptimization, ObjectiveFunction
    from skfolio.preprocessing import PricesPreprocessor
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False
    logger.warning("skfolio not available, using fallback implementation")

logger = logging.getLogger(__name__)

def load_multiple_assets_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads historical price data for multiple assets and pivots it into a single DataFrame.
    """
    logger.info(f"Loading data for symbols: {symbols} from {start_date} to {end_date}")
    db = get_db()
    try:
        query = """
            SELECT time, symbol, price 
            FROM financial_time_series 
            WHERE symbol = ANY(%s) AND time >= %s AND time <= %s
            ORDER BY time;
        """
        data = db.execute_query(query, params=(symbols, start_date, end_date), fetch='all')
        if not data:
            raise ValueError("No data found for the given symbols in the specified date range.")
        
        df = pd.DataFrame(data, columns=['time', 'symbol', 'price'])
        # Pivot the table to have symbols as columns and time as index
        price_df = df.pivot(index='time', columns='symbol', values='price')
        price_df.index = pd.to_datetime(price_df.index)
        
        # Forward-fill missing values, which can occur on non-trading days for some assets
        price_df = price_df.ffill()
        
        logger.info(f"Successfully loaded and pivoted data for {len(symbols)} assets.")
        return price_df
    finally:
        db.close()

def optimize_portfolio(symbols: list[str], start_date: str, end_date: str) -> dict:
    """
    Performs mean-variance portfolio optimization to find the portfolio with the max Sharpe ratio.
    """
    logger.info(f"Starting portfolio optimization for {symbols}")
    
    try:
        # 1. Load data
        prices = load_multiple_assets_data(symbols, start_date, end_date)
        
        if SKFOLIO_AVAILABLE:
            return _optimize_with_skfolio(prices, symbols)
        else:
            return _optimize_with_fallback(prices, symbols)

    except Exception as e:
        logger.error(f"An error occurred during portfolio optimization: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def _optimize_with_skfolio(prices: pd.DataFrame, symbols: list[str]) -> dict:
    """Optimize portfolio using skfolio library"""
    # 2. Preprocess prices to returns
    preprocessor = PricesPreprocessor()
    preprocessor.fit(prices)
    
    # 3. Define the optimization model
    model = MeanVarianceOptimization(
        objective_function=ObjectiveFunction.MAXIMIZE_SHARPE_RATIO
    )
    
    # 4. Fit the model to the preprocessed data
    model.fit(preprocessor)
    
    # 5. Get results
    weights = model.weights_
    
    portfolio = Portfolio(
        returns=preprocessor.returns_,
        weights=weights
    )

    annual_return = portfolio.annualized_mean
    annual_volatility = portfolio.annualized_std
    sharpe_ratio = portfolio.sharpe_ratio
    
    results = {
        "status": "success",
        "symbols": symbols,
        "optimal_weights": {symbol: weight for symbol, weight in zip(symbols, weights)},
        "expected_annual_return": annual_return,
        "expected_annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "method": "skfolio"
    }
    
    return results


def _optimize_with_fallback(prices: pd.DataFrame, symbols: list[str]) -> dict:
    """Fallback portfolio optimization using scipy and basic mean-variance optimization"""
    from scipy.optimize import minimize
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate expected returns and covariance matrix
    expected_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Number of assets
    n_assets = len(symbols)
    
    # Objective function: negative Sharpe ratio (to minimize)
    def objective(weights):
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Risk-free rate assumption
        risk_free_rate = 0.05
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Negative because we minimize
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1 (long-only)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_guess = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        portfolio_return = np.sum(expected_returns * optimal_weights)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - 0.05) / portfolio_volatility
        
        results = {
            "status": "success",
            "symbols": symbols,
            "optimal_weights": {symbol: weight for symbol, weight in zip(symbols, optimal_weights)},
            "expected_annual_return": portfolio_return,
            "expected_annual_volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "method": "scipy_fallback"
        }
        
        logger.info(f"Portfolio optimization successful using fallback method. Results: {results}")
        return results
    else:
        raise Exception(f"Optimization failed: {result.message}")

if __name__ == '__main__':
    # For manual testing
    # Requires data for these symbols in the database
    # test_symbols = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'USDC-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD', 'GLD', 'SLV']
    # optimize_portfolio(test_symbols, '2023-01-01', '2023-12-31')
    pass 