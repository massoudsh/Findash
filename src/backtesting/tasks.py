import pandas as pd
from prophet import Prophet
from src.core.celery_app import celery_app
from src.database.postgres_connection import get_db
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads historical financial data from the database within a specified date range.
    """
    logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")
    db = get_db()
    try:
        query = """
            SELECT time, price 
            FROM financial_time_series 
            WHERE symbol = %s AND time >= %s AND time <= %s 
            ORDER BY time;
        """
        data = db.execute_query(query, params=(symbol, start_date, end_date), fetch='all')
        if not data:
            raise ValueError(f"No data found for symbol {symbol} in the given date range.")
        
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    finally:
        db.close()

def run_prophet_prediction(historical_data: pd.DataFrame) -> float:
    """
    Trains a Prophet model on historical data and predicts the next day's price.
    A simplified, synchronous version for backtesting purposes.
    """
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(historical_data)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-1]

@celery_app.task(name='backtesting.run_backtest')
def run_backtest_task(symbol: str, start_date: str, end_date: str, initial_capital: float = 10000.0):
    """
    Runs a backtest for a simple trading strategy based on Prophet predictions.
    """
    logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
    
    try:
        df = load_historical_data(symbol, start_date, end_date)
        
        cash = initial_capital
        position = 0  # Number of shares/units held
        portfolio_history = []
        
        # We need at least 2 days of data to start making predictions
        for i in range(1, len(df) - 1):
            # The training data for the model is all data up to the current day `i`
            train_df = df.iloc[:i]
            current_price = df['y'].iloc[i]
            
            if len(train_df) < 2:  # Prophet requires at least 2 data points
                current_portfolio_value = cash
                portfolio_history.append({
                    "date": df['ds'].iloc[i].strftime('%Y-%m-%d'),
                    "value": current_portfolio_value
                })
                continue

            # Predict next day's price
            predicted_price_next_day = run_prophet_prediction(train_df)
            
            # Trading strategy
            if predicted_price_next_day > current_price:
                # Buy signal
                if cash > current_price:
                    shares_to_buy = cash / current_price
                    position += shares_to_buy
                    cash = 0
                    logger.debug(f"Day {df['ds'].iloc[i]}: Buying {shares_to_buy} at {current_price}")
            elif predicted_price_next_day < current_price:
                # Sell signal
                if position > 0:
                    cash += position * current_price
                    position = 0
                    logger.debug(f"Day {df['ds'].iloc[i]}: Selling at {current_price}")

            # Update portfolio value for the current day
            current_portfolio_value = cash + (position * current_price)
            portfolio_history.append({
                "date": df['ds'].iloc[i].strftime('%Y-%m-%d'),
                "value": current_portfolio_value
            })

        # Final portfolio value
        final_price = df['y'].iloc[-1]
        final_portfolio_value = cash + (position * final_price)
        
        # Performance metrics
        total_return_pct = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        
        results = {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_return_pct": total_return_pct,
            "portfolio_history": portfolio_history,
        }
        logger.info(f"Backtest for {symbol} completed. Results: {results}")
        return results

    except Exception as e:
        logger.error(f"An error occurred during backtesting for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    # For manual testing
    # run_backtest_task.delay('BTC-USD', '2023-01-01', '2023-12-31')
    pass 