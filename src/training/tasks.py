import pandas as pd
from prophet import Prophet
from src.core.celery_app import celery_app
from src.database.postgres_connection import get_db
import logging
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher
from src.training.prophet_trainer import ProphetTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory to save models
MODELS_DIR = Path('./models/prophet')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data_from_db(symbol: str) -> pd.DataFrame:
    """
    Loads time-series data for a given symbol from the database.
    """
    logger.info(f"Loading data for symbol: {symbol}")
    db = get_db()
    try:
        query = "SELECT time, price FROM financial_time_series WHERE symbol = %s ORDER BY time;"
        data = db.execute_query(query, params=(symbol,), fetch='all')
        if not data:
            raise ValueError(f"No data found for symbol {symbol}")
        
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])
        logger.info(f"Loaded {len(df)} records for {symbol}")
        return df
    finally:
        db.close()

def evaluate_model(y_true, y_pred):
    """
    Calculates performance metrics for the model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape
    }

@celery_app.task
def train_prophet_model(symbol: str, auto_tune: bool = True):
    """
    Celery task to train a Prophet model for a given financial symbol.

    Args:
        symbol (str): The symbol to train the model for (e.g., 'BTC-USD').
        auto_tune (bool): Whether to perform hyperparameter tuning.
    """
    logger.info(f"Starting Prophet model training for symbol: {symbol}")
    
    try:
        data_fetcher = TimeSeriesDataFetcher()
        data = data_fetcher.fetch_data_for_symbol(symbol, days=365*3)
        
        if data.empty:
            logger.error(f"No data found for symbol {symbol}. Aborting training.")
            return {"status": "error", "message": f"No data for {symbol}"}

        trainer = ProphetTrainer(ticker=symbol)
        trainer.fit(data, auto_tune=auto_tune)

        model_path = trainer.save_model(model_dir="./models/prophet")
        
        logger.info(f"Successfully trained and saved Prophet model for {symbol} at {model_path}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "model_path": model_path,
            "best_params": trainer.best_params
        }

    except Exception as e:
        logger.error(f"Error during Prophet model training for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def train_sklearn_model(symbol: str, model_type: str):
    """
    Trains a scikit-learn model (e.g., Linear Regression, RandomForest) for a given symbol.
    """
    # Implementation of train_sklearn_model method
    pass

# For manual testing of the script
# This requires the database to be populated with data first
# Example:
# train_prophet_model.delay('BTC-USD') 