import logging
from pathlib import Path
from src.core.celery_app import celery_app
from src.prediction.prophet_service import ProphetPredictionService
from src.prediction.student_service import StudentPredictionService
from src.data_processing.time_series_data_fetcher import TimeSeriesDataFetcher
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR_PROPHET = Path('./models/prophet')
MODELS_DIR_STUDENT = Path('./models/student')

def _prepare_features_for_prediction(data):
    """Helper to create features from time series data for prediction."""
    data['returns'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data.dropna(inplace=True)
    features = ['returns', 'SMA_10', 'SMA_30']
    return data[features].values

@celery_app.task(name='prediction.predict_with_prophet')
def predict_with_prophet(symbol: str, periods: int = 30, freq: str = 'D'):
    """
    Generates a forecast for a given symbol using a trained Prophet model.

    Args:
        symbol (str): The financial symbol for which to generate a forecast.
        periods (int): The number of future periods to predict.
        freq (str): The frequency of the prediction ('D' for daily).
    
    Returns:
        A dictionary containing the status and the forecast data.
    """
    logger.info(f"Received prediction task for {symbol} for {periods} periods.")
    model_path = MODELS_DIR_PROPHET / f"{symbol}_prophet_model.json"

    if not model_path.exists():
        logger.error(f"Model for {symbol} not found at {model_path}")
        return {"status": "error", "message": "Model not found. Please train it first."}
    
    try:
        service = ProphetPredictionService(str(model_path))
        forecast = service.predict(periods=periods, freq=freq)

        if forecast is not None:
            # Convert timestamp to string for JSON serialization
            forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
            return {
                "status": "success",
                "symbol": symbol,
                "forecast": forecast.to_dict('records')
            }
        else:
            return {"status": "error", "message": "Prediction failed."}

    except Exception as e:
        logger.error(f"Error during prediction for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@celery_app.task(name='prediction.predict_with_student')
def predict_with_student(symbol: str):
    """
    Generates a prediction using a trained student model.
    """
    logger.info(f"Received student prediction task for {symbol}")
    model_dir = MODELS_DIR_STUDENT / symbol

    if not model_dir.exists():
        return {"status": "error", "message": f"Student model for {symbol} not found."}

    try:
        # Fetch the last 30 days of data to build features
        data_fetcher = TimeSeriesDataFetcher()
        data = data_fetcher.fetch_data_for_symbol(symbol, days=60)
        features = _prepare_features_for_prediction(data)
        
        # We only need the latest feature set for the next prediction
        latest_features = np.array([features[-1]])

        service = StudentPredictionService(str(model_dir))
        prediction = service.predict(latest_features)
        
        return {
            "status": "success",
            "symbol": symbol,
            "prediction": prediction.tolist()
        }
    except Exception as e:
        logger.error(f"Error during student prediction for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@celery_app.task(name='prediction.predict_next_day_price')
def predict_next_day_price(symbol: str, model_type: str = 'prophet'):
    """
    Predict the next day price for a given symbol.
    
    Args:
        symbol: Financial symbol
        model_type: Type of model to use ('prophet' or 'student')
    
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Predicting next day price for {symbol} using {model_type}")
    
    try:
        if model_type == 'prophet':
            result = predict_with_prophet(symbol, periods=1)
            if result['status'] == 'success' and result['forecast']:
                next_day_price = result['forecast'][0]['yhat']
                return {
                    "status": "success",
                    "symbol": symbol,
                    "model_type": model_type,
                    "next_day_price": next_day_price,
                    "forecast": result['forecast'][0]
                }
            else:
                return result
        
        elif model_type == 'student':
            result = predict_with_student(symbol)
            if result['status'] == 'success':
                return {
                    "status": "success",
                    "symbol": symbol,
                    "model_type": model_type,
                    "next_day_price": result['prediction'][0],
                    "prediction": result['prediction']
                }
            else:
                return result
        
        else:
            return {
                "status": "error",
                "message": f"Unknown model type: {model_type}"
            }
    
    except Exception as e:
        logger.error(f"Error predicting next day price for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(e)
        }

if __name__ == '__main__':
    # For manual testing
    # Requires a trained model to be present
    # predict_with_prophet.delay('BTC-USD')
    pass 