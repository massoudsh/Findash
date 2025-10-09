import pandas as pd
import logging
import json
from pathlib import Path
from typing import Optional

from prophet import Prophet
from prophet.serialize import model_from_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProphetPredictionService:
    """
    Handles loading a trained Prophet model and making predictions.
    """
    
    def __init__(self, model_path: str):
        self.model: Optional[Prophet] = self._load_model(model_path)
        if self.model is None:
            raise FileNotFoundError(f"Could not load model from {model_path}")

    def _load_model(self, model_path: str) -> Optional[Prophet]:
        """Loads a serialized Prophet model from a file."""
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model file not found at {model_path}")
            return None
        
        try:
            with open(path, 'r') as f:
                model = model_from_json(json.load(f))
            logger.info(f"Successfully loaded Prophet model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def predict(self, periods: int = 30, freq: str = 'D') -> Optional[pd.DataFrame]:
        """
        Generates future predictions.

        Args:
            periods (int): The number of periods to forecast into the future.
            freq (str): The frequency of the forecast (e.g., 'D' for day, 'H' for hour).

        Returns:
            A pandas DataFrame with the forecast, or None if prediction fails.
        """
        if self.model is None:
            logger.error("Cannot predict, no model is loaded.")
            return None
            
        try:
            future_df = self.model.make_future_dataframe(periods=periods, freq=freq)
            forecast = self.model.predict(future_df)
            logger.info(f"Generated forecast for the next {periods} periods.")
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None 