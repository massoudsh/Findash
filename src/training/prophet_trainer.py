import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProphetTrainer:
    """
    Handles training, tuning, and evaluation of Prophet models.
    """
    
    def __init__(self, 
                 ticker: str, 
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 holidays: pd.DataFrame = None,
                 interval_width: float = 0.8):
        self.ticker = ticker
        self.model: Optional[Prophet] = None
        self.best_params: Optional[Dict[str, Any]] = None
        
        self.params = {
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'holidays': holidays,
            'interval_width': interval_width
        }
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index(pd.to_datetime(data.index))
        
        prophet_df = pd.DataFrame({'ds': data.index, 'y': data[target_col]})
        prophet_df = prophet_df.dropna()
        return prophet_df
        
    def hyperparameter_tuning(self, data: pd.DataFrame, optimization_metric: str = 'mape',
                             param_grid: Optional[Dict] = None) -> Dict:
        if len(data) < 100:
            logger.warning("Insufficient data for hyperparameter tuning. Using default parameters.")
            return self.params
            
        logger.info(f"Starting hyperparameter tuning for {self.ticker}")
        
        param_grid = param_grid or {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        
        all_params = [dict(zip(param_grid.keys(), v)) for v in pd.np.array(pd.np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]
        rmses = []

        for params in all_params:
            m = Prophet(**params).fit(data)
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        
        self.best_params = all_params[np.argmin(rmses)]
        logger.info(f"Best parameters for {self.ticker}: {self.best_params}")
        self.params.update(self.best_params)
        return self.params

    def fit(self, data: pd.DataFrame, auto_tune: bool = True):
        prophet_data = self.prepare_data(data)
        
        if auto_tune:
            self.hyperparameter_tuning(prophet_data)
        
        self.model = Prophet(**self.params)
        self.model.fit(prophet_data)
        logger.info(f"Successfully trained Prophet model for {self.ticker}")

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .fit() first.")
        
        test_df = self.prepare_data(test_data)
        forecast = self.model.predict(test_df)
        
        metrics = performance_metrics(
            cross_validation(self.model, initial=f'{len(test_data) - 30} days', period='15 days', horizon='30 days')
        )
        logger.info(f"Evaluation metrics for {self.ticker}: \n{metrics.head()}")
        return metrics.to_dict('records')[0]

    def save_model(self, model_dir: str) -> str:
        if self.model is None:
            raise ValueError("No model to save.")

        path = Path(model_dir) / f"{self.ticker}_prophet_model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(model_to_json(self.model), f)
            
        logger.info(f"Model for {self.ticker} saved to {path}")
        return str(path) 