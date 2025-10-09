import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostTeacher:
    """Teacher model using XGBoost for regression tasks."""
    
    def __init__(self, params: Optional[Dict] = None):
        self.default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
        }
        self.params = {**self.default_params, **(params or {})}
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            eval_set: Optional[List] = None, early_stopping_rounds: int = 50):
        logger.info("Training XGBoost teacher model...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = xgb.XGBRegressor(**self.params)
        
        eval_set_scaled = None
        if eval_set:
            eval_set_scaled = [(self.scaler.transform(eval_set[0][0]), eval_set[0][1])]
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set_scaled,
            early_stopping_rounds=early_stopping_rounds,
            verbose=True
        )
        logger.info("XGBoost teacher model training completed.")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained, call fit() first")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, 'xgboost_teacher.json')
        self.model.save_model(model_path)
        
        scaler_path = os.path.join(folder_path, 'xgboost_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        params_path = os.path.join(folder_path, 'xgboost_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.params, f)
            
        logger.info(f"XGBoost teacher model saved to {folder_path}")
        
    @classmethod
    def load(cls, folder_path: str) -> 'XGBoostTeacher':
        model_path = os.path.join(folder_path, 'xgboost_teacher.json')
        scaler_path = os.path.join(folder_path, 'xgboost_scaler.joblib')
        params_path = os.path.join(folder_path, 'xgboost_params.json')
        
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        instance = cls(params)
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(model_path)
        instance.scaler = joblib.load(scaler_path)
        
        logger.info(f"XGBoost teacher model loaded from {folder_path}")
        return instance
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        } 