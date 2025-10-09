import torch
import joblib
import json
import os
import numpy as np
from typing import Dict, Any

from src.training.student_model import StudentModel

class StudentPredictionService:
    """Handles loading and running the distilled student model."""

    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model, self.scaler = self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        params_path = os.path.join(model_dir, 'student_params.json')
        with open(params_path, 'r') as f:
            params = json.load(f)

        model = StudentModel(input_dim=params['input_dim'], hidden_dims=params['hidden_dims'])
        model_path = os.path.join(model_dir, 'student_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        scaler_path = os.path.join(model_dir, 'student_scaler.joblib')
        scaler = joblib.load(scaler_path)

        return model, scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes a prediction with the loaded student model."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction = self.model(X_tensor)
        
        return prediction.cpu().numpy() 