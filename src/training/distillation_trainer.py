import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any

from .xgboost_teacher import XGBoostTeacher
from .student_model import StudentModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistillationTrainer:
    """Orchestrates the knowledge distillation process."""
    
    def __init__(self, teacher_model: XGBoostTeacher, device: str = 'cpu'):
        self.teacher_model = teacher_model
        self.device = torch.device(device)
        self.scaler = StandardScaler()
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, torch.Tensor]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_teacher = self.teacher_model.predict(X_train)
        
        return {
            'X_train': torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device),
            'y_train': torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device),
            'y_train_teacher': torch.tensor(y_train_teacher.reshape(-1, 1), dtype=torch.float32).to(self.device),
            'X_test': torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device),
            'y_test': torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(self.device),
        }
        
    def train_student(self, data_tensors: Dict[str, torch.Tensor], student_model: StudentModel, 
                      alpha: float = 0.5, epochs: int = 100, learning_rate: float = 0.001) -> Tuple[StudentModel, Dict]:
        student_model = student_model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            student_model.train()
            optimizer.zero_grad()
            
            student_pred = student_model(data_tensors['X_train'])
            student_loss = criterion(student_pred, data_tensors['y_train'])
            distill_loss = criterion(student_pred, data_tensors['y_train_teacher'])
            combined_loss = alpha * student_loss + (1 - alpha) * distill_loss
            
            combined_loss.backward()
            optimizer.step()
            
            student_model.eval()
            with torch.no_grad():
                val_pred = student_model(data_tensors['X_test'])
                val_loss = criterion(val_pred, data_tensors['y_test']).item()
            
            history['train_loss'].append(combined_loss.item())
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {combined_loss.item():.4f}, Val Loss: {val_loss:.4f}")
                
        return student_model, history

    def save_student_model(self, student_model: StudentModel, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, 'student_model.pth')
        torch.save(student_model.state_dict(), model_path)

        scaler_path = os.path.join(folder_path, 'student_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

        model_params = {
            'input_dim': student_model.input_dim,
            'hidden_dims': student_model.hidden_dims,
        }
        params_path = os.path.join(folder_path, 'student_params.json')
        with open(params_path, 'w') as f:
            json.dump(model_params, f)
            
        logger.info(f"Student model saved to {folder_path}") 