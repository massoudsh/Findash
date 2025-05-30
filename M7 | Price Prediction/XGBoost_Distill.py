import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import pandas as pd
import numpy as np
import logging
import os
import json
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostTeacher:
    """Teacher model using XGBoost for regression tasks."""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize the XGBoost teacher model.
        
        Args:
            params: Dictionary of XGBoost parameters
        """
        self.default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'gamma': 0,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        self.params = self.default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            eval_set: Optional[List] = None, early_stopping_rounds: int = 50) -> 'XGBoostTeacher':
        """Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target values
            eval_set: Optional evaluation set for early stopping
            early_stopping_rounds: Number of rounds for early stopping
            
        Returns:
            Self instance for method chaining
        """
        logger.info("Training XGBoost teacher model...")
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = xgb.XGBRegressor(**self.params)
        
        if eval_set:
            # Scale evaluation features
            eval_set_scaled = [(self.scaler.transform(eval_set[0][0]), eval_set[0][1])]
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set_scaled,
                early_stopping_rounds=early_stopping_rounds,
                verbose=True
            )
        else:
            self.model.fit(X_train_scaled, y_train)
            
        logger.info("XGBoost teacher model training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained, call fit() first")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, folder_path: str) -> None:
        """Save the model and scaler to disk.
        
        Args:
            folder_path: Directory to save model files
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        model_path = os.path.join(folder_path, 'xgboost_teacher.json')
        self.model.save_model(model_path)
        
        # Save scaler
        import joblib
        scaler_path = os.path.join(folder_path, 'xgboost_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save parameters
        params_path = os.path.join(folder_path, 'xgboost_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.params, f)
            
        logger.info(f"XGBoost teacher model saved to {folder_path}")
        
    @classmethod
    def load(cls, folder_path: str) -> 'XGBoostTeacher':
        """Load a saved model from disk.
        
        Args:
            folder_path: Directory containing model files
            
        Returns:
            Loaded model instance
        """
        import joblib
        
        model_path = os.path.join(folder_path, 'xgboost_teacher.json')
        scaler_path = os.path.join(folder_path, 'xgboost_scaler.joblib')
        params_path = os.path.join(folder_path, 'xgboost_params.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, params_path]):
            raise FileNotFoundError(f"Model files not found in {folder_path}")
            
        # Load parameters
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        # Create instance and load model components
        instance = cls(params)
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(model_path)
        instance.scaler = joblib.load(scaler_path)
        
        logger.info(f"XGBoost teacher model loaded from {folder_path}")
        return instance
    
    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance from the model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained, call fit() first")
            
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained, call fit() first")
            
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

class StudentModel(nn.Module):
    """Student neural network model for knowledge distillation."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        """Initialize the student model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(StudentModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DistillationTrainer:
    """Trainer for knowledge distillation from teacher to student."""
    
    def __init__(self, teacher_model: Any, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the distillation trainer.
        
        Args:
            teacher_model: Trained teacher model
            device: Device to use for PyTorch computations
        """
        self.teacher_model = teacher_model
        self.device = device
        self.scaler = StandardScaler()
        logger.info(f"Using device: {device}")
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    valid_size: float = 0.1, random_state: int = 42) -> Dict[str, torch.Tensor]:
        """Prepare data for training and evaluation.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data for testing
            valid_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of data tensors
        """
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Further split into train and validation
        valid_ratio = valid_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=valid_ratio, random_state=random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get teacher predictions
        y_train_teacher = self.teacher_model.predict(X_train)
        y_val_teacher = self.teacher_model.predict(X_val)
        y_test_teacher = self.teacher_model.predict(X_test)
        
        # Convert to PyTorch tensors
        tensors = {
            'X_train': torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device),
            'y_train': torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device),
            'y_train_teacher': torch.tensor(y_train_teacher.reshape(-1, 1), dtype=torch.float32).to(self.device),
            
            'X_val': torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device),
            'y_val': torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device),
            'y_val_teacher': torch.tensor(y_val_teacher.reshape(-1, 1), dtype=torch.float32).to(self.device),
            
            'X_test': torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device),
            'y_test': torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(self.device),
            'y_test_teacher': torch.tensor(y_test_teacher.reshape(-1, 1), dtype=torch.float32).to(self.device),
        }
        
        logger.info(f"Data prepared: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test samples")
        return tensors
    
    def train_student(self, data_tensors: Dict[str, torch.Tensor], 
                      student_model: StudentModel, 
                      alpha: float = 0.5, 
                      epochs: int = 100, 
                      batch_size: int = 64,
                      learning_rate: float = 0.001,
                      patience: int = 10) -> Tuple[StudentModel, Dict]:
        """Train the student model through knowledge distillation.
        
        Args:
            data_tensors: Dictionary of data tensors
            student_model: Untrained student model
            alpha: Weight for balancing student loss vs distillation loss
                  (alpha=1 means only student loss, alpha=0 means only distillation loss)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            patience: Patience for early stopping
            
        Returns:
            Trained student model and training history
        """
        student_model = student_model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
        
        # For tracking metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_student_loss': [],
            'train_distill_loss': [],
            'val_student_loss': [],
            'val_distill_loss': []
        }
        
        # For early stopping
        best_val_loss = float('inf')
        no_improvement = 0
        best_model_state = None
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            data_tensors['X_train'], 
            data_tensors['y_train'], 
            data_tensors['y_train_teacher']
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        logger.info(f"Starting student model training for {epochs} epochs")
        logger.info(f"Student model has {student_model.count_parameters()} parameters")
        logger.info(f"Distillation alpha: {alpha}")
        
    for epoch in range(epochs):
            # Training
            student_model.train()
            train_losses = []
            train_student_losses = []
            train_distill_losses = []
            
            for X_batch, y_batch, teacher_pred_batch in train_loader:
        optimizer.zero_grad()
                
                # Student predictions
                student_pred = student_model(X_batch)
                
                # Calculate losses
                student_loss = criterion(student_pred, y_batch)
                distill_loss = criterion(student_pred, teacher_pred_batch)
                combined_loss = alpha * student_loss + (1 - alpha) * distill_loss
                
                combined_loss.backward()
        optimizer.step()

                train_losses.append(combined_loss.item())
                train_student_losses.append(student_loss.item())
                train_distill_losses.append(distill_loss.item())
            
            # Validation
            student_model.eval()
            with torch.no_grad():
                val_pred = student_model(data_tensors['X_val'])
                val_student_loss = criterion(val_pred, data_tensors['y_val']).item()
                val_distill_loss = criterion(val_pred, data_tensors['y_val_teacher']).item()
                val_loss = alpha * val_student_loss + (1 - alpha) * val_distill_loss
            
            # Update history
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_student_loss'].append(np.mean(train_student_losses))
            history['train_distill_loss'].append(np.mean(train_distill_losses))
            history['val_student_loss'].append(val_student_loss)
            history['val_distill_loss'].append(val_distill_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                best_model_state = student_model.state_dict().copy()
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    student_model.load_state_dict(best_model_state)
                    break
        
        if best_model_state:
            student_model.load_state_dict(best_model_state)
            
        logger.info("Student model training completed")
        return student_model, history
    
    def evaluate_student(self, student_model: StudentModel, 
                         data_tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the student model on test data.
        
        Args:
            student_model: Trained student model
            data_tensors: Dictionary of data tensors
            
        Returns:
            Dictionary of evaluation metrics
        """
        student_model.eval()
        with torch.no_grad():
            y_pred = student_model(data_tensors['X_test']).cpu().numpy()
            y_true = data_tensors['y_test'].cpu().numpy()
            y_teacher = data_tensors['y_test_teacher'].cpu().numpy()
        
        # Calculate metrics vs ground truth
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate metrics vs teacher
        teacher_mse = mean_squared_error(y_teacher, y_pred)
        teacher_mae = mean_absolute_error(y_teacher, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'teacher_mse': teacher_mse,
            'teacher_mae': teacher_mae
        }
    
    def save_student_model(self, student_model: StudentModel, 
                           folder_path: str, model_name: str = 'student_model') -> None:
        """Save the student model to disk.
        
        Args:
            student_model: Trained student model
            folder_path: Directory to save the model
            model_name: Name prefix for saved files
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Save model
        model_path = os.path.join(folder_path, f'{model_name}.pt')
        torch.save(student_model.state_dict(), model_path)
        
        # Save model architecture info
        architecture = {
            'input_dim': student_model.input_dim,
            'hidden_dims': student_model.hidden_dims
        }
        architecture_path = os.path.join(folder_path, f'{model_name}_architecture.json')
        with open(architecture_path, 'w') as f:
            json.dump(architecture, f)
        
        # Save scaler
        import joblib
        scaler_path = os.path.join(folder_path, f'{model_name}_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Student model saved to {folder_path}")
    
    @staticmethod
    def load_student_model(folder_path: str, model_name: str = 'student_model', 
                          device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[StudentModel, Any]:
        """Load a saved student model from disk.
        
        Args:
            folder_path: Directory containing model files
            model_name: Name prefix for saved files
            device: Device to load the model to
            
        Returns:
            Loaded model and scaler
        """
        import joblib
        
        model_path = os.path.join(folder_path, f'{model_name}.pt')
        architecture_path = os.path.join(folder_path, f'{model_name}_architecture.json')
        scaler_path = os.path.join(folder_path, f'{model_name}_scaler.joblib')
        
        if not all(os.path.exists(p) for p in [model_path, architecture_path, scaler_path]):
            raise FileNotFoundError(f"Model files not found in {folder_path}")
        
        # Load architecture
        with open(architecture_path, 'r') as f:
            architecture = json.load(f)
            
        # Create and load model
        model = StudentModel(
            input_dim=architecture['input_dim'],
            hidden_dims=architecture['hidden_dims']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Student model loaded from {folder_path}")
        return model, scaler
    
    def plot_training_history(self, history: Dict) -> None:
        """Plot training metrics history.
        
        Args:
            history: Dictionary of training metrics
        """
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Combined Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot component losses
        plt.subplot(2, 1, 2)
        plt.plot(history['train_student_loss'], label='Train Student Loss')
        plt.plot(history['train_distill_loss'], label='Train Distillation Loss')
        plt.plot(history['val_student_loss'], label='Val Student Loss')
        plt.plot(history['val_distill_loss'], label='Val Distillation Loss')
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_comparison(self, student_model: StudentModel, 
                                 data_tensors: Dict[str, torch.Tensor], 
                                 n_samples: int = 200) -> None:
        """Plot comparison between actual values, teacher predictions, and student predictions.
        
        Args:
            student_model: Trained student model
            data_tensors: Dictionary of data tensors
            n_samples: Number of samples to plot
        """
        student_model.eval()
        with torch.no_grad():
            student_preds = student_model(data_tensors['X_test'][:n_samples]).cpu().numpy()
            
        y_true = data_tensors['y_test'][:n_samples].cpu().numpy()
        teacher_preds = data_tensors['y_test_teacher'][:n_samples].cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual Values')
        plt.plot(teacher_preds, label='Teacher Predictions')
        plt.plot(student_preds, label='Student Predictions')
        plt.title('Prediction Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

class RealTimeInference:
    """Class for real-time financial data inference using the trained student model."""
    
    def __init__(self, model: torch.nn.Module, scaler: Any, api_key: str, 
                feature_length: int, device: str = 'cpu'):
        """Initialize the real-time inference system.
        
        Args:
            model: Trained model for inference
            scaler: Feature scaler
            api_key: API key for financial data service
            feature_length: Number of features expected by the model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.api_key = api_key
        self.feature_length = feature_length
        self.device = device
        
    def fetch_data(self, symbol: str) -> Optional[torch.Tensor]:
        """Fetch real-time data for a symbol.
        
        Args:
            symbol: Stock symbol to fetch data for
            
        Returns:
            Tensor of processed features or None on error
        """
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                return None
                
            data = response.json()
            
            # Check for error messages or missing data
            if 'Error Message' in data:
                logger.error(f"API error: {data['Error Message']}")
                return None
                
            if 'Time Series (1min)' not in data:
                logger.error("Missing time series data in API response")
                return None
                
            # Extract closing prices
            time_series = data['Time Series (1min)']
            closes = [float(v['4. close']) for k, v in sorted(time_series.items(), reverse=True)]
            
            if len(closes) < self.feature_length:
                logger.warning(f"Insufficient data points: {len(closes)} < {self.feature_length}")
                return None
                
            # Process data
            features = np.array(closes[:self.feature_length]).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            
            return torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
        except ValueError as e:
            logger.error(f"Value error processing data: {e}")
        except KeyError as e:
            logger.error(f"Key error in response data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            
        return None
    
    def run(self, symbol: str, interval: int = 60, max_iterations: Optional[int] = None) -> None:
        """Run real-time inference.
        
        Args:
            symbol: Stock symbol to predict
            interval: Seconds between predictions
            max_iterations: Maximum number of iterations (None for infinite)
        """
        iteration = 0
        predictions = []
        timestamps = []
        
        logger.info(f"Starting real-time inference for {symbol}")
        try:
            while True:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                    
                logger.info(f"Fetching data for {symbol}, iteration {iteration+1}")
                features = self.fetch_data(symbol)
                
                if features is not None:
                    with torch.no_grad():
                        prediction = self.model(features).cpu().item()
                    
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"Prediction at {timestamp}: {prediction:.4f}")
                    
                    # Store for potential plotting
                    predictions.append(prediction)
                    timestamps.append(timestamp)
                    
                    # Optional: save to CSV or database here
                    
                else:
                    logger.warning("Failed to get valid data, skipping prediction")
                
                iteration += 1
                
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Waiting {interval} seconds until next prediction")
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            logger.info("Inference stopped by user")
        
        # Optional: create a final plot of predictions
        if predictions:
            plt.figure(figsize=(12, 6))
            plt.plot(predictions)
            plt.title(f"Real-time Predictions for {symbol}")
            plt.xlabel("Iteration")
            plt.ylabel("Predicted Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Main execution function
def main():
    """Main function for demonstrating the XGBoost distillation pipeline."""
    try:
    # Load your dataset (replace with your actual dataset)
        logger.info("Loading financial data")
        data_path = 'financial_data.csv'  # Replace with your data path
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            logger.info("Using synthetic data for demonstration")
            
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            X = np.random.randn(n_samples, n_features)
            y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1.0 * X[:, 2] + 0.5 * np.random.randn(n_samples)
            
            feature_names = [f'feature_{i}' for i in range(n_features)]
            data = pd.DataFrame(X, columns=feature_names)
            data['target'] = y
        else:
            # Load actual data
            data = pd.read_csv(data_path)

    # Prepare features and target
        if 'target' not in data.columns:
            logger.error("No 'target' column found in the data")
            return
            
        X = data.drop(['target'], axis=1).values
        y = data['target'].values
        
        # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 1: Train Teacher Model
        logger.info("Training XGBoost teacher model")
        teacher = XGBoostTeacher()
        teacher.fit(X_train, y_train)
        
        # Evaluate teacher model
        teacher_metrics = teacher.evaluate(X_test, y_test)
        logger.info("Teacher model performance:")
        for metric, value in teacher_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        # Step 2: Prepare for distillation
        logger.info("Setting up distillation trainer")
        distiller = DistillationTrainer(teacher.model)
        data_tensors = distiller.prepare_data(X, y)
        
        # Step 3: Create and train the student model
        logger.info("Training student model via distillation")
        student_model = StudentModel(
            input_dim=X.shape[1],
            hidden_dims=[64, 32],
            dropout=0.2
        )
        
        trained_student, history = distiller.train_student(
            data_tensors=data_tensors,
            student_model=student_model,
            alpha=0.5,  # Balance between student and distillation loss
            epochs=100,
            batch_size=64,
            patience=10
        )
        
        # Step 4: Evaluate student model
        logger.info("Evaluating student model")
        student_metrics = distiller.evaluate_student(trained_student, data_tensors)
        
        logger.info("Student model performance:")
        for metric, value in student_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        # Compare teacher and student models
        logger.info("\nModel comparison:")
        logger.info(f"  Teacher RMSE: {teacher_metrics['rmse']:.4f}")
        logger.info(f"  Student RMSE: {student_metrics['rmse']:.4f}")
        logger.info(f"  Student-Teacher MSE: {student_metrics['teacher_mse']:.4f}")
        
        # Save models
        output_dir = 'output/models'
        logger.info(f"Saving models to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        teacher.save(os.path.join(output_dir, 'teacher'))
        distiller.save_student_model(trained_student, os.path.join(output_dir, 'student'))
        
        # Plot training history
        distiller.plot_training_history(history)
        
        # Plot prediction comparison
        distiller.plot_prediction_comparison(trained_student, data_tensors)
        
        # Optionally demonstrate real-time inference
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'your_alpha_vantage_api_key')
        if api_key == 'your_alpha_vantage_api_key':
            logger.warning("No API key provided for Alpha Vantage. Skipping real-time inference.")
        else:
            logger.info("Demonstrating real-time inference")
            inference = RealTimeInference(
                model=trained_student,
                scaler=distiller.scaler,
                api_key=api_key,
                feature_length=X.shape[1],
                device='cpu'
            )
            
            # Run for just 3 iterations as a demonstration
            inference.run(symbol='AAPL', interval=5, max_iterations=3)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

    