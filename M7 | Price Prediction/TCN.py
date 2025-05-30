import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Base class for price prediction models
class BasePricePredictor:
    def __init__(self, patience: int = 10):
        # Initialize MinMaxScaler to normalize price data between 0 and 1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.patience = patience  # For early stopping
        
    def prepare_data(self, data: pd.DataFrame, seq_len: int = 60, test_size: float = 0.2) -> Tuple:
        # Scale the price data
        scaled_data = self.scaler.fit_transform(data['price'].values.reshape(-1, 1))
        X, y = [], []

        # Create sequences of length seq_len for time series prediction
        # Each X is a sequence of prices, and y is the next price to predict
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i, 0])
            y.append(scaled_data[i, 0])

        # Convert to PyTorch tensors and add dimension for model input
        X, y = torch.tensor(X).float(), torch.tensor(y).float()
        X = X.unsqueeze(-1)
        
        # Split into training and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
        
    def predict(self, model, X_test) -> np.ndarray:
        # Make predictions in evaluation mode
        model.eval()
        with torch.no_grad():
            predictions = model(X_test).squeeze().numpy()
        # Convert predictions back to original scale
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        return predictions
        
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        # Get predictions
        y_pred = self.predict(model, X_test)
        
        # Convert y_test to original scale
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
    def plot_predictions(self, y_true, y_pred, title: str = "Model Predictions vs Actual") -> None:
        """Plot predictions against actual values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual Prices')
        plt.plot(y_pred, label='Predicted Prices')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# TransformerPricePredictor: Model for price prediction using Transformer architecture
class TransformerPricePredictor(BasePricePredictor):
    # Nested Transformer model class
    class TransformerModel(nn.Module):
        def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3):
            super().__init__()
            # Transformer architecture components
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
            # Final linear layer for prediction
            self.fc_out = nn.Linear(d_model, 1)
            self.d_model = d_model

        def forward(self, x):
            # Scale input by sqrt(d_model) as per transformer paper
            x = x * (self.d_model ** 0.5)
            # Pass through encoder and decoder
            encoded = self.transformer_encoder(x)
            decoded = self.transformer_decoder(encoded, encoded)
            # Use last timestep for final prediction
            out = self.fc_out(decoded[:, -1])
            return out

    def build_and_train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        # Initialize model, loss function, and optimizer
        model = self.TransformerModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create DataLoader for batch processing
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # For early stopping
        best_val_loss = float('inf')
        no_improvement = 0
        
        # If validation data is not provided, use a portion of training data
        if X_val is None or y_val is None:
            val_size = int(0.1 * len(X_train))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batches = 0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
                
            avg_train_loss = total_loss / batches
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()
                
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
        
        return model
    
    def distill_from_teacher(self, teacher_model, X_train, y_train, alpha=0.5, epochs=30, batch_size=32):
        """Train model through knowledge distillation from a teacher model"""
        student_model = self.TransformerModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        
        # Create DataLoader for batch processing
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get teacher predictions
        teacher_model.eval()
        with torch.no_grad():
            teacher_preds = teacher_model(X_train)
        
        # Training loop with distillation
        for epoch in range(epochs):
            student_model.train()
            for i, (X_batch, y_batch) in enumerate(loader):
                optimizer.zero_grad()
                
                # Get batch teacher predictions
                with torch.no_grad():
                    teacher_batch_preds = teacher_model(X_batch)
                
                # Student predictions
                student_preds = student_model(X_batch)
                
                # Combined loss: alpha * student_loss + (1-alpha) * distillation_loss
                student_loss = criterion(student_preds, y_batch)
                distillation_loss = criterion(student_preds, teacher_batch_preds)
                loss = alpha * student_loss + (1-alpha) * distillation_loss
                
                loss.backward()
                optimizer.step()
                
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
            
        return student_model

# TCN (Temporal Convolutional Network) for price prediction
class TCNPricePredictor(BasePricePredictor):
    # TCN model class
    class TCNModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.2):
            super().__init__()
            layers = []
            # Create dilated causal convolutions with proper activations and dropout
            for i in range(num_layers):
                dilation = 2 ** i  # Exponential dilation growth
                # Causal padding: only pad from the left side
                padding = (kernel_size - 1) * dilation
                
                # First layer takes the input dimension
                in_channels = input_dim if i == 0 else hidden_dim
                
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, hidden_dim, kernel_size, 
                                 padding=padding, dilation=dilation),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
            
            # Combine layers
            self.network = nn.ModuleList(layers)
            self.final_layer = nn.Linear(hidden_dim, 1)
            self.hidden_dim = hidden_dim
            
        def forward(self, x):
            # Reshape input for 1D convolution [batch, channel, sequence_length]
            x = x.permute(0, 2, 1)
            
            # Apply TCN layers
            for i, layer in enumerate(self.network):
                # Apply current layer
                residual = x if i > 0 else 0  # For residual connections after first layer
                x = layer(x)
                
                # Add residual connection if dimensions match
                if i > 0 and x.size(1) == residual.size(1):
                    x = x + residual
            
            # Take the last time step for prediction
            x = x[:, :, -1]
            return self.final_layer(x)

    def build_and_train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        # Initialize model, loss function, and optimizer
        model = self.TCNModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create DataLoader for batch processing
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # For early stopping
        best_val_loss = float('inf')
        no_improvement = 0
        
        # If validation data is not provided, use a portion of training data
        if X_val is None or y_val is None:
            val_size = int(0.1 * len(X_train))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batches = 0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
                
            avg_train_loss = total_loss / batches
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds.squeeze(), y_val).item()
                
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
        
        return model
        
    def distill_from_teacher(self, teacher_model, X_train, y_train, alpha=0.5, epochs=30, batch_size=32):
        """Train model through knowledge distillation from a teacher model"""
        student_model = self.TCNModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        
        # Create DataLoader for batch processing
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get teacher predictions
        teacher_model.eval()
        with torch.no_grad():
            teacher_preds = teacher_model(X_train)
        
        # Training loop with distillation
        for epoch in range(epochs):
            student_model.train()
            for i, (X_batch, y_batch) in enumerate(loader):
                optimizer.zero_grad()
                
                # Get batch teacher predictions
                with torch.no_grad():
                    teacher_batch_preds = teacher_model(X_batch)
                
                # Student predictions
                student_preds = student_model(X_batch)
                
                # Combined loss: alpha * student_loss + (1-alpha) * distillation_loss
                student_loss = criterion(student_preds.squeeze(), y_batch)
                distillation_loss = criterion(student_preds.squeeze(), teacher_batch_preds.squeeze())
                loss = alpha * student_loss + (1-alpha) * distillation_loss
                
                loss.backward()
                optimizer.step()
                
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
            
        return student_model

# Factory function to create the appropriate model
def create_price_predictor(model_type: str = 'tcn') -> BasePricePredictor:
    """Factory function to create a price predictor model
    
    Parameters:
    -----------
    model_type : str
        Type of model to create ('tcn' or 'transformer')
        
    Returns:
    --------
    BasePricePredictor
        An instance of the appropriate model class
    """
    if model_type.lower() == 'tcn':
        return TCNPricePredictor()
    elif model_type.lower() == 'transformer':
        return TransformerPricePredictor()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'tcn' or 'transformer'")

# Example usage
if __name__ == "__main__":
    # Sample code to demonstrate usage
    try:
        # Load your financial data
        # This is just a placeholder - replace with your actual data loading code
        data = pd.read_csv('example_stock_data.csv')
        
        # Create model using factory function
        model_type = 'tcn'  # or 'transformer'
        predictor = create_price_predictor(model_type)
        
        # Prepare data
        X_train, y_train, X_test, y_test = predictor.prepare_data(data, seq_len=60)
        
        print(f"Training {model_type.upper()} model...")
        # Train model
        trained_model = predictor.build_and_train_model(
            X_train, y_train, 
            epochs=50, 
            batch_size=32
        )
        
        # Evaluate model
        metrics = predictor.evaluate_model(trained_model, X_test, y_test)
        print("\nModel Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Make predictions
        predictions = predictor.predict(trained_model, X_test)
        
        # Optional: Plot predictions vs actual
        actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
        predictor.plot_predictions(actual, predictions)
        
    except Exception as e:
        print(f"Error in example: {e}")
        print("Note: This example requires actual data to run. Replace example_stock_data.csv with your data file.")
    


    
