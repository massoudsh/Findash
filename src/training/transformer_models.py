"""
M5 - Deep Learning Models Agent
Advanced neural networks for time series prediction and pattern recognition.
Includes Transformers, TCN, and AutoEncoders.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from ..core.cache import TradingCache
from ..core.exceptions import MLModelError, TradingError


@dataclass
class ModelPrediction:
    """Model prediction with confidence intervals."""
    symbol: str
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    confidence_score: float
    model_name: str
    timestamp: datetime
    features_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "predicted_price": self.predicted_price,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "confidence_score": self.confidence_score,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "features_used": self.features_used
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TradingTransformer(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 output_dim: int = 1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch_size, input_dim)
        src = self.input_projection(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        output = self.transformer(src, src_mask)
        
        # Use the last output for prediction
        prediction = self.output_projection(output[-1])
        
        return prediction


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time series."""
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 num_channels: List[int] = [64, 64, 64],
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Temporal convolutional layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                           stride=1, dilation=dilation_size,
                           padding=(kernel_size-1) * dilation_size)
            
            # Add components
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Residual connection
            if in_channels != out_channels:
                layers.append(nn.Conv1d(in_channels, out_channels, 1))
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        output = self.network(x)
        
        # Global average pooling
        output = output.mean(dim=-1)
        
        return self.output_layer(output)


class TradingAutoEncoder(nn.Module):
    """AutoEncoder for anomaly detection and feature learning."""
    
    def __init__(self, 
                 input_dim: int,
                 encoding_dims: List[int] = [64, 32, 16],
                 dropout: float = 0.1):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        encoding_dims_reversed = list(reversed(encoding_dims[:-1])) + [input_dim]
        
        for dim in encoding_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Identity(),
                nn.Dropout(dropout) if dim != input_dim else nn.Identity()
            ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Latent dimension
        self.latent_dim = encoding_dims[-1]
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, data: np.ndarray, sequence_length: int, prediction_horizon: int = 1):
        self.data = data
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class DeepLearningAgent:
    """
    M5 - Deep Learning Models Agent
    Manages multiple neural network models for trading predictions.
    """
    
    def __init__(self, cache: TradingCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Training configurations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 60  # 60 time steps
        self.prediction_horizon = 1
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize default models
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize default neural network models."""
        try:
            # Default feature dimension (OHLCV + technical indicators)
            input_dim = 20
            
            # Initialize Transformer model
            transformer_model = TradingTransformer(
                input_dim=input_dim,
                d_model=128,
                nhead=8,
                num_layers=4,
                dim_feedforward=256,
                dropout=0.1,
                output_dim=1
            ).to(self.device)
            
            self.models['transformer'] = transformer_model
            self.scalers['transformer'] = StandardScaler()
            self.model_metadata['transformer'] = {
                'type': 'transformer',
                'input_dim': input_dim,
                'trained': False,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Initialize TCN model
            tcn_model = TemporalConvNet(
                input_size=input_dim,
                output_size=1,
                num_channels=[64, 64, 64],
                kernel_size=3,
                dropout=0.2
            ).to(self.device)
            
            self.models['tcn'] = tcn_model
            self.scalers['tcn'] = StandardScaler()
            self.model_metadata['tcn'] = {
                'type': 'tcn',
                'input_dim': input_dim,
                'trained': False,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Initialize AutoEncoder
            autoencoder_model = TradingAutoEncoder(
                input_dim=input_dim,
                encoding_dims=[64, 32, 16],
                dropout=0.1
            ).to(self.device)
            
            self.models['autoencoder'] = autoencoder_model
            self.scalers['autoencoder'] = StandardScaler()
            self.model_metadata['autoencoder'] = {
                'type': 'autoencoder',
                'input_dim': input_dim,
                'trained': False,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            self.logger.info("Initialized deep learning models: transformer, tcn, autoencoder")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise MLModelError(f"Failed to initialize models: {e}")
    
    async def prepare_training_data(self, symbol: str, timeframe: str = "1h") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare training data from market data and technical indicators."""
        try:
            # Get market data
            cache_key = f"market_data:{symbol}:{timeframe}"
            market_data = await self.cache.get(cache_key)
            
            if not market_data:
                self.logger.warning(f"No market data available for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            
            if len(df) < self.sequence_length + 50:  # Need enough data
                self.logger.warning(f"Insufficient data for training: {len(df)} rows")
                return None
            
            # Feature engineering
            features_df = await self._engineer_features(df)
            
            # Prepare sequences
            X, y = await self._create_sequences(features_df)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for {symbol}: {e}")
            raise MLModelError(f"Failed to prepare training data: {e}")
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for deep learning models."""
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = df['close']  # Fallback
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_pct'] = (df['high'] - df['low']) / df['close']
        df['oc_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
        # Select final features
        feature_columns = [
            'returns', 'log_returns', 'hl_pct', 'oc_pct',
            'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
            'rsi', 'bb_position', 'volume_ratio', 'volatility', 'volatility_ratio'
        ]
        
        # Add price levels (normalized)
        price_features = ['open', 'high', 'low', 'close']
        for col in price_features:
            df[f'{col}_norm'] = df[col] / df['close'].rolling(20).mean() - 1
            feature_columns.append(f'{col}_norm')
        
        # Handle missing values
        features_df = df[feature_columns].fillna(method='ffill').fillna(0)
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    async def _create_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        
        data = features_df.values
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length])
            
            # Target (next price return)
            target_idx = i + self.sequence_length
            if target_idx < len(data):
                # Predict next return
                y.append(data[target_idx, 0])  # First column is returns
            
        return np.array(X), np.array(y)
    
    async def train_model(self, model_name: str, symbol: str, epochs: int = 100) -> bool:
        """Train a specific model."""
        try:
            if model_name not in self.models:
                raise MLModelError(f"Model {model_name} not found")
            
            # Prepare training data
            X, y = await self.prepare_training_data(symbol)
            
            if X is None or len(X) < 100:
                self.logger.warning(f"Insufficient training data for {model_name}")
                return False
            
            # Scale data
            scaler = self.scalers[model_name]
            
            # Reshape for scaling
            original_shape = X.shape
            X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(original_shape)
            
            # Split data
            split_idx = int(0.8 * len(X_scaled))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1),
                self.sequence_length
            )
            val_dataset = TimeSeriesDataset(
                np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1),
                self.sequence_length
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train the model
            model = self.models[model_name]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if model_name == 'transformer':
                        # Transformer expects (seq_len, batch_size, features)
                        batch_X = batch_X.transpose(0, 1)
                        outputs = model(batch_X)
                    elif model_name == 'autoencoder':
                        # AutoEncoder reconstruction
                        batch_X_flat = batch_X.view(batch_X.size(0), -1)
                        outputs, _ = model(batch_X_flat)
                        batch_y = batch_X_flat  # Reconstruction target
                    else:  # TCN
                        outputs = model(batch_X)
                    
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        if model_name == 'transformer':
                            batch_X = batch_X.transpose(0, 1)
                            outputs = model(batch_X)
                        elif model_name == 'autoencoder':
                            batch_X_flat = batch_X.view(batch_X.size(0), -1)
                            outputs, _ = model(batch_X_flat)
                            batch_y = batch_X_flat
                        else:  # TCN
                            outputs = model(batch_X)
                        
                        loss = criterion(outputs.squeeze(), batch_y.squeeze())
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'models/{model_name}_{symbol}_best.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
            
            # Update metadata
            self.model_metadata[model_name]['trained'] = True
            self.model_metadata[model_name]['last_updated'] = datetime.utcnow().isoformat()
            self.model_metadata[model_name]['best_val_loss'] = best_val_loss
            
            self.logger.info(f"Training completed for {model_name}. Best validation loss: {best_val_loss:.6f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")
            return False
    
    async def predict(self, model_name: str, symbol: str, timeframe: str = "1h") -> Optional[ModelPrediction]:
        """Generate prediction using a specific model."""
        try:
            if model_name not in self.models:
                raise MLModelError(f"Model {model_name} not found")
            
            if not self.model_metadata[model_name].get('trained', False):
                self.logger.warning(f"Model {model_name} is not trained")
                return None
            
            # Get recent data
            cache_key = f"market_data:{symbol}:{timeframe}"
            market_data = await self.cache.get(cache_key)
            
            if not market_data:
                return None
            
            # Prepare features
            df = pd.DataFrame(market_data)
            features_df = await self._engineer_features(df)
            
            if len(features_df) < self.sequence_length:
                return None
            
            # Get last sequence
            last_sequence = features_df.iloc[-self.sequence_length:].values
            
            # Scale data
            scaler = self.scalers[model_name]
            last_sequence_scaled = scaler.transform(last_sequence.reshape(1, -1)).reshape(self.sequence_length, -1)
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            # Generate prediction
            model = self.models[model_name]
            model.eval()
            
            with torch.no_grad():
                if model_name == 'transformer':
                    input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch_size, features)
                    prediction = model(input_tensor)
                elif model_name == 'autoencoder':
                    # For autoencoder, use reconstruction error for anomaly detection
                    input_flat = input_tensor.view(1, -1)
                    reconstructed, encoded = model(input_flat)
                    reconstruction_error = F.mse_loss(reconstructed, input_flat).item()
                    # Convert error to prediction signal
                    prediction = torch.tensor([[reconstruction_error]])
                else:  # TCN
                    prediction = model(input_tensor)
            
            predicted_return = prediction.squeeze().cpu().numpy()
            
            # Convert return to price prediction
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence intervals (simplified)
            confidence_score = min(0.95, max(0.05, 1.0 / (1.0 + abs(predicted_return))))
            price_std = current_price * 0.02  # 2% standard deviation
            
            confidence_lower = predicted_price - 1.96 * price_std
            confidence_upper = predicted_price + 1.96 * price_std
            
            # Create prediction object
            prediction_obj = ModelPrediction(
                symbol=symbol,
                predicted_price=float(predicted_price),
                confidence_lower=float(confidence_lower),
                confidence_upper=float(confidence_upper),
                confidence_score=float(confidence_score),
                model_name=model_name,
                timestamp=datetime.utcnow(),
                features_used=features_df.columns.tolist()
            )
            
            # Cache prediction
            cache_key = f"ml_prediction:{model_name}:{symbol}:{timeframe}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, prediction_obj.to_dict(), ttl=1800)  # 30 minutes
            
            return prediction_obj
            
        except Exception as e:
            self.logger.error(f"Error generating prediction with {model_name}: {e}")
            return None
    
    async def ensemble_predict(self, symbol: str, timeframe: str = "1h") -> Optional[ModelPrediction]:
        """Generate ensemble prediction from all trained models."""
        try:
            predictions = []
            
            for model_name in self.models.keys():
                if self.model_metadata[model_name].get('trained', False):
                    pred = await self.predict(model_name, symbol, timeframe)
                    if pred:
                        predictions.append(pred)
            
            if not predictions:
                return None
            
            # Ensemble methods
            weights = []
            predicted_prices = []
            confidence_scores = []
            
            for pred in predictions:
                # Weight by inverse of validation loss (if available)
                val_loss = self.model_metadata[pred.model_name].get('best_val_loss', 1.0)
                weight = 1.0 / (1.0 + val_loss)
                
                weights.append(weight)
                predicted_prices.append(pred.predicted_price)
                confidence_scores.append(pred.confidence_score)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted average prediction
            ensemble_price = sum(w * p for w, p in zip(weights, predicted_prices))
            ensemble_confidence = sum(w * c for w, c in zip(weights, confidence_scores))
            
            # Calculate ensemble confidence intervals
            price_variance = sum(w * (p - ensemble_price) ** 2 for w, p in zip(weights, predicted_prices))
            price_std = np.sqrt(price_variance)
            
            confidence_lower = ensemble_price - 1.96 * price_std
            confidence_upper = ensemble_price + 1.96 * price_std
            
            # Create ensemble prediction
            ensemble_prediction = ModelPrediction(
                symbol=symbol,
                predicted_price=float(ensemble_price),
                confidence_lower=float(confidence_lower),
                confidence_upper=float(confidence_upper),
                confidence_score=float(ensemble_confidence),
                model_name="ensemble",
                timestamp=datetime.utcnow(),
                features_used=predictions[0].features_used  # Use features from first model
            )
            
            # Cache ensemble prediction
            cache_key = f"ml_prediction:ensemble:{symbol}:{timeframe}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, ensemble_prediction.to_dict(), ttl=1800)
            
            return ensemble_prediction
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble prediction: {e}")
            return None
    
    async def detect_anomalies(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Detect anomalies using the autoencoder model."""
        try:
            if 'autoencoder' not in self.models or not self.model_metadata['autoencoder'].get('trained', False):
                return {"anomaly_detected": False, "reason": "AutoEncoder not trained"}
            
            # Get prediction with autoencoder
            prediction = await self.predict('autoencoder', symbol, timeframe)
            
            if not prediction:
                return {"anomaly_detected": False, "reason": "No prediction available"}
            
            # Reconstruction error threshold (adjust based on training data)
            threshold = 0.1  # This should be calibrated during training
            
            reconstruction_error = prediction.predicted_price  # For autoencoder, this contains the error
            
            anomaly_detected = reconstruction_error > threshold
            
            return {
                "anomaly_detected": anomaly_detected,
                "reconstruction_error": reconstruction_error,
                "threshold": threshold,
                "severity": "high" if reconstruction_error > threshold * 2 else "medium" if anomaly_detected else "low",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {"anomaly_detected": False, "error": str(e)}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the Deep Learning Agent."""
        
        return {
            "agent_id": "M5_deep_learning_agent",
            "models": {
                name: {
                    "type": metadata.get("type"),
                    "trained": metadata.get("trained", False),
                    "input_dim": metadata.get("input_dim"),
                    "last_updated": metadata.get("last_updated"),
                    "best_val_loss": metadata.get("best_val_loss")
                }
                for name, metadata in self.model_metadata.items()
            },
            "performance": self.model_performance,
            "device": str(self.device),
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "last_updated": datetime.utcnow().isoformat()
        } 