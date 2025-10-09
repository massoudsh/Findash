"""
M7 - Advanced Price Prediction Agent
Enhanced Prophet models, pattern detection, and multi-timeframe analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
import cv2
from scipy import signal
import talib

from ..core.cache import TradingCache
from ..core.exceptions import TradingError, MLModelError


@dataclass
class PatternMatch:
    """Detected pattern in price data."""
    pattern_type: str
    confidence: float
    start_idx: int
    end_idx: int
    breakout_probability: float
    target_price: Optional[float]
    timeframe: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "breakout_probability": self.breakout_probability,
            "target_price": self.target_price,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PricePrediction:
    """Comprehensive price prediction with multiple components."""
    symbol: str
    timeframe: str
    predictions: Dict[str, float]  # Different time horizons
    confidence_intervals: Dict[str, Tuple[float, float]]
    trend_direction: str  # "up", "down", "sideways"
    trend_strength: float
    pattern_signals: List[PatternMatch]
    seasonal_components: Dict[str, float]
    volatility_forecast: float
    prediction_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "predictions": self.predictions,
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "pattern_signals": [p.to_dict() for p in self.pattern_signals],
            "seasonal_components": self.seasonal_components,
            "volatility_forecast": self.volatility_forecast,
            "prediction_timestamp": self.prediction_timestamp.isoformat()
        }


class YOLOPatternDetector:
    """YOLO-inspired pattern detection for technical analysis patterns."""
    
    def __init__(self):
        self.pattern_templates = self._initialize_pattern_templates()
        self.scaler = StandardScaler()
        
    def _initialize_pattern_templates(self) -> Dict[str, np.ndarray]:
        """Initialize pattern templates for matching."""
        templates = {}
        
        # Head and Shoulders
        t = np.linspace(0, 2*np.pi, 50)
        templates['head_and_shoulders'] = np.sin(t) + 0.5 * np.sin(3*t)
        
        # Double Top
        templates['double_top'] = np.concatenate([
            np.sin(np.linspace(0, np.pi, 25)),
            np.sin(np.linspace(0, np.pi, 25))
        ])
        
        # Double Bottom (inverted double top)
        templates['double_bottom'] = -templates['double_top']
        
        # Triangle patterns
        templates['ascending_triangle'] = np.linspace(0, 0.5, 50) + 0.1 * np.sin(np.linspace(0, 4*np.pi, 50))
        templates['descending_triangle'] = np.linspace(0.5, 0, 50) + 0.1 * np.sin(np.linspace(0, 4*np.pi, 50))
        
        # Cup and Handle
        cup = np.concatenate([
            np.linspace(0, -0.5, 20),
            np.linspace(-0.5, 0, 20)
        ])
        handle = np.linspace(0, -0.2, 10)
        templates['cup_and_handle'] = np.concatenate([cup, handle])
        
        # Wedge patterns
        templates['rising_wedge'] = np.linspace(0, 0.3, 50) * (1 - np.linspace(0, 0.8, 50))
        templates['falling_wedge'] = -templates['rising_wedge']
        
        return templates
    
    async def detect_patterns(self, price_data: np.ndarray, 
                            timeframe: str = "1h") -> List[PatternMatch]:
        """Detect patterns in price data using template matching."""
        patterns = []
        
        if len(price_data) < 30:
            return patterns
        
        # Normalize price data
        normalized_prices = self.scaler.fit_transform(price_data.reshape(-1, 1)).flatten()
        
        # Sliding window pattern matching
        window_sizes = [30, 40, 50, 60]
        
        for window_size in window_sizes:
            if len(normalized_prices) < window_size:
                continue
                
            for i in range(len(normalized_prices) - window_size + 1):
                window = normalized_prices[i:i + window_size]
                
                # Match against all templates
                for pattern_name, template in self.pattern_templates.items():
                    if len(template) != window_size:
                        # Interpolate template to match window size
                        template_interp = np.interp(
                            np.linspace(0, 1, window_size),
                            np.linspace(0, 1, len(template)),
                            template
                        )
                    else:
                        template_interp = template
                    
                    # Calculate correlation
                    correlation = np.corrcoef(window, template_interp)[0, 1]
                    
                    if np.isnan(correlation):
                        continue
                    
                    # Threshold for pattern detection
                    if abs(correlation) > 0.7:  # Strong correlation
                        confidence = abs(correlation)
                        
                        # Calculate breakout probability
                        breakout_prob = await self._calculate_breakout_probability(
                            window, pattern_name, timeframe
                        )
                        
                        # Estimate target price
                        target_price = await self._estimate_target_price(
                            price_data[i:i + window_size], pattern_name
                        )
                        
                        pattern_match = PatternMatch(
                            pattern_type=pattern_name,
                            confidence=confidence,
                            start_idx=i,
                            end_idx=i + window_size - 1,
                            breakout_probability=breakout_prob,
                            target_price=target_price,
                            timeframe=timeframe,
                            timestamp=datetime.utcnow()
                        )
                        
                        patterns.append(pattern_match)
        
        # Remove overlapping patterns, keep highest confidence
        patterns = await self._remove_overlapping_patterns(patterns)
        
        return patterns
    
    async def _calculate_breakout_probability(self, window: np.ndarray, 
                                           pattern_name: str, timeframe: str) -> float:
        """Calculate probability of breakout based on pattern characteristics."""
        
        # Volume analysis (if available)
        volume_factor = 1.0  # Default
        
        # Pattern-specific breakout probabilities
        pattern_breakout_base = {
            'head_and_shoulders': 0.7,
            'double_top': 0.65,
            'double_bottom': 0.65,
            'ascending_triangle': 0.72,
            'descending_triangle': 0.68,
            'cup_and_handle': 0.75,
            'rising_wedge': 0.6,
            'falling_wedge': 0.6
        }
        
        base_prob = pattern_breakout_base.get(pattern_name, 0.5)
        
        # Adjust based on trend strength
        trend_strength = abs(window[-1] - window[0]) / len(window)
        trend_factor = min(1.2, 1.0 + trend_strength)
        
        # Adjust based on timeframe
        timeframe_factors = {
            "1m": 0.8,
            "5m": 0.85,
            "15m": 0.9,
            "1h": 1.0,
            "4h": 1.1,
            "1d": 1.2
        }
        
        timeframe_factor = timeframe_factors.get(timeframe, 1.0)
        
        # Final probability
        probability = base_prob * trend_factor * timeframe_factor * volume_factor
        
        return min(0.95, max(0.05, probability))
    
    async def _estimate_target_price(self, price_window: np.ndarray, pattern_name: str) -> Optional[float]:
        """Estimate target price based on pattern type."""
        
        if len(price_window) < 10:
            return None
        
        current_price = price_window[-1]
        pattern_height = np.max(price_window) - np.min(price_window)
        
        # Pattern-specific target calculations
        if pattern_name in ['head_and_shoulders', 'double_top']:
            # Bearish patterns - target below
            target = current_price - pattern_height
        elif pattern_name in ['double_bottom', 'cup_and_handle']:
            # Bullish patterns - target above
            target = current_price + pattern_height
        elif pattern_name == 'ascending_triangle':
            # Bullish breakout
            target = current_price + pattern_height * 0.8
        elif pattern_name == 'descending_triangle':
            # Bearish breakout
            target = current_price - pattern_height * 0.8
        else:
            # Default: 50% of pattern height
            direction = 1 if price_window[-1] > price_window[0] else -1
            target = current_price + direction * pattern_height * 0.5
        
        return float(target)
    
    async def _remove_overlapping_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping patterns, keeping the highest confidence ones."""
        
        if not patterns:
            return patterns
        
        # Sort by confidence (descending)
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_patterns = []
        
        for pattern in patterns:
            # Check if this pattern overlaps with any already selected
            overlaps = False
            
            for selected in filtered_patterns:
                # Check for overlap
                if (pattern.start_idx <= selected.end_idx and 
                    pattern.end_idx >= selected.start_idx):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_patterns.append(pattern)
        
        return filtered_patterns


class EnhancedProphetModel:
    """Enhanced Prophet model with custom seasonalities and external regressors."""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.performance_metrics = {}
        
    async def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        
        # Ensure required columns
        if 'timestamp' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' and 'close' columns")
        
        # Prepare Prophet format
        prophet_df = df[['timestamp', 'close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Add external regressors
        if 'volume' in df.columns:
            prophet_df['volume'] = df['volume']
        
        # Add technical indicators as regressors
        prophet_df['rsi'] = talib.RSI(df['close'].values)
        prophet_df['macd'], prophet_df['macd_signal'], _ = talib.MACD(df['close'].values)
        prophet_df['bb_upper'], _, prophet_df['bb_lower'] = talib.BBANDS(df['close'].values)
        
        # Add volatility regressor
        prophet_df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Add market regime indicator
        prophet_df['trend_strength'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
        
        # Fill missing values
        prophet_df = prophet_df.fillna(method='ffill').fillna(method='bfill')
        
        return prophet_df
    
    async def train(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train the Prophet model."""
        
        try:
            # Prepare data
            prophet_df = await self.prepare_data(df)
            
            # Initialize Prophet model
            self.model = Prophet(
                # Growth
                growth='linear',
                
                # Seasonality
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                
                # Holidays and events
                holidays=None,  # Can add market holidays
                
                # Uncertainty
                interval_width=0.95,
                
                # Changepoints
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                
                # Mcmc samples for uncertainty
                mcmc_samples=0  # Set to 300+ for better uncertainty estimates
            )
            
            # Add custom seasonalities
            self.model.add_seasonality(name='hourly', period=24, fourier_order=8)
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            # Add external regressors
            regressors = ['volume', 'rsi', 'macd', 'volatility', 'trend_strength']
            for regressor in regressors:
                if regressor in prophet_df.columns:
                    self.model.add_regressor(regressor)
            
            # Fit the model
            self.model.fit(prophet_df)
            self.is_fitted = True
            
            # Cross-validation for performance metrics
            if len(prophet_df) > 100:  # Only if enough data
                cv_results = cross_validation(
                    self.model, 
                    initial='30 days', 
                    period='7 days', 
                    horizon='7 days'
                )
                
                self.performance_metrics = performance_metrics(cv_results)
            
            return {
                "status": "success",
                "training_samples": len(prophet_df),
                "regressors_used": regressors,
                "performance_metrics": self.performance_metrics.to_dict() if hasattr(self.performance_metrics, 'to_dict') else {}
            }
            
        except Exception as e:
            raise MLModelError(f"Failed to train Prophet model: {e}")
    
    async def predict(self, periods: int = 24, freq: str = 'H') -> pd.DataFrame:
        """Generate predictions."""
        
        if not self.is_fitted:
            raise MLModelError("Model must be trained before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Add regressor values for future periods (using last known values)
            regressors = ['volume', 'rsi', 'macd', 'volatility', 'trend_strength']
            for regressor in regressors:
                if regressor in self.model.extra_regressors:
                    # Forward fill the last known value
                    last_value = future[regressor].iloc[-periods-1] if regressor in future.columns else 0
                    future[regressor] = future[regressor].fillna(last_value)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            return forecast
            
        except Exception as e:
            raise MLModelError(f"Failed to generate predictions: {e}")
    
    async def predict_with_confidence(self, periods: int = 24) -> Dict[str, Any]:
        """Generate predictions with confidence intervals."""
        
        forecast = await self.predict(periods)
        
        # Extract prediction components
        latest_predictions = forecast.tail(periods)
        
        predictions = {
            "yhat": latest_predictions['yhat'].tolist(),
            "yhat_lower": latest_predictions['yhat_lower'].tolist(),
            "yhat_upper": latest_predictions['yhat_upper'].tolist(),
            "trend": latest_predictions['trend'].tolist(),
            "seasonal": (latest_predictions['weekly'] + 
                        latest_predictions['yearly'] + 
                        latest_predictions.get('daily', 0)).tolist(),
            "timestamps": latest_predictions['ds'].dt.isoformat().tolist()
        }
        
        return predictions


class AdvancedPredictionAgent:
    """
    M7 - Advanced Price Prediction Agent
    Combines Prophet models, pattern detection, and multi-timeframe analysis.
    """
    
    def __init__(self, cache: TradingCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.prophet_models: Dict[str, EnhancedProphetModel] = {}
        self.pattern_detector = YOLOPatternDetector()
        
        # Timeframes for multi-timeframe analysis
        self.timeframes = ["1h", "4h", "1d"]
        
        # Performance tracking
        self.prediction_performance: Dict[str, Dict[str, float]] = {}
    
    async def train_models(self, symbol: str) -> Dict[str, Any]:
        """Train Prophet models for all timeframes."""
        
        results = {}
        
        for timeframe in self.timeframes:
            try:
                # Get market data
                cache_key = f"market_data:{symbol}:{timeframe}"
                market_data = await self.cache.get(cache_key)
                
                if not market_data:
                    self.logger.warning(f"No market data for {symbol}:{timeframe}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(market_data)
                
                if len(df) < 100:  # Need enough data
                    self.logger.warning(f"Insufficient data for {symbol}:{timeframe}")
                    continue
                
                # Ensure timestamp column
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(df.index)
                
                # Initialize and train model
                model_key = f"{symbol}_{timeframe}"
                self.prophet_models[model_key] = EnhancedProphetModel()
                
                training_result = await self.prophet_models[model_key].train(df, symbol)
                results[timeframe] = training_result
                
                self.logger.info(f"Trained Prophet model for {symbol}:{timeframe}")
                
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}:{timeframe}: {e}")
                results[timeframe] = {"status": "error", "message": str(e)}
        
        return results
    
    async def generate_comprehensive_prediction(self, symbol: str) -> Optional[PricePrediction]:
        """Generate comprehensive prediction combining all methods."""
        
        try:
            # Multi-timeframe predictions
            predictions = {}
            confidence_intervals = {}
            
            # Pattern detection
            pattern_signals = []
            
            # Trend analysis
            trend_direction = "sideways"
            trend_strength = 0.0
            
            # Process each timeframe
            for timeframe in self.timeframes:
                model_key = f"{symbol}_{timeframe}"
                
                if model_key in self.prophet_models:
                    # Prophet prediction
                    try:
                        prophet_pred = await self.prophet_models[model_key].predict_with_confidence(periods=24)
                        
                        predictions[f"{timeframe}_24h"] = prophet_pred["yhat"][-1]  # 24h ahead
                        confidence_intervals[f"{timeframe}_24h"] = (
                            prophet_pred["yhat_lower"][-1],
                            prophet_pred["yhat_upper"][-1]
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error getting Prophet prediction for {timeframe}: {e}")
                
                # Pattern detection for this timeframe
                cache_key = f"market_data:{symbol}:{timeframe}"
                market_data = await self.cache.get(cache_key)
                
                if market_data:
                    df = pd.DataFrame(market_data)
                    if len(df) > 30:
                        patterns = await self.pattern_detector.detect_patterns(
                            df['close'].values, timeframe
                        )
                        pattern_signals.extend(patterns)
            
            # Analyze overall trend
            if predictions:
                # Get current price
                cache_key = f"market_data:{symbol}:1h"
                market_data = await self.cache.get(cache_key)
                
                if market_data:
                    current_price = market_data[-1]['close']
                    
                    # Analyze trend across timeframes
                    trend_votes = []
                    
                    for timeframe in self.timeframes:
                        pred_key = f"{timeframe}_24h"
                        if pred_key in predictions:
                            predicted_price = predictions[pred_key]
                            price_change = (predicted_price - current_price) / current_price
                            
                            if price_change > 0.01:  # > 1% up
                                trend_votes.append("up")
                                trend_strength += abs(price_change)
                            elif price_change < -0.01:  # > 1% down
                                trend_votes.append("down")
                                trend_strength += abs(price_change)
                            else:
                                trend_votes.append("sideways")
                    
                    # Determine overall trend
                    if trend_votes:
                        trend_counts = {
                            "up": trend_votes.count("up"),
                            "down": trend_votes.count("down"),
                            "sideways": trend_votes.count("sideways")
                        }
                        trend_direction = max(trend_counts.keys(), key=lambda k: trend_counts[k])
                        trend_strength = trend_strength / len(trend_votes)
            
            # Calculate seasonal components
            seasonal_components = await self._analyze_seasonality(symbol)
            
            # Forecast volatility
            volatility_forecast = await self._forecast_volatility(symbol)
            
            # Create comprehensive prediction
            comprehensive_prediction = PricePrediction(
                symbol=symbol,
                timeframe="multi",
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                pattern_signals=pattern_signals,
                seasonal_components=seasonal_components,
                volatility_forecast=volatility_forecast,
                prediction_timestamp=datetime.utcnow()
            )
            
            # Cache the prediction
            cache_key = f"comprehensive_prediction:{symbol}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, comprehensive_prediction.to_dict(), ttl=3600)  # 1 hour
            
            return comprehensive_prediction
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive prediction for {symbol}: {e}")
            return None
    
    async def _analyze_seasonality(self, symbol: str) -> Dict[str, float]:
        """Analyze seasonal components in price data."""
        
        try:
            # Get daily data for seasonality analysis
            cache_key = f"market_data:{symbol}:1d"
            market_data = await self.cache.get(cache_key)
            
            if not market_data or len(market_data) < 100:
                return {}
            
            df = pd.DataFrame(market_data)
            prices = df['close'].values
            
            # Calculate seasonal components
            seasonality = {}
            
            # Day of week effect
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['dow'] = df['timestamp'].dt.dayofweek
                
                dow_returns = df.groupby('dow')['close'].pct_change().mean()
                seasonality['day_of_week'] = dow_returns.to_dict()
            
            # Monthly effect
            if 'timestamp' in df.columns:
                df['month'] = df['timestamp'].dt.month
                monthly_returns = df.groupby('month')['close'].pct_change().mean()
                seasonality['monthly'] = monthly_returns.to_dict()
            
            # Hour of day effect (if intraday data available)
            cache_key_hourly = f"market_data:{symbol}:1h"
            hourly_data = await self.cache.get(cache_key_hourly)
            
            if hourly_data and len(hourly_data) > 100:
                hourly_df = pd.DataFrame(hourly_data)
                if 'timestamp' in hourly_df.columns:
                    hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
                    hourly_df['hour'] = hourly_df['timestamp'].dt.hour
                    
                    hourly_returns = hourly_df.groupby('hour')['close'].pct_change().mean()
                    seasonality['hourly'] = hourly_returns.to_dict()
            
            return seasonality
            
        except Exception as e:
            self.logger.error(f"Error analyzing seasonality: {e}")
            return {}
    
    async def _forecast_volatility(self, symbol: str) -> float:
        """Forecast volatility using GARCH-like approach."""
        
        try:
            # Get hourly data
            cache_key = f"market_data:{symbol}:1h"
            market_data = await self.cache.get(cache_key)
            
            if not market_data or len(market_data) < 50:
                return 0.02  # Default 2% volatility
            
            df = pd.DataFrame(market_data)
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return 0.02
            
            # Simple volatility forecast using exponential smoothing
            # Calculate rolling volatility
            rolling_vol = returns.rolling(20).std()
            
            # Exponential smoothing
            alpha = 0.1
            forecast_vol = rolling_vol.iloc[-1]
            
            for i in range(len(rolling_vol) - 20, len(rolling_vol)):
                if not np.isnan(rolling_vol.iloc[i]):
                    forecast_vol = alpha * rolling_vol.iloc[i] + (1 - alpha) * forecast_vol
            
            # Annualize (assuming hourly data)
            annualized_vol = forecast_vol * np.sqrt(24 * 365)
            
            return float(min(1.0, max(0.01, annualized_vol)))
            
        except Exception as e:
            self.logger.error(f"Error forecasting volatility: {e}")
            return 0.02
    
    async def get_pattern_signals(self, symbol: str, timeframe: str = "1h") -> List[PatternMatch]:
        """Get current pattern signals for a symbol."""
        
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            market_data = await self.cache.get(cache_key)
            
            if not market_data:
                return []
            
            df = pd.DataFrame(market_data)
            
            if len(df) < 30:
                return []
            
            patterns = await self.pattern_detector.detect_patterns(
                df['close'].values, timeframe
            )
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting pattern signals: {e}")
            return []
    
    async def validate_prediction(self, symbol: str, timeframe: str, 
                                predicted_price: float, actual_price: float) -> None:
        """Validate and update prediction performance."""
        
        try:
            prediction_error = abs(predicted_price - actual_price) / actual_price
            
            # Update performance tracking
            model_key = f"{symbol}_{timeframe}"
            
            if model_key not in self.prediction_performance:
                self.prediction_performance[model_key] = {
                    "total_predictions": 0,
                    "average_error": 0.0,
                    "last_updated": datetime.utcnow().isoformat()
                }
            
            perf = self.prediction_performance[model_key]
            
            # Update metrics with exponential moving average
            alpha = 0.1
            perf["average_error"] = alpha * prediction_error + (1 - alpha) * perf["average_error"]
            perf["total_predictions"] += 1
            perf["last_updated"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Updated prediction performance for {model_key}: "
                           f"error={prediction_error:.4f}, avg_error={perf['average_error']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the Advanced Prediction Agent."""
        
        return {
            "agent_id": "M7_advanced_prediction_agent",
            "trained_models": list(self.prophet_models.keys()),
            "timeframes": self.timeframes,
            "prediction_performance": self.prediction_performance,
            "pattern_templates": list(self.pattern_detector.pattern_templates.keys()),
            "last_updated": datetime.utcnow().isoformat()
        } 