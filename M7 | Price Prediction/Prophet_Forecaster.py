import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# Prophet-specific imports
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric, plot_components, plot_forecast_component

# Import necessary modules from existing codebase
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from M8___Paper_Trading.Paper_Trading import PaperTradingSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProphetForecaster:
    """
    Advanced time series forecasting using Facebook/Meta Prophet with best practices.
    
    This class implements time series forecasting using Meta's Prophet library, including:
    - Automatic hyperparameter tuning
    - Cross-validation
    - Performance evaluation
    - Visualization
    - Integration with trading signals
    - Seasonality and holiday analysis
    - Uncertainty quantification
    """
    
    def __init__(self, 
                 ticker: str = None, 
                 seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05,
                 changepoint_range: float = 0.8,
                 yearly_seasonality: Union[bool, int] = 'auto',
                 weekly_seasonality: Union[bool, int] = 'auto',
                 daily_seasonality: Union[bool, int] = 'auto',
                 holidays: pd.DataFrame = None,
                 interval_width: float = 0.8):
        """
        Initialize the Prophet forecaster.
        
        Args:
            ticker: Stock ticker symbol
            seasonality_mode: 'multiplicative' or 'additive'
            changepoint_prior_scale: Controls flexibility of trend, higher values = more flexible
            changepoint_range: Proportion of history in which trend changepoints occur
            yearly_seasonality: Fit yearly seasonality
            weekly_seasonality: Fit weekly seasonality
            daily_seasonality: Fit daily seasonality
            holidays: DataFrame with holiday dates and effects
            interval_width: Width of uncertainty intervals (0.8 = 80% interval)
        """
        self.ticker = ticker
        self.model = None
        self.forecast = None
        self.last_data = None
        self.cv_results = None
        self.best_params = None
        
        # Default model parameters (can be overridden by hyperparameter tuning)
        self.params = {
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'changepoint_range': changepoint_range,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'holidays': holidays,
            'interval_width': interval_width
        }
        
        # Create models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prophet_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Prepare data for Prophet forecasting.
        
        Args:
            data: DataFrame with datetime index and price data
            target_col: Column name for the target variable to forecast
            
        Returns:
            DataFrame formatted for Prophet (with 'ds' and 'y' columns)
        """
        if data is None or len(data) == 0:
            logger.error("No data povided for Prophet forecastingr")
            return None
            
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data = data.set_index(pd.DatetimeIndex(data.index))
                logger.info("Converted index to DatetimeIndex")
            except:
                logger.error("Failed to convert index to DatetimeIndex")
                return None
        
        # Create prophet dataframe with ds (datestamp) and y (target) columns
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = data.index
        prophet_df['y'] = data[target_col]
        
        # Check for missing values
        if prophet_df['y'].isnull().any():
            logger.warning(f"Found {prophet_df['y'].isnull().sum()} missing values in target column")
            prophet_df = prophet_df.dropna(subset=['y'])
            logger.info(f"Dropped rows with missing values, {len(prophet_df)} rows remaining")
        
        # Add additional regressors if needed
        if 'Volume' in data.columns:
            prophet_df['volume'] = data['Volume']
            logger.info("Added volume as additional regressor")
        
        self.last_data = prophet_df
        return prophet_df
        
    def add_market_regime_features(self, prophet_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features as additional regressors.
        
        Args:
            prophet_df: DataFrame prepared for Prophet
            data: Original DataFrame with price data
            
        Returns:
            Enhanced DataFrame with market regime features
        """
        # Calculate volatility (20-day rolling standard deviation)
        if 'Close' in data.columns:
            data['volatility'] = data['Close'].pct_change().rolling(window=20).std()
            prophet_df['volatility'] = data['volatility']
        
        # Calculate trend direction (1 for uptrend, -1 for downtrend)
        if 'SMA20' in data.columns and 'SMA50' in data.columns:
            data['trend'] = np.where(data['SMA20'] > data['SMA50'], 1, -1)
            prophet_df['trend'] = data['trend']
        
        # Market regime based on VIX (if available)
        if 'VIX' in data.columns:
            # High volatility regime when VIX > 20
            data['high_vol_regime'] = np.where(data['VIX'] > 20, 1, 0)
            prophet_df['high_vol_regime'] = data['high_vol_regime']
        
        return prophet_df
    
    def hyperparameter_tuning(self, data: pd.DataFrame, optimization_metric: str = 'mape',
                             param_grid: Dict = None) -> Dict:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            data: DataFrame formatted for Prophet (with 'ds' and 'y' columns)
            optimization_metric: Metric to optimize ('mse', 'rmse', 'mae', 'mape', 'coverage')
            param_grid: Dictionary of parameters to grid search
            
        Returns:
            Dictionary of best parameters
        """
        if data is None or len(data) < 100:
            logger.warning("Insufficient data for hyperparameter tuning")
            return self.params
            
        logger.info("Starting hyperparameter tuning")
        
        # Default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_mode': ['multiplicative', 'additive'],
                'changepoint_range': [0.8, 0.9, 0.95],
                'interval_width': [0.8, 0.9, 0.95]
            }
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in 
                     np.meshgrid(*param_grid.values(), indexing='ij')]
        all_params = [dict(zip(param_grid.keys(), v)) for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid.keys()))]
        
        # Initialize best parameters and metric
        best_params = None
        best_metric_value = float('inf')  # Lower is better for error metrics
        
        # Cross-validation for each parameter combination
        for params in all_params:
            logger.info(f"Evaluating parameters: {params}")
            
            # Create and fit model with current parameters
            m = Prophet(
                seasonality_mode=params.get('seasonality_mode', self.params['seasonality_mode']),
                changepoint_prior_scale=params.get('changepoint_prior_scale', self.params['changepoint_prior_scale']),
                changepoint_range=params.get('changepoint_range', self.params['changepoint_range']),
                interval_width=params.get('interval_width', self.params['interval_width']),
                yearly_seasonality=self.params['yearly_seasonality'],
                weekly_seasonality=self.params['weekly_seasonality'],
                daily_seasonality=self.params['daily_seasonality']
            )
            
            # Add additional regressors if present
            for col in data.columns:
                if col not in ['ds', 'y']:
                    m.add_regressor(col)
            
            # Fit the model
            m.fit(data)
            
            # Cross-validation
            try:
                horizon = int(len(data) * 0.2)  # 20% of data length for validation
                horizon_str = f"{horizon} days"
                cv_results = cross_validation(
                    m, 
                    horizon=horizon_str,
                    initial=int(len(data) * 0.5),  # 50% of data for initial training
                    parallel="processes",
                    disable_tqdm=True
                )
                
                # Calculate performance metrics
                metrics = performance_metrics(cv_results)
                metric_value = metrics[optimization_metric].mean()
                
                # Update best parameters if current parameters are better
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_params = params
                    logger.info(f"New best {optimization_metric}: {metric_value:.4f}")
            except Exception as e:
                logger.error(f"Error during cross-validation: {e}")
                continue
        
        if best_params:
            logger.info(f"Best parameters: {best_params}, {optimization_metric}: {best_metric_value:.4f}")
            self.best_params = best_params
            self.params.update(best_params)
        else:
            logger.warning("Hyperparameter tuning failed, using default parameters")
            
        return self.params
    
    def fit(self, data: pd.DataFrame, target_col: str = 'Close', auto_tune: bool = True,
           add_market_features: bool = True, original_data: pd.DataFrame = None) -> 'ProphetForecaster':
        """
        Fit the Prophet model to the data.
        
        Args:
            data: DataFrame with datetime index and price data
            target_col: Column name for the target variable to forecast
            auto_tune: Whether to perform automatic hyperparameter tuning
            add_market_features: Whether to add market regime features as regressors
            original_data: Original DataFrame with additional market data
            
        Returns:
            Self instance for method chaining
        """
        # Prepare data
        prophet_df = self.prepare_data(data, target_col)
        if prophet_df is None:
            logger.error("Failed to prepare data for Prophet")
            return self
        
        # Add market regime features if requested
        if add_market_features and original_data is not None:
            prophet_df = self.add_market_regime_features(prophet_df, original_data)
            logger.info("Added market regime features")
        
        # Hyperparameter tuning if requested
        if auto_tune:
            self.hyperparameter_tuning(prophet_df)
        
        # Create and fit the model
        self.model = Prophet(
            seasonality_mode=self.params['seasonality_mode'],
            changepoint_prior_scale=self.params['changepoint_prior_scale'],
            changepoint_range=self.params['changepoint_range'],
            yearly_seasonality=self.params['yearly_seasonality'],
            weekly_seasonality=self.params['weekly_seasonality'],
            daily_seasonality=self.params['daily_seasonality'],
            interval_width=self.params['interval_width']
        )
        
        # Add additional regressors if present
        for col in prophet_df.columns:
            if col not in ['ds', 'y']:
                self.model.add_regressor(col)
        
        # Fit the model
        self.model.fit(prophet_df)
        logger.info(f"Fitted Prophet model for {self.ticker}")
        
        return self
    
    def predict(self, periods: int = 30, freq: str = 'D', include_history: bool = True,
               return_components: bool = False) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecast ('D' for daily, 'W' for weekly, etc.)
            include_history: Whether to include historical data in the forecast
            return_components: Whether to return forecast components
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None:
            logger.error("Model not fitted, call fit() first")
            return None
            
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        
        # Add additional regressors for future periods if needed
        # This is a simplified approach; in practice, you would need to forecast these regressors
        if self.last_data is not None:
            for col in self.last_data.columns:
                if col not in ['ds', 'y']:
                    if col in future.columns:
                        continue
                    # Just use the last value for future periods as a simple approach
                    last_value = self.last_data[col].iloc[-1]
                    future[col] = last_value
        
        # Generate forecast
        self.forecast = self.model.predict(future)
        
        if return_components:
            components = {}
            for component in ['trend', 'yearly', 'weekly', 'daily']:
                if component in self.forecast.columns:
                    components[component] = self.forecast[['ds', component]]
            return self.forecast, components
        
        return self.forecast
    
    def evaluate(self, test_data: pd.DataFrame = None, target_col: str = 'Close') -> Dict[str, float]:
        """
        Evaluate the model's performance on test data.
        
        Args:
            test_data: Test data (if None, uses cross-validation)
            target_col: Column name for the target variable
            
        Returns:
            Dictionary of performance metrics
        """
        if self.model is None:
            logger.error("Model not fitted, call fit() first")
            return None
            
        results = {}
        
        # If test data is provided, evaluate on it
        if test_data is not None:
            # Prepare test data
            test_prophet = self.prepare_data(test_data, target_col)
            
            # Make predictions on test data timestamps
            predictions = self.model.predict(test_prophet[['ds']])
            
            # Calculate metrics
            y_true = test_prophet['y'].values
            y_pred = predictions['yhat'].values
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_true - y_pred))
            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            results = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
        else:
            # Use cross-validation
            horizon = int(len(self.last_data) * 0.2)  # 20% of data length
            horizon_str = f"{horizon} days"
            
            try:
                cv_results = cross_validation(
                    self.model, 
                    horizon=horizon_str,
                    initial=int(len(self.last_data) * 0.5),  # 50% of data
                    parallel="processes",
                    disable_tqdm=True
                )
                
                self.cv_results = cv_results
                metrics = performance_metrics(cv_results)
                
                results = {
                    'mae': metrics['mae'].mean(),
                    'rmse': metrics['rmse'].mean(),
                    'mape': metrics['mape'].mean(),
                    'coverage': metrics['coverage'].mean()
                }
            except Exception as e:
                logger.error(f"Error during cross-validation: {e}")
                return None
        
        # Print evaluation results
        logger.info(f"Evaluation results for {self.ticker}:")
        for metric, value in results.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return results
    
    def plot_forecast(self, forecast: pd.DataFrame = None, include_components: bool = False, 
                     save_path: str = None) -> None:
        """
        Plot the forecast and optionally save it.
        
        Args:
            forecast: Forecast DataFrame (if None, uses the last forecast)
            include_components: Whether to include forecast components in the plot
            save_path: Path to save the plot
        """
        if forecast is None:
            forecast = self.forecast
            
        if forecast is None:
            logger.error("No forecast available, call predict() first")
            return
            
        # Create figure with appropriate size
        plt.figure(figsize=(12, 8))
        
        # Main forecast plot
        self.model.plot(forecast, uncertainty=True, xlabel='Date', ylabel='Price')
        plt.title(f'{self.ticker} Price Forecast')
        plt.grid(True)
        
        # Save the plot if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Forecast plot saved to {save_path}")
            
        plt.show()
        
        # Plot components if requested
        if include_components:
            fig = self.model.plot_components(forecast)
            
            # Save components plot if path provided
            if save_path:
                component_path = save_path.replace('.png', '_components.png')
                fig.savefig(component_path)
                logger.info(f"Components plot saved to {component_path}")
            
            plt.show()
    
    def plot_cross_validation(self, metric: str = 'mape', save_path: str = None) -> None:
        """
        Plot cross-validation results.
        
        Args:
            metric: Metric to plot ('mse', 'rmse', 'mae', 'mape', 'coverage')
            save_path: Path to save the plot
        """
        if self.cv_results is None:
            logger.error("No cross-validation results available, run evaluate() first")
            return
            
        fig = plot_cross_validation_metric(self.cv_results, metric)
        
        # Save the plot if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Cross-validation plot saved to {save_path}")
            
        plt.show()
    
    def generate_trading_signals(self, forecast: pd.DataFrame = None, horizon: int = 5,
                              confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Generate trading signals from the forecast.
        
        Args:
            forecast: Forecast DataFrame (if None, uses the last forecast)
            horizon: Number of periods to consider for signals
            confidence_threshold: Threshold for signal confidence
            
        Returns:
            List of trading signal dictionaries
        """
        if forecast is None:
            forecast = self.forecast
            
        if forecast is None:
            logger.error("No forecast available, call predict() first")
            return []
            
        signals = []
        
        # Get the last actual data point
        last_date = self.last_data['ds'].max()
        last_price = self.last_data.loc[self.last_data['ds'] == last_date, 'y'].values[0]
        
        # Get forecast for the specified horizon
        future_forecast = forecast[forecast['ds'] > last_date].iloc[:horizon]
        
        # Check trend direction
        price_diff = future_forecast['yhat'].iloc[-1] - last_price
        price_pct_change = (price_diff / last_price) * 100
        
        # Calculate mean trend and confidence
        trend_direction = np.sign(price_diff)
        
        # Calculate confidence based on uncertainty
        uncertainty = np.mean(future_forecast['yhat_upper'] - future_forecast['yhat_lower'])
        rel_uncertainty = uncertainty / future_forecast['yhat'].mean()
        confidence = max(0, 1 - rel_uncertainty)
        
        # Generate signal based on trend and confidence
        if abs(price_pct_change) > 2 and confidence > confidence_threshold:
            if trend_direction > 0:
                signals.append({
                    "type": "BUY",
                    "reason": f"Prophet forecasts {price_pct_change:.2f}% increase over {horizon} days",
                    "strength": "Strong" if confidence > 0.85 else "Medium",
                    "confidence": confidence
                })
            else:
                signals.append({
                    "type": "SELL",
                    "reason": f"Prophet forecasts {abs(price_pct_change):.2f}% decrease over {horizon} days",
                    "strength": "Strong" if confidence > 0.85 else "Medium",
                    "confidence": confidence
                })
        
        # Check for trend acceleration/deceleration
        # For this, look at the change in slope of the forecast
        if len(future_forecast) > 3:
            try:
                slopes = np.diff(future_forecast['yhat'].values)
                accel = np.diff(slopes)
                mean_accel = np.mean(accel)
                
                # If we have significant acceleration/deceleration, generate signal
                if abs(mean_accel) > 0.01 * last_price and confidence > confidence_threshold:
                    if mean_accel > 0:
                        signals.append({
                            "type": "BUY",
                            "reason": "Prophet forecasts accelerating upward trend",
                            "strength": "Medium",
                            "confidence": confidence
                        })
                    else:
                        signals.append({
                            "type": "SELL",
                            "reason": "Prophet forecasts accelerating downward trend",
                            "strength": "Medium",
                            "confidence": confidence
                        })
            except:
                pass
                
        return signals
    
    def save_model(self, filename: str = None) -> str:
        """
        Save the Prophet model to disk.
        
        Args:
            filename: Filename to save model (if None, generates based on ticker)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            logger.error("No model to save, call fit() first")
            return None
            
        if filename is None:
            filename = f"{self.ticker}_prophet_model.json" if self.ticker else "prophet_model.json"
            
        # Create full path
        model_path = os.path.join(self.models_dir, filename)
        
        try:
            # Serialize the model
            with open(model_path, 'w') as f:
                json.dump(self.model.to_json(), f)
                
            # Save forecast if available
            if self.forecast is not None:
                forecast_path = model_path.replace('.json', '_forecast.csv')
                self.forecast.to_csv(forecast_path, index=False)
                
            # Save parameters
            params_path = model_path.replace('.json', '_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.params, f)
                
            # Save last data
            if self.last_data is not None:
                data_path = model_path.replace('.json', '_data.csv')
                self.last_data.to_csv(data_path, index=False)
                
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    @classmethod
    def load_model(cls, path: str, ticker: str = None) -> 'ProphetForecaster':
        """
        Load a saved Prophet model.
        
        Args:
            path: Path to the saved model
            ticker: Ticker symbol (if None, extracts from filename)
            
        Returns:
            ProphetForecaster instance with loaded model
        """
        # Extract ticker from filename if not provided
        if ticker is None:
            ticker = os.path.basename(path).split('_')[0]
            
        # Create instance
        instance = cls(ticker=ticker)
        
        try:
            # Load model
            with open(path, 'r') as f:
                model_json = json.load(f)
                
            instance.model = Prophet.from_json(model_json)
            
            # Load parameters if available
            params_path = path.replace('.json', '_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    instance.params = json.load(f)
            
            # Load forecast if available
            forecast_path = path.replace('.json', '_forecast.csv')
            if os.path.exists(forecast_path):
                instance.forecast = pd.read_csv(forecast_path)
                
            # Load data if available
            data_path = path.replace('.json', '_data.csv')
            if os.path.exists(data_path):
                instance.last_data = pd.read_csv(data_path)
                
            logger.info(f"Model loaded from {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def backtest_trading_strategy(self, data: pd.DataFrame, 
                               forecast_horizon: int = 5, 
                               retrain_freq: int = 20,
                               confidence_threshold: float = 0.7,
                               starting_capital: float = 10000.0) -> Dict:
        """
        Backtest a trading strategy based on Prophet forecasts.
        
        Args:
            data: Historical data for backtesting
            forecast_horizon: Number of days to forecast
            retrain_freq: Number of days between model retraining
            confidence_threshold: Threshold for signal confidence
            starting_capital: Initial capital for paper trading
            
        Returns:
            Dictionary of backtest results
        """
        if data is None or len(data) < 100:
            logger.error("Insufficient data for backtesting")
            return None
            
        # Initialize paper trading simulator
        simulator = PaperTradingSimulator(starting_capital=starting_capital)
        
        # Prepare for backtesting
        train_size = int(len(data) * 0.7)  # Initial 70% for training
        
        # Convert index to datetime if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Converting index to DatetimeIndex")
            data = data.reset_index()
            data.set_index(pd.DatetimeIndex(data['index']), inplace=True)
        
        # Sort by date
        data = data.sort_index()
        
        # Split data
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        logger.info(f"Backtesting with {len(train_data)} training points and {len(test_data)} testing points")
        
        # Initialize trading signals function
        def prophet_signal_generator(data_slice):
            # Skip if we have insufficient data
            if len(data_slice) < 50:
                return []
                
            # Get the latest date
            latest_date = data_slice.index[-1]
            
            # Check if we need to retrain
            retrain = False
            if not hasattr(prophet_signal_generator, 'last_train_date'):
                retrain = True
            elif (latest_date - prophet_signal_generator.last_train_date).days >= retrain_freq:
                retrain = True
                
            # Retrain model if needed
            if retrain:
                try:
                    # Fit model on available data
                    self.fit(data_slice, auto_tune=False)
                    
                    # Make forecast
                    self.predict(periods=forecast_horizon)
                    
                    # Store last training date
                    prophet_signal_generator.last_train_date = latest_date
                    
                    logger.info(f"Retrained Prophet model at {latest_date}")
                except Exception as e:
                    logger.error(f"Error training model: {e}")
                    return []
            
            # Generate signals
            try:
                signals = self.generate_trading_signals(
                    horizon=forecast_horizon,
                    confidence_threshold=confidence_threshold
                )
                return signals
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                return []
        
        # Run backtest
        backtest_results = simulator.backtest(
            ticker=self.ticker,
            start_date=test_data.index[0].strftime("%Y-%m-%d"),
            end_date=test_data.index[-1].strftime("%Y-%m-%d"),
            signals_func=prophet_signal_generator
        )
        
        return backtest_results


# Example usage
if __name__ == "__main__":
    # Example: Forecasting stock price with Prophet
    import yfinance as yf
    
    # Download data
    ticker = "AAPL"
    data = yf.download(ticker, period="2y")
    
    # Initialize forecaster
    forecaster = ProphetForecaster(ticker=ticker)
    
    # Fit model with automatic hyperparameter tuning
    forecaster.fit(data, auto_tune=True)
    
    # Make forecast
    forecast = forecaster.predict(periods=30)
    
    # Plot forecast
    forecaster.plot_forecast(include_components=True)
    
    # Evaluate model
    metrics = forecaster.evaluate()
    
    # Generate trading signals
    signals = forecaster.generate_trading_signals()
    
    if signals:
        print("\nTrading Signals:")
        for signal in signals:
            print(f"  - {signal['type']}: {signal['reason']} (Confidence: {signal['confidence']:.2f})")
    
    # Backtest strategy
    backtest_results = forecaster.backtest_trading_strategy(data)
    
    if backtest_results:
        print("\nBacktest Results:")
        simulator = PaperTradingSimulator()
        simulator.print_performance_summary(backtest_results)
        
        # Show performance chart
        plt.show()
    
    # Save model
    forecaster.save_model() 