# Import required libraries
import zipline
from zipline.api import order_target_percent, record, symbol, get_datetime, attach_pipeline, pipeline_output
from zipline.finance import commission, slippage
from zipline.utils.calendars import get_calendar
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.factors import SimpleMovingAverage, RSI
from zipline.pipeline.data import USEquityPricing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zipline import run_algorithm
from datetime import datetime, timedelta
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# System integration - import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from M9___Market_Sentiment.GeoAgentpy import GeopoliticalAgent
    from M6___Risk_Management.portfolio_optimizer import PortfolioOptimizer
    from M4___Strategies.strategy_framework import StrategySignal
    from M7___Price_Prediction.Prophet_Forecaster import ProphetForecaster
except ImportError:
    logging.warning("Some Quantum Trading Matrix modules could not be imported. Limited functionality available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zipline_backtest.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ZiplineBacktester")

# =============================
# 1. Custom Factors for Integration
# =============================

class GeopoliticalRiskFactor(CustomFactor):
    """
    Custom factor for incorporating geopolitical risk signals into Zipline pipeline.
    """
    inputs = [USEquityPricing.close]
    window_length = 1
    
    def __init__(self, geo_agent, country_mapping, default_risk=0.5):
        self.geo_agent = geo_agent
        self.country_mapping = country_mapping
        self.default_risk = default_risk
        self.risk_cache = {}
        super(GeopoliticalRiskFactor, self).__init__()
    
    def compute(self, today, assets, out, close):
        # Convert assets to tickers
        tickers = [asset.symbol for asset in assets]
        
        # Calculate risk for each ticker
        for i, ticker in enumerate(tickers):
            # Look up which country this ticker is associated with
            country = None
            for c, markets in self.country_mapping.items():
                if ticker in markets:
                    country = c
                    break
            
            # If we found a country, get its risk score
            if country:
                risk = self.geo_agent.assess_risk(country)
            else:
                risk = self.default_risk
            
            # Store in output array
            out[i] = risk

class ProphetPredictionFactor(CustomFactor):
    """
    Custom factor for incorporating Prophet predictions into Zipline pipeline.
    """
    inputs = [USEquityPricing.close]
    window_length = 252  # Need historical data for forecasting
    
    def __init__(self, prophet_forecaster, forecast_period=5):
        self.prophet_forecaster = prophet_forecaster
        self.forecast_period = forecast_period
        self.forecast_cache = {}
        super(ProphetPredictionFactor, self).__init__()
    
    def compute(self, today, assets, out, close):
        # Convert today to string format
        today_str = today.strftime('%Y-%m-%d')
        
        # Convert assets to tickers
        tickers = [asset.symbol for asset in assets]
        
        # Generate forecasts for each ticker
        for i, ticker in enumerate(tickers):
            # Check cache first
            if ticker in self.forecast_cache and today_str in self.forecast_cache[ticker]:
                forecast = self.forecast_cache[ticker][today_str]
            else:
                # Get historical data for this asset
                history = pd.DataFrame(close[:, i], columns=['close'])
                history.index = [today - timedelta(days=self.window_length-1-j) for j in range(self.window_length)]
                
                # Generate forecast
                try:
                    forecast = self.prophet_forecaster.forecast_ticker(
                        ticker, 
                        history, 
                        periods=self.forecast_period
                    )
                    
                    # Cache the result
                    if ticker not in self.forecast_cache:
                        self.forecast_cache[ticker] = {}
                    self.forecast_cache[ticker][today_str] = forecast
                    
                except Exception as e:
                    logger.error(f"Error forecasting {ticker}: {e}")
                    forecast = None
            
            # Store expected return in output
            if forecast is not None and len(forecast) > 0:
                # Calculate expected return over forecast period
                expected_return = (forecast.iloc[-1]['yhat'] / close[-1, i]) - 1
                out[i] = expected_return
            else:
                out[i] = 0.0

# =============================
# 2. Enhanced Backtester Class
# =============================

class QuantumBacktester:
    """
    Enhanced backtester for Quantum Trading Matrix that integrates multiple signal sources.
    """
    
    def __init__(self, 
                tickers: List[str], 
                start_date: str, 
                end_date: str,
                capital_base: float = 100000.0,
                data_frequency: str = 'daily',
                benchmark: str = 'SPY'):
        """
        Initialize the backtester with configuration.
        
        Args:
            tickers: List of ticker symbols to include in the backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            capital_base: Initial capital
            data_frequency: Data frequency ('daily' or 'minute')
            benchmark: Benchmark ticker
        """
        self.tickers = tickers
        self.start_date = pd.Timestamp(start_date, tz='utc')
        self.end_date = pd.Timestamp(end_date, tz='utc')
        self.capital_base = capital_base
        self.data_frequency = data_frequency
        self.benchmark = benchmark
        
        # Initialize integration components
        self.geo_agent = None
        self.prophet_forecaster = None
        self.portfolio_optimizer = None
        
        # Initialize strategy parameters
        self.strategy_params = {}
        
        logger.info(f"Initialized QuantumBacktester with {len(tickers)} tickers from {start_date} to {end_date}")
    
    def connect_geo_agent(self, geo_agent: GeopoliticalAgent = None):
        """
        Connect a GeopoliticalAgent to the backtester.
        
        Args:
            geo_agent: GeopoliticalAgent instance (creates a new one if None)
        """
        if geo_agent is None:
            try:
                geo_agent = GeopoliticalAgent(auto_fetch=False)
                
                # Load sample data for backtesting
                dummy_data = {
                    'GPI': pd.DataFrame({
                        'Country': ['USA', 'China', 'Russia', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada'],
                        'Value': [0.7, 0.6, 0.85, 0.4, 0.3, 0.5, 0.45, 0.65, 0.55, 0.35],
                        'Rank': [10, 15, 5, 30, 40, 20, 25, 12, 18, 35]
                    }),
                    'CPI': pd.DataFrame({
                        'Country': ['USA', 'China', 'Russia', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada'],
                        'Value': [0.3, 0.6, 0.7, 0.2, 0.15, 0.25, 0.3, 0.5, 0.55, 0.2],
                        'Rank': [25, 78, 136, 10, 8, 11, 22, 85, 96, 13]
                    })
                }
                geo_agent.data = dummy_data
                geo_agent.preprocess_data()
                geo_agent.calculate_risk_scores()
                
            except Exception as e:
                logger.error(f"Error creating GeopoliticalAgent: {e}")
                return False
        
        self.geo_agent = geo_agent
        logger.info("Connected GeopoliticalAgent to backtester")
        return True
    
    def connect_prophet_forecaster(self, forecaster: ProphetForecaster = None):
        """
        Connect a ProphetForecaster to the backtester.
        
        Args:
            forecaster: ProphetForecaster instance (creates a new one if None)
        """
        if forecaster is None:
            try:
                forecaster = ProphetForecaster()
            except Exception as e:
                logger.error(f"Error creating ProphetForecaster: {e}")
                return False
        
        self.prophet_forecaster = forecaster
        logger.info("Connected ProphetForecaster to backtester")
        return True
    
    def connect_portfolio_optimizer(self, optimizer: PortfolioOptimizer = None):
        """
        Connect a PortfolioOptimizer to the backtester.
        
        Args:
            optimizer: PortfolioOptimizer instance (creates a new one if None)
        """
        if optimizer is None:
            try:
                optimizer = PortfolioOptimizer()
            except Exception as e:
                logger.error(f"Error creating PortfolioOptimizer: {e}")
                return False
        
        self.portfolio_optimizer = optimizer
        logger.info("Connected PortfolioOptimizer to backtester")
        return True
    
    def set_strategy_params(self, params: Dict):
        """
        Set parameters for the strategy.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.strategy_params = params
        logger.info(f"Set strategy parameters: {params}")
    
    def create_pipeline(self):
        """
        Create a Zipline pipeline with integrated factors.
        
        Returns:
            Configured Pipeline object
        """
        pipeline = Pipeline()
        
        # Add standard technical factors
        pipeline.add(
            SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=20),
            'sma20'
        )
        
        pipeline.add(
            SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=50),
            'sma50'
        )
        
        pipeline.add(
            RSI(window_length=14),
            'rsi'
        )
        
        # Add geopolitical risk factor if available
        if self.geo_agent:
            pipeline.add(
                GeopoliticalRiskFactor(
                    self.geo_agent, 
                    self.geo_agent.country_mapping
                ),
                'geo_risk'
            )
        
        # Add prophet prediction factor if available
        if self.prophet_forecaster:
            pipeline.add(
                ProphetPredictionFactor(self.prophet_forecaster),
                'prophet_prediction'
            )
        
        return pipeline
    
    def initialize_algo(self, context):
        """
        Initialize the algorithm with our custom setup.
        
        Args:
            context: Zipline context object
        """
        # Set up assets
        context.assets = [symbol(ticker) for ticker in self.tickers]
        
        # Store strategy parameters
        context.strategy_params = self.strategy_params
        
        # Set up moving average parameters
        context.short_window = self.strategy_params.get('short_window', 20)
        context.long_window = self.strategy_params.get('long_window', 50)
        
        # Set up risk management parameters
        context.max_position_size = self.strategy_params.get('max_position_size', 0.25)
        context.use_geo_risk = self.strategy_params.get('use_geo_risk', True)
        context.use_prophet = self.strategy_params.get('use_prophet', True)
        
        # Set up trading costs
        context.set_commission(commission.PerShare(cost=0.001))
        context.set_slippage(slippage.FixedSlippage(spread=0.01))
        
        # Attach pipeline
        attach_pipeline(self.create_pipeline(), 'quantum_pipeline')
        
        logger.info("Algorithm initialized with Quantum Trading Matrix pipeline")
    
    def handle_data_algo(self, context, data):
        """
        Main trading logic executed each day.
        
        Args:
            context: Zipline context object
            data: Zipline data object
        """
        # Get pipeline outputs
        pipeline_data = pipeline_output('quantum_pipeline')
        
        # Calculate position sizes
        position_sizes = {}
        total_signal = 0
        
        for asset in context.assets:
            # Skip assets that don't have data
            if asset not in pipeline_data.index:
                continue
                
            # Get technical signals
            asset_pipeline = pipeline_data.loc[asset]
            sma_signal = 1 if asset_pipeline['sma20'] > asset_pipeline['sma50'] else -1
            
            # Get RSI signals (oversold/overbought)
            rsi = asset_pipeline['rsi']
            rsi_signal = 0
            if rsi < 30:  # Oversold
                rsi_signal = 1
            elif rsi > 70:  # Overbought
                rsi_signal = -1
            
            # Combine base signals
            base_signal = (sma_signal + rsi_signal) / 2
            
            # Adjust with geopolitical risk if available
            if context.use_geo_risk and 'geo_risk' in asset_pipeline:
                geo_risk = asset_pipeline['geo_risk']
                # Higher risk reduces position size
                geo_adjustment = 1 - geo_risk
                base_signal *= geo_adjustment
            
            # Adjust with prophet predictions if available
            if context.use_prophet and 'prophet_prediction' in asset_pipeline:
                prophet_pred = asset_pipeline['prophet_prediction']
                # Scale prediction to -1 to 1 range
                scaled_pred = np.clip(prophet_pred * 5, -1, 1)  # 20% prediction = full signal
                base_signal = 0.7 * base_signal + 0.3 * scaled_pred  # 70% technical, 30% prediction
            
            # Scale signal to position size
            position_size = base_signal * context.max_position_size
            position_sizes[asset] = position_size
            total_signal += abs(position_size)
        
        # Normalize position sizes if total exceeds 1.0
        if total_signal > 1.0:
            for asset in position_sizes:
                position_sizes[asset] /= total_signal
        
        # Execute trades
        for asset, size in position_sizes.items():
            order_target_percent(asset, size)
            current_price = data.current(asset, 'price')
            
            # Record position information
            record(**{
                f"{asset.symbol}_pos": size,
                f"{asset.symbol}_price": current_price
            })
        
        # Record pipeline outputs for analysis
        for asset in context.assets:
            if asset in pipeline_data.index:
                record(**{
                    f"{asset.symbol}_sma20": pipeline_data.loc[asset, 'sma20'],
                    f"{asset.symbol}_sma50": pipeline_data.loc[asset, 'sma50'],
                    f"{asset.symbol}_rsi": pipeline_data.loc[asset, 'rsi']
                })
                
                if 'geo_risk' in pipeline_data.columns:
                    record(**{f"{asset.symbol}_geo_risk": pipeline_data.loc[asset, 'geo_risk']})
                
                if 'prophet_prediction' in pipeline_data.columns:
                    record(**{f"{asset.symbol}_prophet_pred": pipeline_data.loc[asset, 'prophet_prediction']})
    
    def analyze_results(self, context, perf):
        """
        Analyze backtest results with enhanced visualizations.
        
        Args:
            context: Zipline context object
            perf: Performance DataFrame
        """
        # Create main performance figure
        fig = plt.figure(figsize=(16, 16))
        
        # Portfolio value subplot
        ax1 = fig.add_subplot(3, 1, 1)
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Quantum Trading Matrix Backtest Results')
        ax1.grid(True)
        
        # Returns subplot
        ax2 = fig.add_subplot(3, 1, 2)
        perf.returns.plot(ax=ax2)
        ax2.set_ylabel('Returns')
        ax2.grid(True)
        
        # Position allocation subplot
        ax3 = fig.add_subplot(3, 1, 3)
        
        # Get position columns
        pos_cols = [col for col in perf.columns if '_pos' in col]
        
        if pos_cols:
            perf[pos_cols].plot(ax=ax3)
            ax3.set_ylabel('Position Allocation')
            ax3.set_xlabel('Date')
            ax3.grid(True)
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Save the figure
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(results_dir, f"backtest_results_{timestamp}.png")
        plt.savefig(fig_path)
        
        # Save performance metrics
        metrics = {
            'total_return': (perf.portfolio_value[-1] / perf.portfolio_value[0]) - 1,
            'annual_return': ((perf.portfolio_value[-1] / perf.portfolio_value[0]) ** (252 / len(perf))) - 1,
            'sharpe_ratio': (perf.returns.mean() / perf.returns.std()) * np.sqrt(252),
            'max_drawdown': (perf.portfolio_value / perf.portfolio_value.cummax() - 1.0).min(),
            'winning_days': (perf.returns > 0).sum() / len(perf),
            'backtest_length_days': len(perf)
        }
        
        metrics_path = os.path.join(results_dir, f"backtest_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Backtest results saved to {fig_path}")
        logger.info(f"Backtest metrics saved to {metrics_path}")
        
        return metrics, fig
    
    def run_backtest(self):
        """
        Run the backtest with all configured components.
        
        Returns:
            Tuple of (performance DataFrame, metrics dictionary)
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Set up algorithm configuration
        algo_config = {
            'initialize': self.initialize_algo,
            'handle_data': self.handle_data_algo,
            'analyze': self.analyze_results,
            'start': self.start_date,
            'end': self.end_date,
            'capital_base': self.capital_base,
            'data_frequency': self.data_frequency,
            'bundle': 'quantopian-quandl',
            'trading_calendar': get_calendar("XNYS")
        }
        
        try:
            # Run the algorithm
            perf = run_algorithm(**algo_config)
            
            # Get metrics from the performance data
            metrics = {
                'total_return': (perf.portfolio_value[-1] / perf.portfolio_value[0]) - 1,
                'annual_return': ((perf.portfolio_value[-1] / perf.portfolio_value[0]) ** (252 / len(perf))) - 1,
                'sharpe_ratio': (perf.returns.mean() / perf.returns.std()) * np.sqrt(252),
                'max_drawdown': (perf.portfolio_value / perf.portfolio_value.cummax() - 1.0).min(),
                'winning_days': (perf.returns > 0).sum() / len(perf),
                'backtest_length_days': len(perf)
            }
            
            logger.info(f"Backtest completed successfully. Total return: {metrics['total_return']:.2%}")
            return perf, metrics
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None, None

# =============================
# 3. Integration with Quantum Trading Matrix
# =============================

def integrate_with_quantum_trading_matrix():
    """
    Integrate Zipline backtester with the Quantum Trading Matrix system.
    
    Returns:
        Dictionary with integration status and components
    """
    integration = {
        'status': 'success',
        'components': []
    }
    
    try:
        # Try to import GeopoliticalAgent
        try:
            from M9___Market_Sentiment.GeoAgentpy import GeopoliticalAgent
            geo_agent = GeopoliticalAgent(auto_fetch=False)
            integration['components'].append('GeopoliticalAgent')
        except ImportError:
            logger.warning("GeopoliticalAgent could not be imported")
            geo_agent = None
        
        # Try to import ProphetForecaster
        try:
            from M7___Price_Prediction.Prophet_Forecaster import ProphetForecaster
            prophet_forecaster = ProphetForecaster()
            integration['components'].append('ProphetForecaster')
        except ImportError:
            logger.warning("ProphetForecaster could not be imported")
            prophet_forecaster = None
        
        # Try to import PortfolioOptimizer
        try:
            from M6___Risk_Management.portfolio_optimizer import PortfolioOptimizer
            portfolio_optimizer = PortfolioOptimizer()
            integration['components'].append('PortfolioOptimizer')
        except ImportError:
            logger.warning("PortfolioOptimizer could not be imported")
            portfolio_optimizer = None
        
        # Add other components as needed
        
        logger.info(f"Zipline integration with Quantum Trading Matrix completed. Components: {integration['components']}")
        
    except Exception as e:
        logger.error(f"Error during Quantum Trading Matrix integration: {e}")
        integration['status'] = 'error'
        integration['error'] = str(e)
    
    return integration

# =============================
# 4. Example Usage
# =============================

def run_sample_backtest():
    """
    Run a sample backtest with Quantum Trading Matrix integration.
    """
    # Initialize backtester
    backtester = QuantumBacktester(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        start_date='2019-01-01',
        end_date='2022-01-01',
        capital_base=100000.0
    )
    
    # Connect components
    backtester.connect_geo_agent()
    backtester.connect_prophet_forecaster()
    backtester.connect_portfolio_optimizer()
    
    # Configure strategy
    backtester.set_strategy_params({
        'short_window': 20,
        'long_window': 50,
        'max_position_size': 0.2,
        'use_geo_risk': True,
        'use_prophet': True
    })
    
    # Run the backtest
    perf, metrics = backtester.run_backtest()
    
    # Display results if successful
    if metrics:
        print("\nBacktest Results:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['winning_days']:.2%}")
    
    return perf, metrics

# =============================
# 5. Original Simple Example (kept for reference)
# =============================

def initialize(context):
    # Define the stock we want to trade (Apple in this case)
    context.asset = symbol('AAPL')
    # Set moving average windows: 20 days (short) and 50 days (long)
    context.short_window = 20
    context.long_window = 50
    
    # Set trading costs
    # Commission of $0.001 per share
    context.set_commission(commission.PerShare(cost=0.001))
    # Fixed slippage of 1% to simulate market impact
    context.set_slippage(slippage.FixedSlippage(spread=0.01))

def handle_data(context, data):
    # Calculate moving averages using historical price data
    short_mavg = data.history(context.asset, 'price', bar_count=context.short_window, frequency="1d").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=context.long_window, frequency="1d").mean()

    # Implement moving average crossover strategy
    if short_mavg > long_mavg:
        order_target_percent(context.asset, 1.0)  # Buy signal: Invest 100% of portfolio
    elif short_mavg < long_mavg:
        order_target_percent(context.asset, 0.0)  # Sell signal: Sell entire position

    # Store data for later analysis
    record(AAPL=data.current(context.asset, "price"),
           short_mavg=short_mavg,
           long_mavg=long_mavg)

def analyze(context, perf):
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Left y-axis: Portfolio Value
    ax1.plot(perf.index, perf['portfolio_value'], label='Portfolio Value')
    ax1.set_ylabel('Portfolio Value in $')
    ax1.legend(loc='upper left')

    # Right y-axis: Stock Price and Moving Averages
    ax2 = ax1.twinx()
    ax2.plot(perf.index, perf['AAPL'], color='orange', label='AAPL Price')
    ax2.plot(perf.index, perf['short_mavg'], color='blue', label='20-Day MA')
    ax2.plot(perf.index, perf['long_mavg'], color='red', label='50-Day MA')
    ax2.set_ylabel('Price in $')
    ax2.legend(loc='upper right')

    plt.show()

# =============================
# 6. Main Entry Point
# =============================

if __name__ == "__main__":
    # Integrate with Quantum Trading Matrix
    integration_status = integrate_with_quantum_trading_matrix()
    print(f"Integration status: {integration_status['status']}")
    print(f"Integrated components: {integration_status['components']}")
    
    # Run either the enhanced or original example based on integration status
    if integration_status['status'] == 'success' and integration_status['components']:
        print("\nRunning enhanced Quantum Trading Matrix backtest...")
        perf, metrics = run_sample_backtest()
    else:
        print("\nRunning basic backtest example...")
        # Original simple example
start = pd.Timestamp("2019-01-01", tz="utc")
end = pd.Timestamp("2022-01-01", tz="utc")

perf = run_algorithm(
    start=start,
    end=end,
    initialize=initialize,
            capital_base=100000,
    handle_data=handle_data,
    analyze=analyze,
    data_frequency='daily',
            bundle='quantopian-quandl',
            trading_calendar=get_calendar("XNYS")
)


