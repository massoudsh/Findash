import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import backtrader as bt
import os
import sys
import logging
from pathlib import Path
import torch
from PIL import Image
import io

# Add the parent directory to the path so we can import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from M7___Price_Prediction.yolo import load_yolo_model, check_gpu_availability

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualTechnicalAnalysis:
    """Class for combining visual pattern recognition with traditional technical analysis"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize with YOLO model for chart pattern detection"""
        self.device = check_gpu_availability()
        try:
            self.model = load_yolo_model(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
            
    def get_historical_data(self, ticker, period="1y", interval="1d"):
        """Get historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            logger.info(f"Downloaded {len(hist)} data points for {ticker}")
            return hist
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
            
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or len(data) == 0:
            logger.error("No data provided for indicator calculation")
            return None
            
        # Calculate common technical indicators
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        data['BBMiddle'] = data['Close'].rolling(window=20).mean()
        data['BBStd'] = data['Close'].rolling(window=20).std()
        data['BBUpper'] = data['BBMiddle'] + (data['BBStd'] * 2)
        data['BBLower'] = data['BBMiddle'] - (data['BBStd'] * 2)
        
        return data
        
    def plot_chart(self, data, ticker, indicators=None, save_path=None):
        """Plot stock chart with indicators and save it for pattern detection"""
        if data is None or len(data) == 0:
            logger.error("No data provided for plotting")
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Plot main price chart
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['Close'], label='Close Price')
        
        # Plot selected indicators
        if indicators is None:
            indicators = ['SMA20', 'SMA50']
            
        for indicator in indicators:
            if indicator in data.columns:
                plt.plot(data.index, data[indicator], label=indicator)
                
        # Plot Bollinger Bands if available
        if all(band in data.columns for band in ['BBUpper', 'BBMiddle', 'BBLower']):
            plt.plot(data.index, data['BBUpper'], 'r--', label='BB Upper')
            plt.plot(data.index, data['BBMiddle'], 'g--', label='BB Middle')
            plt.plot(data.index, data['BBLower'], 'r--', label='BB Lower')
        
        plt.title(f'{ticker} Stock Price with Indicators')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        plt.title('Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the chart for pattern detection if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Chart saved to {save_path}")
            
        # Return the figure for display
        return plt.gcf()
        
    def detect_chart_patterns(self, chart_image_path):
        """Detect technical patterns in the chart image using YOLO"""
        if self.model is None:
            logger.error("YOLO model not loaded, cannot detect patterns")
            return None
            
        if not os.path.exists(chart_image_path):
            logger.error(f"Chart image not found at {chart_image_path}")
            return None
            
        try:
            # Run detection on the chart image
            results = self.model.predict(chart_image_path, conf=0.25)
            
            patterns = []
            if len(results) > 0:
                for r in results:
                    for i, box in enumerate(r.boxes):
                        if hasattr(r, 'names') and hasattr(box, 'cls'):
                            class_id = int(box.cls.item())
                            if class_id in r.names:
                                pattern_name = r.names[class_id]
                                confidence = float(box.conf.item())
                                patterns.append({
                                    "pattern": pattern_name,
                                    "confidence": confidence
                                })
            
            logger.info(f"Detected {len(patterns)} patterns in chart")
            return patterns
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return None

    def analyze_stock(self, ticker, period="1y", interval="1d"):
        """Complete analysis workflow: fetch data, calculate indicators, detect patterns"""
        # Get data
        data = self.get_historical_data(ticker, period, interval)
        if data is None:
            return None
            
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Generate and save chart
        chart_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, f"{ticker}_analysis.png")
        
        # Plot and save chart
        fig = self.plot_chart(data, ticker, save_path=chart_path)
        
        # Detect patterns
        patterns = self.detect_chart_patterns(chart_path)
        
        # Generate trading signals
        signals = self.generate_signals(data, patterns)
        
        return {
            "data": data,
            "chart_path": chart_path,
            "detected_patterns": patterns,
            "signals": signals
        }
    
    def generate_signals(self, data, patterns=None):
        """Generate trading signals based on technical indicators and detected patterns"""
        if data is None or len(data) < 50:
            return []
            
        signals = []
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # Check for SMA crossovers
        if 'SMA20' in data.columns and 'SMA50' in data.columns:
            # Golden Cross (short-term SMA crosses above long-term SMA)
            if prev_row['SMA20'] <= prev_row['SMA50'] and last_row['SMA20'] > last_row['SMA50']:
                signals.append({
                    "type": "BUY",
                    "reason": "Golden Cross (SMA20 crossed above SMA50)",
                    "strength": "Medium"
                })
            
            # Death Cross (short-term SMA crosses below long-term SMA)
            if prev_row['SMA20'] >= prev_row['SMA50'] and last_row['SMA20'] < last_row['SMA50']:
                signals.append({
                    "type": "SELL",
                    "reason": "Death Cross (SMA20 crossed below SMA50)",
                    "strength": "Medium"
                })
        
        # Check for RSI signals
        if 'RSI' in data.columns:
            # Oversold condition
            if last_row['RSI'] < 30:
                signals.append({
                    "type": "BUY",
                    "reason": f"RSI Oversold ({last_row['RSI']:.2f})",
                    "strength": "Strong" if last_row['RSI'] < 20 else "Medium"
                })
            
            # Overbought condition
            if last_row['RSI'] > 70:
                signals.append({
                    "type": "SELL",
                    "reason": f"RSI Overbought ({last_row['RSI']:.2f})",
                    "strength": "Strong" if last_row['RSI'] > 80 else "Medium"
                })
        
        # Check for MACD signals
        if all(indicator in data.columns for indicator in ['MACD', 'Signal']):
            # MACD crosses above Signal Line
            if prev_row['MACD'] <= prev_row['Signal'] and last_row['MACD'] > last_row['Signal']:
                signals.append({
                    "type": "BUY",
                    "reason": "MACD crossed above Signal Line",
                    "strength": "Medium"
                })
            
            # MACD crosses below Signal Line
            if prev_row['MACD'] >= prev_row['Signal'] and last_row['MACD'] < last_row['Signal']:
                signals.append({
                    "type": "SELL",
                    "reason": "MACD crossed below Signal Line",
                    "strength": "Medium"
                })
        
        # Check for Bollinger Band signals
        if all(band in data.columns for band in ['BBUpper', 'BBMiddle', 'BBLower']):
            # Price below lower band (potential buy)
            if last_row['Close'] < last_row['BBLower']:
                signals.append({
                    "type": "BUY",
                    "reason": "Price below Bollinger Lower Band",
                    "strength": "Medium"
                })
            
            # Price above upper band (potential sell)
            if last_row['Close'] > last_row['BBUpper']:
                signals.append({
                    "type": "SELL",
                    "reason": "Price above Bollinger Upper Band",
                    "strength": "Medium"
                })
        
        # Consider visual patterns if available
        if patterns:
            for pattern in patterns:
                pattern_name = pattern.get("pattern", "").lower()
                confidence = pattern.get("confidence", 0)
                
                # Map common patterns to signals
                if confidence >= 0.5:  # Only consider patterns with confidence >= 50%
                    if any(p in pattern_name for p in ["double_bottom", "inverse_head_shoulders", "bullish_flag"]):
                        signals.append({
                            "type": "BUY",
                            "reason": f"Visual pattern: {pattern_name}",
                            "strength": "Strong" if confidence > 0.7 else "Medium",
                            "confidence": confidence
                        })
                    elif any(p in pattern_name for p in ["double_top", "head_shoulders", "bearish_flag"]):
                        signals.append({
                            "type": "SELL",
                            "reason": f"Visual pattern: {pattern_name}",
                            "strength": "Strong" if confidence > 0.7 else "Medium",
                            "confidence": confidence
                        })
        
        return signals


class PaperTradingSimulator:
    """Simulate paper trading based on signals and historical data"""
    
    def __init__(self, starting_capital=10000.0):
        """Initialize paper trading account"""
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}  # {ticker: {"shares": quantity, "avg_price": price}}
        self.trades = []
        self.equity_history = []
        
    def calculate_portfolio_value(self, current_prices=None):
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        # Add value of all positions
        for ticker, position in self.positions.items():
            if current_prices and ticker in current_prices:
                price = current_prices[ticker]
            else:
                # Get latest price if not provided
                try:
                    price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                except:
                    logger.warning(f"Couldn't get current price for {ticker}, using last known price")
                    price = position.get("last_price", position["avg_price"])
            
            position_value = position["shares"] * price
            portfolio_value += position_value
            
            # Update last known price
            position["last_price"] = price
        
        return portfolio_value
    
    def execute_trade(self, ticker, trade_type, shares, price, timestamp=None, reason=None):
        """Execute a buy or sell trade"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # For buys
        if trade_type.upper() == "BUY":
            cost = shares * price
            if cost > self.cash:
                logger.warning(f"Insufficient funds to buy {shares} shares of {ticker}")
                # Buy as many as possible
                shares = int(self.cash / price)
                if shares == 0:
                    return False
                cost = shares * price
            
            # Update cash
            self.cash -= cost
            
            # Update position
            if ticker in self.positions:
                # Average down
                current_shares = self.positions[ticker]["shares"]
                current_avg = self.positions[ticker]["avg_price"]
                new_shares = current_shares + shares
                new_avg = (current_shares * current_avg + shares * price) / new_shares
                self.positions[ticker] = {
                    "shares": new_shares,
                    "avg_price": new_avg,
                    "last_price": price
                }
            else:
                # New position
                self.positions[ticker] = {
                    "shares": shares,
                    "avg_price": price,
                    "last_price": price
                }
        
        # For sells
        elif trade_type.upper() == "SELL":
            if ticker not in self.positions or self.positions[ticker]["shares"] < shares:
                logger.warning(f"Insufficient shares to sell {shares} shares of {ticker}")
                # Sell all available
                if ticker in self.positions:
                    shares = self.positions[ticker]["shares"]
                else:
                    return False
            
            # Calculate proceeds
            proceeds = shares * price
            
            # Update cash
            self.cash += proceeds
            
            # Update position
            if ticker in self.positions:
                current_shares = self.positions[ticker]["shares"]
                if current_shares == shares:
                    # Sold all shares
                    del self.positions[ticker]
                else:
                    # Partial sell
                    self.positions[ticker]["shares"] = current_shares - shares
                    self.positions[ticker]["last_price"] = price
        
        # Record the trade
        self.trades.append({
            "timestamp": timestamp,
            "ticker": ticker,
            "type": trade_type,
            "shares": shares,
            "price": price,
            "value": shares * price,
            "reason": reason
        })
        
        # Update equity history
        portfolio_value = self.calculate_portfolio_value({ticker: price})
        self.equity_history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })
        
        logger.info(f"Executed {trade_type} of {shares} shares of {ticker} at ${price:.2f}")
        return True
    
    def process_signals(self, ticker, signals, current_price):
        """Process trading signals and execute appropriate trades"""
        for signal in signals:
            signal_type = signal.get("type", "").upper()
            reason = signal.get("reason", "N/A")
            strength = signal.get("strength", "Medium")
            
            # Determine position size based on signal strength
            portfolio_value = self.calculate_portfolio_value({ticker: current_price})
            if strength == "Strong":
                position_pct = 0.1  # 10% of portfolio
            else:  # Medium or Weak
                position_pct = 0.05  # 5% of portfolio
            
            if signal_type == "BUY":
                # Calculate number of shares to buy
                allocation = portfolio_value * position_pct
                shares = int(allocation / current_price)
                
                if shares > 0:
                    self.execute_trade(ticker, "BUY", shares, current_price, reason=reason)
            
            elif signal_type == "SELL":
                # If we have this position, sell it
                if ticker in self.positions and self.positions[ticker]["shares"] > 0:
                    shares = self.positions[ticker]["shares"]
                    self.execute_trade(ticker, "SELL", shares, current_price, reason=reason)
    
    def backtest(self, ticker, start_date, end_date=None, signals_func=None):
        """Backtest a strategy using historical data and signals function"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Reset simulator
        self.cash = self.starting_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        
        # Get historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if len(hist) == 0:
            logger.error(f"No data available for {ticker} from {start_date} to {end_date}")
            return None
        
        # Initialize portfolio tracking
        portfolio_values = []
        dates = []
        
        # Process each day
        for date, row in hist.iterrows():
            current_price = row['Close']
            current_date = date.to_pydatetime()
            
            # Skip if we don't have enough data for signals
            if signals_func is not None:
                # Get signals for this day
                data_until_today = hist.loc[:date]
                signals = signals_func(data_until_today)
                
                # Process signals
                if signals:
                    self.process_signals(ticker, signals, current_price)
            
            # Track portfolio value
            portfolio_value = self.calculate_portfolio_value({ticker: current_price})
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
            
            self.equity_history.append({
                "timestamp": current_date,
                "portfolio_value": portfolio_value
            })
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate daily returns
        daily_returns = [0]
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # Calculate annualized metrics
        daily_returns_arr = np.array(daily_returns[1:])  # Skip the first zero
        annual_return = np.mean(daily_returns_arr) * 252 * 100  # 252 trading days
        annual_volatility = np.std(daily_returns_arr) * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate drawdown
        peak = portfolio_values[0]
        drawdowns = []
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(dates, portfolio_values)
        plt.title(f'Portfolio Value Over Time - {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(dates, drawdowns, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Return performance summary
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": len(self.trades),
            "portfolio_values": portfolio_values,
            "dates": dates,
            "equity_curve": plt.gcf()
        }
    
    def print_performance_summary(self, performance):
        """Print a summary of backtest performance"""
        if performance is None:
            logger.error("No performance data available")
            return
            
        print("\n" + "="*50)
        print(f"PERFORMANCE SUMMARY FOR {performance['ticker']}")
        print("="*50)
        print(f"Period: {performance['start_date']} to {performance['end_date']}")
        print(f"Initial Capital: ${performance['initial_value']:.2f}")
        print(f"Final Capital: ${performance['final_value']:.2f}")
        print(f"Total Return: {performance['total_return']:.2f}%")
        print(f"Annual Return: {performance['annual_return']:.2f}%")
        print(f"Annual Volatility: {performance['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
        print(f"Number of Trades: {performance['trades']}")
        print("="*50)


# Example usage demonstrating the integration of traditional analysis with visual pattern recognition
if __name__ == "__main__":
    # Initialize the visual technical analysis
    vta = VisualTechnicalAnalysis()
    
    # Analyze a stock
    ticker = "AAPL"
    analysis = vta.analyze_stock(ticker, period="1y")
    
    if analysis:
        # Display detected patterns
        if analysis["detected_patterns"]:
            print("\nDetected Chart Patterns:")
            for pattern in analysis["detected_patterns"]:
                print(f"  - {pattern['pattern']} (Confidence: {pattern['confidence']:.2f})")
        
        # Display trading signals
        if analysis["signals"]:
            print("\nTrading Signals:")
            for signal in analysis["signals"]:
                print(f"  - {signal['type']}: {signal['reason']} ({signal['strength']})")
                
        # Show the chart
        plt.figure(figsize=(12, 6))
        plt.plot(analysis["data"].index, analysis["data"]['Close'], label='Close Price')
        plt.plot(analysis["data"].index, analysis["data"]['SMA20'], label='20-day SMA')
        plt.plot(analysis["data"].index, analysis["data"]['SMA50'], label='50-day SMA')
        plt.title(f'{ticker} Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        # Initialize paper trading simulator
        simulator = PaperTradingSimulator(starting_capital=10000.0)
        
        # Backtest strategy
        def signal_generator(data):
            # Generate signals based on SMAs
            signals = []
            if len(data) < 50:
                return signals
                
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else None
            
            # Calculate indicators if they don't exist
            if 'SMA20' not in data.columns:
                data['SMA20'] = data['Close'].rolling(window=20).mean()
            if 'SMA50' not in data.columns:
                data['SMA50'] = data['Close'].rolling(window=50).mean()
                
            if prev_row is not None:
                # Golden Cross
                if prev_row['SMA20'] <= prev_row['SMA50'] and last_row['SMA20'] > last_row['SMA50']:
                    signals.append({
                        "type": "BUY",
                        "reason": "Golden Cross",
                        "strength": "Strong"
                    })
                
                # Death Cross
                if prev_row['SMA20'] >= prev_row['SMA50'] and last_row['SMA20'] < last_row['SMA50']:
                    signals.append({
                        "type": "SELL",
                        "reason": "Death Cross",
                        "strength": "Strong"
                    })
            
            return signals
        
        # Run backtest
        performance = simulator.backtest(
            ticker=ticker,
            start_date="2020-01-01",
            end_date="2023-01-01",
            signals_func=signal_generator
        )
        
        # Print performance summary
        if performance:
            simulator.print_performance_summary(performance)
            
            # Show performance chart
            plt.show()

