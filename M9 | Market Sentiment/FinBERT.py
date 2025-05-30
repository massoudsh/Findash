import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Import from other modules (system integration)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from M8___Paper_Trading.Paper_Trading import PaperTradingSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialTextAnalyzer:
    """
    Advanced financial text analysis using FinBERT.
    
    This class implements sentiment analysis for financial texts, including:
    - Individual text analysis
    - Batch processing of multiple texts
    - Sentiment trend analysis
    - News impact scoring
    - Trading signal generation based on sentiment
    - Visualization of sentiment trends
    """
    
    def __init__(self, model_path: str = "ProsusAI/finbert", cache_dir: Optional[str] = None):
        """
        Initialize the financial text analyzer.
        
        Args:
            model_path: Path or name of the model to load
            cache_dir: Directory to cache the model
        """
        logger.info(f"Initializing FinancialTextAnalyzer with model: {model_path}")
        
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model and tokenizer
        self.model_name = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Labels for financial sentiment
        self.labels = ['negative', 'neutral', 'positive']
        
        # Store historical sentiment data
        self.sentiment_history = defaultdict(list)  # {ticker: [sentiment_records]}
        
        # Ticker mapping
        self.ticker_aliases = self._load_ticker_aliases()

    def _load_ticker_aliases(self) -> Dict[str, str]:
        """
        Load mapping of company names to ticker symbols.
        
        Returns:
            Dictionary mapping company name variations to ticker symbols
        """
        # Default basic mapping - in production, this would load from a comprehensive database
        basic_mapping = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'amazon': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'meta': 'META',
            'facebook': 'META',
            'tesla': 'TSLA',
            'jpmorgan': 'JPM',
            'jp morgan': 'JPM',
            'visa': 'V',
            'berkshire': 'BRK-B',
            'berkshire hathaway': 'BRK-B',
            'walmart': 'WMT',
            'nvidia': 'NVDA',
            'johnson & johnson': 'JNJ',
            'johnson and johnson': 'JNJ',
            'exxon': 'XOM',
            'exxonmobil': 'XOM',
            'exxon mobil': 'XOM',
            'bank of america': 'BAC',
            'procter & gamble': 'PG',
            'procter and gamble': 'PG',
            'mastercard': 'MA',
            'disney': 'DIS',
            'walt disney': 'DIS',
            'coca cola': 'KO',
            'coca-cola': 'KO',
            'pfizer': 'PFE',
            'netflix': 'NFLX',
            'verizon': 'VZ',
            'at&t': 'T',
            'comcast': 'CMCSA',
            'intel': 'INTC',
            'cisco': 'CSCO',
            'adobe': 'ADBE',
            'paypal': 'PYPL',
            'salesforce': 'CRM'
        }
        
        # Try to load extended mappings from file
        mapping_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ticker_aliases.json")
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    extended_mapping = json.load(f)
                    basic_mapping.update(extended_mapping)
                    logger.info(f"Loaded {len(extended_mapping)} ticker aliases from file")
        except Exception as e:
            logger.warning(f"Could not load ticker aliases: {e}")
        
        return basic_mapping

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a financial text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Dictionary with sentiment scores (negative, neutral, positive)
        """
        # Prepare the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert predictions to dictionary
            scores = {label: score.item() for label, score in zip(self.labels, predictions[0])}
            return scores
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {label: 0.0 for label in self.labels}

    def analyze_texts(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Analyze multiple financial texts efficiently.
        
        Args:
            texts: List of financial texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with sentiment scores
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch_texts)} texts)")
            
            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert predictions to dictionaries
                batch_results = []
                for pred in predictions:
                    scores = {label: score.item() for label, score in zip(self.labels, pred)}
                    batch_results.append(scores)
                
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error analyzing batch: {e}")
                # Add placeholder results on error
                placeholder = {label: 0.0 for label in self.labels}
                results.extend([placeholder] * len(batch_texts))
        
        return results

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """
        Extract ticker symbols mentioned in text.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            List of ticker symbols found
        """
        # First look for standard ticker formats ($AAPL, $MSFT, etc.)
        ticker_pattern = r'\$([A-Z]{1,5})'
        tickers = re.findall(ticker_pattern, text)
        
        # Then look for company names in our alias mapping
        text_lower = text.lower()
        for company, ticker in self.ticker_aliases.items():
            if company in text_lower and ticker not in tickers:
                tickers.append(ticker)
        
        return tickers

    def analyze_news_with_impact(self, 
                               news_items: List[Dict[str, str]],
                               source_weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """
        Analyze news items with source weighting for impact assessment.
        
        Args:
            news_items: List of news items (each with 'text', 'source', 'date' keys)
            source_weights: Dictionary mapping sources to impact weights
            
        Returns:
            Dictionary mapping tickers to sentiment impact
        """
        if source_weights is None:
            # Default weights for news sources
            source_weights = {
                'bloomberg': 1.5,
                'reuters': 1.5,
                'wsj': 1.4,
                'financial_times': 1.4,
                'cnbc': 1.2,
                'yahoo_finance': 1.0,
                'seeking_alpha': 0.8
            }
            # Default weight for sources not in the mapping
            default_weight = 1.0
        
        # Process news items
        ticker_sentiment = defaultdict(lambda: {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'count': 0})
        
        for item in news_items:
            text = item.get('text', '')
            source = item.get('source', '').lower()
            date = item.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Get source weight
            weight = source_weights.get(source, default_weight)
            
            # Extract tickers
            tickers = self.extract_tickers_from_text(text)
            
            # If no tickers found, use general market
            if not tickers:
                tickers = ['MARKET']
            
            # Analyze sentiment
            sentiment = self.analyze_text(text)
            
            # Record sentiment for each ticker
            for ticker in tickers:
                # Update aggregated sentiment
                for label in self.labels:
                    ticker_sentiment[ticker][label] += sentiment[label] * weight
                ticker_sentiment[ticker]['count'] += 1
                
                # Store in history with date
                self.sentiment_history[ticker].append({
                    'date': date,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'source': source,
                    'sentiment': sentiment,
                    'weight': weight
                })
        
        # Calculate average sentiment
        results = {}
        for ticker, data in ticker_sentiment.items():
            count = data['count']
            if count > 0:
                # Calculate weighted averages
                avg_sentiment = {
                    label: data[label] / count for label in self.labels
                }
                
                # Calculate sentiment score from -1 to 1
                sentiment_score = avg_sentiment['positive'] - avg_sentiment['negative']
                
                results[ticker] = {
                    'average_sentiment': avg_sentiment,
                    'sentiment_score': sentiment_score,
                    'mention_count': count
                }
        
        return results

    def generate_trading_signals(self, 
                              sentiment_results: Dict[str, Dict],
                              threshold_positive: float = 0.6,
                              threshold_negative: float = 0.6,
                              min_mentions: int = 3) -> Dict[str, List[Dict]]:
        """
        Generate trading signals based on sentiment analysis.
        
        Args:
            sentiment_results: Results from analyze_news_with_impact
            threshold_positive: Threshold for positive sentiment to generate BUY signal
            threshold_negative: Threshold for negative sentiment to generate SELL signal
            min_mentions: Minimum number of mentions required for signal generation
            
        Returns:
            Dictionary mapping tickers to trading signals
        """
        signals = {}
        
        for ticker, data in sentiment_results.items():
            ticker_signals = []
            sentiment_score = data['sentiment_score']
            mention_count = data['mention_count']
            
            # Skip tickers with insufficient mentions
            if mention_count < min_mentions:
                continue
            
            avg_sentiment = data['average_sentiment']
            
            # Generate BUY signal for strong positive sentiment
            if avg_sentiment['positive'] > threshold_positive and sentiment_score > 0.3:
                signal_strength = "Strong" if sentiment_score > 0.5 else "Medium"
                ticker_signals.append({
                    "type": "BUY",
                    "reason": f"Strong positive sentiment ({sentiment_score:.2f}) from {mention_count} mentions",
                    "strength": signal_strength,
                    "confidence": avg_sentiment['positive']
                })
            
            # Generate SELL signal for strong negative sentiment
            elif avg_sentiment['negative'] > threshold_negative and sentiment_score < -0.3:
                signal_strength = "Strong" if sentiment_score < -0.5 else "Medium"
                ticker_signals.append({
                    "type": "SELL",
                    "reason": f"Strong negative sentiment ({sentiment_score:.2f}) from {mention_count} mentions",
                    "strength": signal_strength,
                    "confidence": avg_sentiment['negative']
                })
            
            if ticker_signals:
                signals[ticker] = ticker_signals
        
        return signals

    def plot_sentiment_trends(self, ticker: str, days: int = 30, save_path: Optional[str] = None) -> None:
        """
        Plot sentiment trends for a specific ticker.
        
        Args:
            ticker: Ticker symbol to plot
            days: Number of days to include in the plot
            save_path: Path to save the plot (if None, display only)
        """
        if ticker not in self.sentiment_history:
            logger.error(f"No sentiment history available for {ticker}")
            return
        
        # Get sentiment history for the ticker
        history = self.sentiment_history[ticker]
        
        # Convert dates to datetime and sort
        for item in history:
            if isinstance(item['date'], str):
                item['date'] = datetime.strptime(item['date'], '%Y-%m-%d')
        
        # Sort by date
        history = sorted(history, key=lambda x: x['date'])
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        history = [item for item in history if item['date'] >= cutoff_date]
        
        if not history:
            logger.error(f"No recent sentiment data available for {ticker}")
            return
        
        # Extract data for plotting
        dates = [item['date'] for item in history]
        positive = [item['sentiment']['positive'] for item in history]
        negative = [item['sentiment']['negative'] for item in history]
        neutral = [item['sentiment']['neutral'] for item in history]
        
        # Calculate moving averages
        window = 5  # 5-day moving average
        if len(history) >= window:
            positive_ma = pd.Series(positive).rolling(window=window).mean().tolist()
            negative_ma = pd.Series(negative).rolling(window=window).mean().tolist()
        else:
            positive_ma = positive
            negative_ma = negative
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot raw sentiment scores
        plt.subplot(2, 1, 1)
        plt.scatter(dates, positive, c='green', alpha=0.6, label='Positive')
        plt.scatter(dates, negative, c='red', alpha=0.6, label='Negative')
        plt.scatter(dates, neutral, c='gray', alpha=0.6, label='Neutral')
        plt.title(f'Sentiment Analysis for {ticker} - Last {days} Days')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot moving averages
        plt.subplot(2, 1, 2)
        plt.plot(dates, positive_ma, c='green', label=f'Positive ({window}-day MA)')
        plt.plot(dates, negative_ma, c='red', label=f'Negative ({window}-day MA)')
        
        # Plot sentiment score
        sentiment_score = [p - n for p, n in zip(positive, negative)]
        plt.plot(dates, sentiment_score, c='blue', label='Sentiment Score (Pos-Neg)')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.fill_between(dates, sentiment_score, 0, where=[s > 0 for s in sentiment_score], 
                         color='green', alpha=0.3)
        plt.fill_between(dates, sentiment_score, 0, where=[s <= 0 for s in sentiment_score], 
                         color='red', alpha=0.3)
        
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Sentiment plot saved to {save_path}")
        else:
            plt.show()

    def save_sentiment_data(self, filename: Optional[str] = None) -> str:
        """
        Save sentiment history to disk.
        
        Args:
            filename: Filename to save data (if None, generates based on timestamp)
            
        Returns:
            Path to saved file
        """
        if not self.sentiment_history:
            logger.error("No sentiment history to save")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_history_{timestamp}.json"
        
        # Create full path
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            # Prepare data for serialization (convert datetime objects to strings)
            save_data = {}
            for ticker, history in self.sentiment_history.items():
                save_data[ticker] = []
                for item in history:
                    item_copy = item.copy()
                    if isinstance(item_copy['date'], datetime):
                        item_copy['date'] = item_copy['date'].strftime('%Y-%m-%d')
                    save_data[ticker].append(item_copy)
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            
            logger.info(f"Sentiment history saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            return None

    def load_sentiment_data(self, file_path: str) -> bool:
        """
        Load sentiment history from disk.
        
        Args:
            file_path: Path to saved sentiment data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Clear existing data
            self.sentiment_history.clear()
            
            # Load data
            for ticker, history in loaded_data.items():
                self.sentiment_history[ticker] = history
            
            logger.info(f"Loaded sentiment history for {len(self.sentiment_history)} tickers from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return False

    def backtest_sentiment_strategy(self, 
                                  ticker: str,
                                  price_data: pd.DataFrame,
                                  sentiment_threshold: float = 0.3,
                                  position_size: float = 0.1,
                                  initial_capital: float = 10000.0) -> Dict:
        """
        Backtest a trading strategy based on sentiment analysis.
        
        Args:
            ticker: Ticker symbol to backtest
            price_data: DataFrame with price data (must have datetime index and 'Close' column)
            sentiment_threshold: Threshold for sentiment score to generate signals
            position_size: Portion of capital to allocate per trade
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        if ticker not in self.sentiment_history:
            logger.error(f"No sentiment history available for {ticker}")
            return None
        
        # Get sentiment history for the ticker
        history = self.sentiment_history[ticker]
        
        # Convert sentiment history to DataFrame
        sentiment_data = []
        for item in history:
            if isinstance(item['date'], str):
                date = datetime.strptime(item['date'], '%Y-%m-%d')
            else:
                date = item['date']
            
            sentiment_score = item['sentiment']['positive'] - item['sentiment']['negative']
            sentiment_data.append({
                'date': date,
                'sentiment_score': sentiment_score,
                'source': item['source'],
                'weight': item.get('weight', 1.0)
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Group by date and calculate weighted average sentiment
        sentiment_daily = sentiment_df.groupby(sentiment_df['date'].dt.date).apply(
            lambda x: pd.Series({
                'sentiment_score': np.average(x['sentiment_score'], weights=x['weight']),
                'count': len(x)
            })
        )
        
        # Convert to DataFrame if result is a Series
        if isinstance(sentiment_daily, pd.Series):
            sentiment_daily = sentiment_daily.reset_index()
        
        # Convert date to datetime for joining
        sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
        sentiment_daily = sentiment_daily.set_index('date')
        
        # Make sure price_data index is datetime
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data = price_data.set_index(pd.DatetimeIndex(price_data.index))
        
        # Join sentiment data with price data
        analysis_df = price_data.join(sentiment_daily, how='left')
        
        # Forward fill sentiment scores for days without news
        analysis_df['sentiment_score'] = analysis_df['sentiment_score'].fillna(method='ffill')
        analysis_df['count'] = analysis_df['count'].fillna(0)
        
        # Generate trading signals
        def generate_signals(data_slice):
            signals = []
            latest_sentiment = data_slice['sentiment_score'].iloc[-1]
            
            if pd.notna(latest_sentiment):
                if latest_sentiment > sentiment_threshold:
                    signals.append({
                        "type": "BUY",
                        "reason": f"Positive sentiment ({latest_sentiment:.2f})",
                        "strength": "Strong" if latest_sentiment > 0.5 else "Medium"
                    })
                elif latest_sentiment < -sentiment_threshold:
                    signals.append({
                        "type": "SELL",
                        "reason": f"Negative sentiment ({latest_sentiment:.2f})",
                        "strength": "Strong" if latest_sentiment < -0.5 else "Medium"
                    })
            
            return signals
        
        # Create a paper trading simulator
        simulator = PaperTradingSimulator(starting_capital=initial_capital)
        
        # Run backtest
        backtest_results = simulator.backtest(
            ticker=ticker,
            start_date=analysis_df.index[0].strftime("%Y-%m-%d"),
            end_date=analysis_df.index[-1].strftime("%Y-%m-%d"),
            signals_func=generate_signals
        )
        
        return backtest_results

    def integrate_with_trading_platform(self, tickers: List[str], news_data: List[Dict]) -> Dict[str, List]:
        """
        Integrate sentiment analysis with the trading platform.
        
        Args:
            tickers: List of ticker symbols to analyze
            news_data: List of news items related to the tickers
            
        Returns:
            Dictionary mapping tickers to trading signals
        """
        # Filter news for the tickers of interest
        filtered_news = []
        for news in news_data:
            text = news.get('text', '')
            extracted_tickers = self.extract_tickers_from_text(text)
            if any(ticker in extracted_tickers for ticker in tickers) or not extracted_tickers:
                filtered_news.append(news)
        
        # Analyze news sentiment with impact assessment
        sentiment_results = self.analyze_news_with_impact(filtered_news)
        
        # Generate trading signals
        trading_signals = self.generate_trading_signals(
            sentiment_results,
            threshold_positive=0.6,
            threshold_negative=0.6,
            min_mentions=2
        )
        
        # Log results
        for ticker, signals in trading_signals.items():
            for signal in signals:
                logger.info(f"Generated {signal['type']} signal for {ticker}: {signal['reason']}")
        
        return trading_signals


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FinancialTextAnalyzer()
    
    # Example financial texts
    texts = [
        "Apple reported strong earnings, beating market expectations with impressive iPhone sales.",
        "Tesla stock plummeted after the disappointing quarterly results and production delays.",
        "The market remained stable throughout the trading session despite mixed economic data.",
        "Microsoft's cloud business continues to show robust growth, lifting the company's outlook.",
        "Amazon faces increasing regulatory scrutiny as lawmakers investigate its business practices."
    ]
    
    # Analyze individual text
    single_result = analyzer.analyze_text(texts[0])
    print("\nSingle text analysis:")
    print(f"Text: {texts[0]}")
    print("Results:", single_result)
    
    # Analyze multiple texts
    results = analyzer.analyze_texts(texts)
    print("\nMultiple texts analysis:")
    for text, result in zip(texts, results):
        print(f"\nText: {text}")
        print("Results:", result)
    
    # Example news items with sources
    news_items = [
        {
            'text': texts[0],
            'source': 'bloomberg',
            'date': '2023-01-15'
        },
        {
            'text': texts[1],
            'source': 'cnbc',
            'date': '2023-01-15'
        },
        {
            'text': texts[3],
            'source': 'wall_street_journal',
            'date': '2023-01-16'
        },
        {
            'text': texts[4],
            'source': 'reuters',
            'date': '2023-01-16'
        }
    ]
    
    # Analyze news with impact assessment
    sentiment_results = analyzer.analyze_news_with_impact(news_items)
    print("\nSentiment analysis with impact assessment:")
    for ticker, data in sentiment_results.items():
        print(f"\nTicker: {ticker}")
        print(f"Sentiment Score: {data['sentiment_score']:.4f}")
        print(f"Mentions: {data['mention_count']}")
        print(f"Average Sentiment: {data['average_sentiment']}")
    
    # Generate trading signals
    signals = analyzer.generate_trading_signals(sentiment_results)
    print("\nTrading signals:")
    for ticker, ticker_signals in signals.items():
        print(f"\nTicker: {ticker}")
        for signal in ticker_signals:
            print(f"  {signal['type']}: {signal['reason']} ({signal['strength']})")
    
    # Save sentiment data
    analyzer.save_sentiment_data()
    
    # Plot sentiment trends for Apple
    if 'AAPL' in analyzer.sentiment_history:
        analyzer.plot_sentiment_trends('AAPL', days=5, save_path="apple_sentiment.png")
