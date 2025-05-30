from django.shortcuts import render
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import alpaca_trade_api as tradeapi
import nltk
from textblob import TextBlob
from collections import Counter
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from decouple import config
import logging
from core.logging_config import setup_logging
from core.exceptions import APIError, ScrapingError, DataValidationError, RateLimitError
from core.utils import log_execution_time, handle_api_errors, retry_on_failure
from core.config import config
import os
from core.monitoring import metrics, monitor_performance
from core.models import MarketData, NewsArticle, SocialMetrics
from core.validation import validator
from core.cache import cached
import asyncio
import aiohttp
from functools import lru_cache
from M6.Risk_Management.risk_management_service import run_risk_analysis


api = tradeapi.REST(
    config('ALPACA_API_KEY'),
    config('ALPACA_SECRET_KEY'),
    base_url=config('ALPACA_BASE_URL', default='https://paper-api.alpaca.markets')
)

# Place a market order to buy 10 shares of AAPL
api.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    type='market',
    time_in_force='gtc'
)

# Check your positions
positions = api.list_positions()
for position in positions:
    print(position)

# Initialize logger with config
logger = setup_logging(level=config.get('logging.level'))

# Configure request session with retries and timeouts
def create_request_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    timeout: int = 10,
    pool_connections: int = 100,
    pool_maxsize: int = 100
) -> requests.Session:
    """Create a requests session with connection pooling and retry strategy."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Configure adapter with retry strategy and pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.timeout = timeout
    
    return session

# Create global session
http_session = create_request_session()

# Rate limiting decorator
def rate_limit(calls: int, period: int):
    """Decorator to implement rate limiting."""
    def decorator(func):
        last_reset = datetime.now()
        calls_made = 0
        
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            
            now = datetime.now()
            if (now - last_reset).seconds >= period:
                calls_made = 0
                last_reset = now
                
            if calls_made >= calls:
                raise RateLimitError(f"Rate limit exceeded: {calls} calls per {period} seconds")
                
            calls_made += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

def build_market_data_key(symbol: str) -> str:
    """Build cache key for market data."""
    return f"market_data:{symbol}:{datetime.now().strftime('%Y-%m-%d:%H')}"

@cached(
    ttl=300,  # 5 minutes
    key_builder=build_market_data_key
)
@monitor_performance('fetch_market_data')
@log_execution_time
@handle_api_errors
@rate_limit(calls=5, period=60)  # 5 calls per minute
@retry_on_failure(
    retries=config.get('scraping.retry_attempts'),
    delay=config.get('scraping.retry_delay')
)
def fetch_real_time_data(symbol: str) -> MarketData:
    """Fetch real-time financial data with caching."""
    try:
        with metrics.track_operation_time('api_request'):
            url = config.get('api.alpha_vantage.base_url')
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "1min",
                "apikey": os.getenv('ALPHA_VANTAGE_API_KEY')
            }
            
            response = http_session.get(url, params=params)
            response.raise_for_status()
            
        with metrics.track_operation_time('data_processing'):
            data = response.json()
            if "Time Series (1min)" not in data:
                metrics.track_error('alpha_vantage', 'data_not_found')
                raise APIError("No time series data available")
                
            time_series = data['Time Series (1min)']
            latest_time = next(iter(time_series.keys()))
            latest_data = time_series[latest_time]
            
            # Validate data using Pydantic model
            market_data = validator.validate_data({
                'symbol': symbol,
                'price': latest_data['4. close'],
                'volume': int(latest_data['5. volume']),
                'timestamp': datetime.fromisoformat(latest_time),
                'exchange': 'NASDAQ'  # or get from config
            }, MarketData)
            
            return market_data
            
    except requests.exceptions.RequestException as e:
        metrics.track_error('alpha_vantage', 'request_failed')
        raise APIError(f"API request failed: {str(e)}")
    except (KeyError, StopIteration) as e:
        metrics.track_error('alpha_vantage', 'parsing_failed')
        raise APIError(f"Data parsing failed: {str(e)}")
    except ValidationError as e:
        metrics.track_error('alpha_vantage', 'validation_failed')
        raise DataValidationError(f"Data validation failed: {str(e)}")
    except RateLimitError as e:
        metrics.track_error('alpha_vantage', 'rate_limit_exceeded')
        raise

# Function to scrape a fintech platform for news using BeautifulSoup
def scrape_fintech_news():
    url = "https://www.finextra.com/news/latestannouncements.aspx"  # Finextra, a fintech news site
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        news_items = soup.find_all('div', class_='item')

        news_list = []
        for item in news_items[:5]:  # Fetch the top 5 news items
            title = item.find('a').get_text(strip=True)
            summary = item.find('div', class_='summary').get_text(strip=True)
            news_list.append({'title': title, 'summary': summary})

        return news_list
    return {"error": "Failed to retrieve news"}

# Function to scrape a fintech dashboard dynamically using Selenium
def scrape_fintech_dashboard() -> List[str]:
    """Scrape fintech dashboard using Selenium with proper resource management."""
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    service = Service(config('CHROME_DRIVER_PATH'))
    
    try:
        with webdriver.Chrome(service=service, options=options) as driver:
            driver.get(config('DASHBOARD_URL'))
            
            # Wait for elements to load
            wait = WebDriverWait(driver, 10)
            stock_elements = wait.until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'stock-price'))
            )
            
            return [stock.text for stock in stock_elements]
    except Exception as e:
        logging.error(f"Dashboard scraping failed: {str(e)}")
        return []

# === NEW: Tweetipy integration ===
def fetch_tweets(query, tweet_count=10):
    """
    Fetch recent tweets based on the provided query using Tweetipy.
    Replace the placeholder credentials with your actual Tweetipy credentials.
    """
    # Import TweetAPI from tweetipy. Adjust this import if your tweetipy library has a different structure.
    from tweetipy import TweetAPI

    # Initialize the TweetAPI client
    client = TweetAPI(
        api_key='YOUR_TWEETIPY_API_KEY',
        api_secret='YOUR_TWEETIPY_API_SECRET',
        access_token='YOUR_TWEETIPY_ACCESS_TOKEN',
        access_token_secret='YOUR_TWEETIPY_ACCESS_TOKEN_SECRET'
    )
    
    # Search for tweets. The exact method and parameters may vary based on the Tweetipy implementation.
    tweets = client.search(query, count=tweet_count)
    tweet_list = []
    for tweet in tweets:
        tweet_list.append({
            'text': tweet.text,
            'username': tweet.user.screen_name,
            'created_at': tweet.created_at
        })
    return tweet_list

# === NEW: Discord integration ===
def fetch_discord_messages(channel_id, token, limit=10):
    """
    Fetch recent messages from a Discord channel using Discord's API.
    Ensure your bot is added to the server and has the necessary permissions.
    """
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit={limit}"
    headers = {"Authorization": f"Bot {token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        messages = response.json()
        message_list = []
        for message in messages:
            message_list.append({
                'content': message.get('content'),
                'author': message.get('author', {}).get('username'),
                'timestamp': message.get('timestamp')
            })
        return message_list
    return {"error": "Failed to fetch messages from Discord"}

class SocialMetricsAnalyzer:
    """A comprehensive social metrics analyzer for trading signals."""
    
    def __init__(self):
        try:
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logging.error(f"Failed to initialize NLTK: {str(e)}")
            raise

    @staticmethod
    def safe_division(n: float, d: float) -> float:
        """Safely perform division with fallback to 0."""
        return n / d if d != 0 else 0

    def analyze_reddit_sentiment(self, subreddit: str, timeframe: str = 'day') -> Dict:
        """
        Analyze sentiment from Reddit posts and comments
        """
        # You'll need to set up Reddit API credentials
        import praw
        
        reddit = praw.Reddit(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
            user_agent="YOUR_USER_AGENT"
        )
        
        subreddit = reddit.subreddit(subreddit)
        posts = subreddit.top(time_filter=timeframe, limit=100)
        
        sentiment_scores = []
        mention_counts = Counter()
        
        for post in posts:
            # Analyze post title and body
            sentiment_scores.append(self.sia.polarity_scores(post.title)['compound'])
            if post.selftext:
                sentiment_scores.append(self.sia.polarity_scores(post.selftext)['compound'])
            
            # Count ticker mentions
            tickers = re.findall(r'\$([A-Z]+)', post.title + post.selftext)
            mention_counts.update(tickers)
        
        return {
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores),
            'top_mentions': dict(mention_counts.most_common(10))
        }

    def analyze_stocktwits_buzz(self, symbol: str) -> Dict:
        """
        Analyze StockTwits messages for the given symbol
        """
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            messages = data['messages']
            
            # Analyze message volume
            message_count = len(messages)
            
            # Calculate sentiment
            bullish_count = sum(1 for msg in messages if msg.get('entities', {}).get('sentiment', {}).get('basic') == 'Bullish')
            bearish_count = sum(1 for msg in messages if msg.get('entities', {}).get('sentiment', {}).get('basic') == 'Bearish')
            
            return {
                'message_volume': message_count,
                'bullish_ratio': bullish_count / message_count if message_count > 0 else 0,
                'bearish_ratio': bearish_count / message_count if message_count > 0 else 0
            }
        return {"error": "Failed to fetch StockTwits data"}

    def analyze_google_trends(self, keyword: str) -> Dict:
        """
        Analyze Google Trends data for trading-related keywords
        """
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 7-d')
        
        interest_over_time = pytrends.interest_over_time()
        if not interest_over_time.empty:
            return {
                'current_interest': interest_over_time[keyword].iloc[-1],
                'weekly_average': interest_over_time[keyword].mean(),
                'trend_direction': 'up' if interest_over_time[keyword].iloc[-1] > interest_over_time[keyword].iloc[0] else 'down'
            }
        return {"error": "No Google Trends data available"}

    def calculate_social_momentum(self, symbol: str) -> Dict:
        """
        Calculate social momentum across platforms
        """
        reddit_data = self.analyze_reddit_sentiment(f"wallstreetbets")  # Add more relevant subreddits
        stocktwits_data = self.analyze_stocktwits_buzz(symbol)
        google_trends = self.analyze_google_trends(symbol)
        
        # Combine metrics into a momentum score
        momentum_score = 0
        if 'average_sentiment' in reddit_data:
            momentum_score += reddit_data['average_sentiment']
        if 'bullish_ratio' in stocktwits_data:
            momentum_score += (stocktwits_data['bullish_ratio'] - stocktwits_data['bearish_ratio'])
        if 'current_interest' in google_trends:
            momentum_score += google_trends['current_interest'] / 100  # Normalize to 0-1 scale
            
        return {
            'momentum_score': momentum_score,
            'reddit_sentiment': reddit_data,
            'stocktwits_metrics': stocktwits_data,
            'google_trends': google_trends
        }

class SocialMediaCredentials:
    """Secure credential management for social media APIs."""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_twitter_creds() -> Dict[str, str]:
        """Get Twitter API credentials from environment variables."""
        return {
            'api_key': config('TWITTER_API_KEY'),
            'api_secret': config('TWITTER_API_SECRET'),
            'access_token': config('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': config('TWITTER_ACCESS_TOKEN_SECRET')
        }
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_discord_creds() -> Dict[str, str]:
        """Get Discord API credentials from environment variables."""
        return {
            'bot_token': config('DISCORD_BOT_TOKEN'),
            'channel_id': config('DISCORD_CHANNEL_ID')
        }

async def fetch_social_data(symbol: str) -> Dict[str, Any]:
    """Fetch social media data asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_tweets_async(session, symbol),
            fetch_discord_messages_async(session),
            fetch_reddit_data_async(session, symbol)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'twitter_data': results[0] if not isinstance(results[0], Exception) else None,
            'discord_data': results[1] if not isinstance(results[1], Exception) else None,
            'reddit_data': results[2] if not isinstance(results[2], Exception) else None
        }

@rate_limit(calls=100, period=900)  # 100 calls per 15 minutes
async def fetch_tweets_async(session: aiohttp.ClientSession, query: str) -> List[Dict]:
    """Async function to fetch tweets."""
    creds = SocialMediaCredentials.get_twitter_creds()
    headers = {
        'Authorization': f"Bearer {creds['api_key']}",
        'User-Agent': 'FinanceBot/1.0'
    }
    
    async with session.get(
        f"https://api.twitter.com/2/tweets/search/recent?query={query}",
        headers=headers
    ) as response:
        if response.status == 429:
            raise RateLimitError("Twitter API rate limit exceeded")
        data = await response.json()
        return data.get('data', [])

@rate_limit(calls=50, period=60)  # 50 calls per minute
async def fetch_discord_messages_async(session: aiohttp.ClientSession) -> List[Dict]:
    """Async function to fetch Discord messages."""
    creds = SocialMediaCredentials.get_discord_creds()
    headers = {
        'Authorization': f"Bot {creds['bot_token']}",
        'User-Agent': 'FinanceBot/1.0'
    }
    
    url = f"https://discord.com/api/v9/channels/{creds['channel_id']}/messages"
    async with session.get(url, headers=headers) as response:
        if response.status == 429:
            raise RateLimitError("Discord API rate limit exceeded")
        return await response.json()

async def fetch_reddit_data_async(session: aiohttp.ClientSession, symbol: str) -> Dict:
    """Async function to fetch Reddit data."""
    headers = {'User-Agent': 'FinanceBot/1.0'}
    url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={symbol}&restrict_sr=1"
    
    async with session.get(url, headers=headers) as response:
        if response.status == 429:
            raise RateLimitError("Reddit API rate limit exceeded")
        data = await response.json()
        return data.get('data', {}).get('children', [])

# Update the main view to use async data fetching
async def fintech_dashboard_view(request):
    symbol = 'AAPL'  # Example symbol
    
    # Fetch market data
    real_time_data = fetch_real_time_data(symbol)
    
    # Fetch social data asynchronously
    social_data = await fetch_social_data(symbol)
    
    # Analyze social metrics
    analyzer = SocialMetricsAnalyzer()
    social_metrics = analyzer.calculate_social_momentum(symbol)
    
    context = {
        'real_time_data': real_time_data,
        'social_data': social_data,
        'social_metrics': social_metrics,
    }
    
    return render(request, 'dashboard.html', context)

# Optional: Add a periodic task to track social metrics over time
def track_social_metrics_history(symbol: str):
    """
    Track and store social metrics history for analysis
    """
    analyzer = SocialMetricsAnalyzer()
    metrics = analyzer.calculate_social_momentum(symbol)
    
    # Store in your database (adjust model/table name as needed)
    SocialMetricsHistory.objects.create(
        symbol=symbol,
        timestamp=datetime.now(),
        momentum_score=metrics['momentum_score'],
        reddit_sentiment=metrics['reddit_sentiment']['average_sentiment'],
        stocktwits_bullish_ratio=metrics['stocktwits_metrics'].get('bullish_ratio', 0),
        google_trends_interest=metrics['google_trends'].get('current_interest', 0)
    )

@cached(ttl=3600)  # 1 hour
def fetch_news_articles(symbol: str) -> List[NewsArticle]:
    """Fetch news articles with caching."""
    # Implementation...

@cached(ttl=1800)  # 30 minutes
def get_social_metrics(symbol: str) -> SocialMetrics:
    """Fetch social metrics with caching."""
    # Implementation...

results = run_risk_analysis(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    optimization_method='both',
    report_format='html'
)

    