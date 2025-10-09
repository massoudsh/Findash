"""
M9 - Market Sentiment Agent
FinBERT-based sentiment analysis, social media processing, and market sentiment indicators.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import re
import json

# NLP and sentiment analysis
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Web scraping and APIs
import aiohttp
import feedparser
from bs4 import BeautifulSoup

# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats

from ..core.cache import TradingCache
from ..core.exceptions import TradingError, MLModelError


@dataclass
class SentimentScore:
    """Individual sentiment score with metadata."""
    source: str
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    score: float  # -1 to 1
    timestamp: datetime
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,  # Truncate for storage
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "metadata": self.metadata
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment analysis for a symbol."""
    symbol: str
    overall_sentiment: str
    sentiment_score: float  # -1 to 1
    confidence: float
    volume_weighted_score: float
    source_breakdown: Dict[str, float]
    sentiment_trend: str  # "improving", "declining", "stable"
    fear_greed_index: float  # 0 to 100
    social_sentiment: float
    news_sentiment: float
    market_sentiment: float
    timestamp: datetime
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "volume_weighted_score": self.volume_weighted_score,
            "source_breakdown": self.source_breakdown,
            "sentiment_trend": self.sentiment_trend,
            "fear_greed_index": self.fear_greed_index,
            "social_sentiment": self.social_sentiment,
            "news_sentiment": self.news_sentiment,
            "market_sentiment": self.market_sentiment,
            "timestamp": self.timestamp.isoformat(),
            "sample_size": self.sample_size
        }


class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize FinBERT model."""
        try:
            # Use FinBERT model for financial sentiment
            model_name = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            self.initialized = True
            logging.info("FinBERT model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing FinBERT: {e}")
            # Fallback to basic sentiment analyzer
            self.pipeline = None
            self.initialized = False
    
    async def analyze_sentiment(self, text: str) -> Tuple[str, float, float]:
        """
        Analyze sentiment of financial text.
        Returns: (sentiment, confidence, score)
        """
        try:
            if not self.initialized or not self.pipeline:
                # Fallback to TextBlob
                return await self._fallback_sentiment(text)
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if len(cleaned_text.strip()) < 10:
                return "neutral", 0.5, 0.0
            
            # Analyze with FinBERT
            result = self.pipeline(cleaned_text)[0]
            
            # Map FinBERT labels
            label_mapping = {
                "positive": "positive",
                "negative": "negative", 
                "neutral": "neutral",
                "POSITIVE": "positive",
                "NEGATIVE": "negative",
                "NEUTRAL": "neutral"
            }
            
            sentiment = label_mapping.get(result['label'].lower(), "neutral")
            confidence = result['score']
            
            # Convert to score (-1 to 1)
            if sentiment == "positive":
                score = confidence
            elif sentiment == "negative":
                score = -confidence
            else:
                score = 0.0
            
            return sentiment, confidence, score
            
        except Exception as e:
            logging.error(f"Error in FinBERT analysis: {e}")
            return await self._fallback_sentiment(text)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (but keep the content)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate to model's max length (512 tokens for BERT)
        words = text.split()
        if len(words) > 100:  # Conservative limit
            text = ' '.join(words[:100])
        
        return text
    
    async def _fallback_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Fallback sentiment analysis using TextBlob and VADER."""
        try:
            # TextBlob analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # VADER analysis
            analyzer = SentimentIntensityAnalyzer()
            vader_scores = analyzer.polarity_scores(text)
            compound = vader_scores['compound']
            
            # Combine scores
            combined_score = (polarity + compound) / 2
            
            # Determine sentiment
            if combined_score > 0.1:
                sentiment = "positive"
            elif combined_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = min(0.8, abs(combined_score) + 0.1)  # Lower confidence for fallback
            
            return sentiment, confidence, combined_score
            
        except Exception as e:
            logging.error(f"Error in fallback sentiment analysis: {e}")
            return "neutral", 0.1, 0.0


class NewsAggregator:
    """News aggregator for financial sentiment analysis."""
    
    def __init__(self):
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.marketwatch.com/rss/topstories",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
        
        # Custom news APIs (if available)
        self.api_sources = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "news_api": "https://newsapi.org/v2/everything"
        }
    
    async def fetch_news(self, symbol: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch news articles related to a symbol."""
        news_articles = []
        
        try:
            # Fetch from RSS feeds
            rss_articles = await self._fetch_rss_news(symbol, hours_back)
            news_articles.extend(rss_articles)
            
            # Fetch from APIs (if keys available)
            api_articles = await self._fetch_api_news(symbol, hours_back)
            news_articles.extend(api_articles)
            
            # Remove duplicates
            news_articles = self._deduplicate_articles(news_articles)
            
            return news_articles
            
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def _fetch_rss_news(self, symbol: str, hours_back: int) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds."""
        articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for feed_url in self.news_sources:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url, timeout=10) as response:
                        content = await response.text()
                
                feed = feedparser.parse(content)
                
                for entry in feed.entries:
                    # Check if article mentions the symbol
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    if symbol.upper() in title.upper() or symbol.upper() in summary.upper():
                        # Parse published date
                        published = entry.get('published_parsed')
                        if published:
                            pub_date = datetime(*published[:6])
                            if pub_date < cutoff_time:
                                continue
                        
                        articles.append({
                            "title": title,
                            "summary": summary,
                            "url": entry.get('link', ''),
                            "published": pub_date.isoformat() if published else datetime.utcnow().isoformat(),
                            "source": "RSS",
                            "feed_url": feed_url
                        })
                        
            except Exception as e:
                logging.error(f"Error fetching RSS feed {feed_url}: {e}")
                continue
        
        return articles
    
    async def _fetch_api_news(self, symbol: str, hours_back: int) -> List[Dict[str, Any]]:
        """Fetch news from API sources."""
        # This would implement API calls to news services
        # For now, return empty list as APIs require keys
        return []
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            # Simple deduplication based on title
            title_words = set(title.split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # If 80% of words overlap, consider it duplicate
                overlap = len(title_words.intersection(seen_words))
                if overlap / max(len(title_words), len(seen_words)) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles


class SocialMediaAnalyzer:
    """Social media sentiment analyzer (Twitter, Reddit, etc.)."""
    
    def __init__(self):
        # Social media keywords for financial sentiment
        self.financial_keywords = [
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'moon', 'crash',
            'pump', 'dump', 'rally', 'dip', 'support', 'resistance',
            'breakout', 'breakdown', 'squeeze', 'rocket', 'diamond hands',
            'paper hands', 'HODL', 'FUD', 'FOMO'
        ]
        
        # Sentiment modifiers
        self.positive_words = ['good', 'great', 'excellent', 'amazing', 'strong', 'up', 'gain']
        self.negative_words = ['bad', 'terrible', 'awful', 'weak', 'down', 'loss', 'crash']
    
    async def analyze_social_sentiment(self, symbol: str, posts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment from social media posts."""
        
        if not posts:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sample_size": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
            }
        
        sentiments = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
        # Analyze each post
        for post in posts:
            sentiment_score = await self._analyze_post_sentiment(post, symbol)
            sentiments.append(sentiment_score)
            
            if sentiment_score > 0.1:
                sentiment_counts["positive"] += 1
            elif sentiment_score < -0.1:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
        
        # Calculate aggregate metrics
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0.0
        confidence = max(0.1, min(0.9, 1.0 - sentiment_std))
        
        return {
            "sentiment_score": float(avg_sentiment),
            "confidence": float(confidence),
            "sample_size": len(posts),
            "sentiment_distribution": sentiment_counts,
            "sentiment_std": float(sentiment_std)
        }
    
    async def _analyze_post_sentiment(self, post: str, symbol: str) -> float:
        """Analyze sentiment of individual social media post."""
        
        post_lower = post.lower()
        score = 0.0
        
        # Check for financial keywords
        keyword_multiplier = 1.0
        for keyword in self.financial_keywords:
            if keyword in post_lower:
                if keyword in ['bullish', 'buy', 'moon', 'rocket', 'diamond hands']:
                    score += 0.3
                elif keyword in ['bearish', 'sell', 'crash', 'dump', 'paper hands']:
                    score -= 0.3
                keyword_multiplier = 1.5  # Boost confidence for financial posts
        
        # Check for positive/negative words
        for word in self.positive_words:
            score += post_lower.count(word) * 0.1
        
        for word in self.negative_words:
            score -= post_lower.count(word) * 0.1
        
        # Check for emoji sentiment (basic)
        positive_emojis = ['😀', '😊', '🚀', '📈', '💎', '🌙']
        negative_emojis = ['😢', '😰', '📉', '💔', '😱']
        
        for emoji in positive_emojis:
            score += post.count(emoji) * 0.2
        
        for emoji in negative_emojis:
            score -= post.count(emoji) * 0.2
        
        # Apply keyword multiplier
        score *= keyword_multiplier
        
        # Normalize to -1, 1 range
        return max(-1.0, min(1.0, score))


class MarketSentimentAgent:
    """
    M9 - Market Sentiment Agent
    Comprehensive sentiment analysis combining news, social media, and market indicators.
    """
    
    def __init__(self, cache: TradingCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.finbert = FinBERTAnalyzer()
        self.news_aggregator = NewsAggregator()
        self.social_analyzer = SocialMediaAnalyzer()
        
        # Sentiment history for trend analysis
        self.sentiment_history: Dict[str, List[AggregatedSentiment]] = {}
        
        # Initialize FinBERT asynchronously
        asyncio.create_task(self.finbert.initialize())
    
    async def analyze_comprehensive_sentiment(self, symbol: str) -> Optional[AggregatedSentiment]:
        """Analyze comprehensive sentiment for a symbol."""
        
        try:
            # Collect sentiment from multiple sources
            news_sentiment = await self._analyze_news_sentiment(symbol)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            market_sentiment = await self._analyze_market_sentiment(symbol)
            
            # Aggregate all sentiment sources
            aggregated = await self._aggregate_sentiments(
                symbol, news_sentiment, social_sentiment, market_sentiment
            )
            
            # Store in history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append(aggregated)
            
            # Keep only recent history (last 7 days)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.sentiment_history[symbol] = [
                s for s in self.sentiment_history[symbol] 
                if s.timestamp > cutoff_time
            ]
            
            # Cache the result
            cache_key = f"sentiment_analysis:{symbol}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, aggregated.to_dict(), ttl=1800)  # 30 minutes
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return None
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        
        try:
            # Fetch recent news
            articles = await self.news_aggregator.fetch_news(symbol, hours_back=24)
            
            if not articles:
                return {"sentiment_score": 0.0, "confidence": 0.0, "sample_size": 0}
            
            sentiments = []
            
            for article in articles:
                # Analyze title and summary
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                
                if len(text.strip()) > 10:
                    sentiment, confidence, score = await self.finbert.analyze_sentiment(text)
                    sentiments.append({
                        "score": score,
                        "confidence": confidence,
                        "text": text[:100]  # Store sample for debugging
                    })
            
            if not sentiments:
                return {"sentiment_score": 0.0, "confidence": 0.0, "sample_size": 0}
            
            # Calculate weighted average
            total_weight = sum(s["confidence"] for s in sentiments)
            if total_weight > 0:
                weighted_score = sum(s["score"] * s["confidence"] for s in sentiments) / total_weight
            else:
                weighted_score = np.mean([s["score"] for s in sentiments])
            
            avg_confidence = np.mean([s["confidence"] for s in sentiments])
            
            return {
                "sentiment_score": float(weighted_score),
                "confidence": float(avg_confidence),
                "sample_size": len(sentiments),
                "source": "news"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.0, "sample_size": 0}
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from social media."""
        
        try:
            # Get cached social media posts (would be populated by data collection)
            cache_key = f"social_posts:{symbol}"
            social_posts = await self.cache.get(cache_key)
            
            if not social_posts:
                # Generate sample social sentiment for demo
                return await self._generate_sample_social_sentiment(symbol)
            
            # Analyze social sentiment
            result = await self.social_analyzer.analyze_social_sentiment(symbol, social_posts)
            result["source"] = "social"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.0, "sample_size": 0}
    
    async def _generate_sample_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Generate sample social sentiment for demonstration."""
        
        # Sample posts that might appear for different symbols
        sample_posts = [
            f"{symbol} looking bullish! 🚀",
            f"Holding {symbol} for the long term 💎",
            f"{symbol} might see some resistance here",
            f"Great earnings from {symbol}!",
            f"{symbol} technical analysis shows upward trend",
            f"Bearish on {symbol} in short term",
            f"{symbol} breaking out of consolidation",
            f"Volume spike in {symbol} today",
            f"{symbol} fundamentals look solid",
            f"Taking profits on {symbol}"
        ]
        
        # Randomly sample and analyze
        import random
        selected_posts = random.sample(sample_posts, min(5, len(sample_posts)))
        
        result = await self.social_analyzer.analyze_social_sentiment(symbol, selected_posts)
        result["source"] = "social_sample"
        
        return result
    
    async def _analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from market indicators."""
        
        try:
            # Get market data
            cache_key = f"market_data:{symbol}:1h"
            market_data = await self.cache.get(cache_key)
            
            if not market_data or len(market_data) < 20:
                return {"sentiment_score": 0.0, "confidence": 0.5, "sample_size": 0}
            
            df = pd.DataFrame(market_data)
            
            # Calculate market-based sentiment indicators
            sentiment_score = 0.0
            confidence = 0.5
            
            # Price momentum
            if len(df) >= 10:
                recent_return = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                sentiment_score += np.tanh(recent_return * 20)  # Normalize to -1, 1
            
            # Volume analysis
            if 'volume' in df.columns and len(df) >= 20:
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                if current_volume > avg_volume * 1.5:  # High volume
                    if df['close'].iloc[-1] > df['close'].iloc[-2]:  # Price up
                        sentiment_score += 0.2
                    else:  # Price down
                        sentiment_score -= 0.2
            
            # RSI-based sentiment
            if len(df) >= 14:
                rsi = self._calculate_rsi(df['close'])
                if rsi > 70:  # Overbought
                    sentiment_score -= 0.1
                elif rsi < 30:  # Oversold
                    sentiment_score += 0.1
            
            # Volatility sentiment
            if len(df) >= 20:
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std().iloc[-1]
                recent_volatility = returns.iloc[-5:].std()
                
                if recent_volatility > volatility * 1.5:  # High recent volatility
                    confidence *= 0.8  # Lower confidence
                    sentiment_score *= 0.9  # Slightly dampen sentiment
            
            # Normalize sentiment score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return {
                "sentiment_score": float(sentiment_score),
                "confidence": float(confidence),
                "sample_size": len(df),
                "source": "market"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.5, "sample_size": 0}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50.0
    
    async def _aggregate_sentiments(self, symbol: str, news_sentiment: Dict[str, Any],
                                  social_sentiment: Dict[str, Any], 
                                  market_sentiment: Dict[str, Any]) -> AggregatedSentiment:
        """Aggregate sentiments from all sources."""
        
        # Extract scores and confidences
        sentiments = [news_sentiment, social_sentiment, market_sentiment]
        
        # Calculate weights based on confidence and sample size
        weights = []
        scores = []
        
        for sentiment in sentiments:
            confidence = sentiment.get("confidence", 0.1)
            sample_size = sentiment.get("sample_size", 1)
            score = sentiment.get("sentiment_score", 0.0)
            
            # Weight based on confidence and sample size
            weight = confidence * np.log(1 + sample_size)
            weights.append(weight)
            scores.append(score)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1/3, 1/3, 1/3]  # Equal weights if no confidence
        
        # Calculate weighted sentiment
        overall_score = sum(w * s for w, s in zip(weights, scores))
        overall_confidence = sum(w * sentiments[i].get("confidence", 0.1) for i, w in enumerate(weights))
        
        # Determine overall sentiment
        if overall_score > 0.1:
            overall_sentiment = "positive"
        elif overall_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Calculate Fear & Greed Index (0-100)
        fear_greed_index = await self._calculate_fear_greed_index(symbol, overall_score)
        
        # Analyze sentiment trend
        sentiment_trend = await self._analyze_sentiment_trend(symbol)
        
        # Create source breakdown
        source_breakdown = {
            "news": news_sentiment.get("sentiment_score", 0.0),
            "social": social_sentiment.get("sentiment_score", 0.0),
            "market": market_sentiment.get("sentiment_score", 0.0)
        }
        
        # Calculate volume-weighted score (placeholder)
        volume_weighted_score = overall_score  # Would incorporate volume data
        
        return AggregatedSentiment(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            sentiment_score=overall_score,
            confidence=overall_confidence,
            volume_weighted_score=volume_weighted_score,
            source_breakdown=source_breakdown,
            sentiment_trend=sentiment_trend,
            fear_greed_index=fear_greed_index,
            social_sentiment=social_sentiment.get("sentiment_score", 0.0),
            news_sentiment=news_sentiment.get("sentiment_score", 0.0),
            market_sentiment=market_sentiment.get("sentiment_score", 0.0),
            timestamp=datetime.utcnow(),
            sample_size=sum(s.get("sample_size", 0) for s in sentiments)
        )
    
    async def _calculate_fear_greed_index(self, symbol: str, sentiment_score: float) -> float:
        """Calculate Fear & Greed Index (0-100 scale)."""
        
        # Base index from sentiment score
        base_index = (sentiment_score + 1) * 50  # Convert -1,1 to 0,100
        
        # Adjust based on volatility
        try:
            cache_key = f"market_data:{symbol}:1h"
            market_data = await self.cache.get(cache_key)
            
            if market_data and len(market_data) >= 20:
                df = pd.DataFrame(market_data)
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std().iloc[-1]
                
                # High volatility indicates fear, low volatility indicates greed
                vol_adjustment = -volatility * 1000  # Scale volatility impact
                base_index += vol_adjustment
        
        except Exception:
            pass  # Use base index if volatility calculation fails
        
        # Ensure 0-100 range
        return max(0.0, min(100.0, base_index))
    
    async def _analyze_sentiment_trend(self, symbol: str) -> str:
        """Analyze sentiment trend over time."""
        
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 3:
            return "stable"
        
        # Get recent sentiment scores
        recent_sentiments = self.sentiment_history[symbol][-5:]  # Last 5 readings
        scores = [s.sentiment_score for s in recent_sentiments]
        
        # Calculate trend
        if len(scores) < 2:
            return "stable"
        
        # Linear regression to detect trend
        x = np.arange(len(scores))
        slope, _, r_value, _, _ = stats.linregress(x, scores)
        
        # Determine trend based on slope and correlation
        if abs(r_value) > 0.5:  # Significant correlation
            if slope > 0.05:
                return "improving"
            elif slope < -0.05:
                return "declining"
        
        return "stable"
    
    async def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment summary for a symbol."""
        
        try:
            # Get latest sentiment analysis
            latest_sentiment = await self.analyze_comprehensive_sentiment(symbol)
            
            if not latest_sentiment:
                return {"error": "No sentiment data available"}
            
            # Get historical trend
            history = self.sentiment_history.get(symbol, [])
            
            # Calculate additional metrics
            summary = {
                "current_sentiment": latest_sentiment.to_dict(),
                "historical_trend": {
                    "data_points": len(history),
                    "avg_sentiment_7d": np.mean([s.sentiment_score for s in history]) if history else 0.0,
                    "sentiment_volatility": np.std([s.sentiment_score for s in history]) if len(history) > 1 else 0.0
                },
                "recommendations": await self._generate_sentiment_recommendations(latest_sentiment),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _generate_sentiment_recommendations(self, sentiment: AggregatedSentiment) -> List[str]:
        """Generate trading recommendations based on sentiment."""
        
        recommendations = []
        
        # Sentiment-based recommendations
        if sentiment.sentiment_score > 0.3 and sentiment.confidence > 0.6:
            recommendations.append("Strong positive sentiment supports bullish outlook")
        elif sentiment.sentiment_score < -0.3 and sentiment.confidence > 0.6:
            recommendations.append("Strong negative sentiment suggests caution")
        
        # Fear & Greed recommendations
        if sentiment.fear_greed_index > 80:
            recommendations.append("Extreme greed levels - consider taking profits")
        elif sentiment.fear_greed_index < 20:
            recommendations.append("Extreme fear levels - potential buying opportunity")
        
        # Trend recommendations
        if sentiment.sentiment_trend == "improving":
            recommendations.append("Improving sentiment trend supports continued upside")
        elif sentiment.sentiment_trend == "declining":
            recommendations.append("Declining sentiment trend warrants defensive positioning")
        
        # Source-specific recommendations
        if abs(sentiment.news_sentiment) > abs(sentiment.social_sentiment):
            recommendations.append("News sentiment is driving overall sentiment")
        elif abs(sentiment.social_sentiment) > abs(sentiment.news_sentiment):
            recommendations.append("Social sentiment is leading news sentiment")
        
        return recommendations
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the Market Sentiment Agent."""
        
        return {
            "agent_id": "M9_market_sentiment_agent",
            "finbert_initialized": self.finbert.initialized,
            "symbols_tracked": list(self.sentiment_history.keys()),
            "total_sentiment_analyses": sum(len(history) for history in self.sentiment_history.values()),
            "news_sources": len(self.news_aggregator.news_sources),
            "sentiment_history_depth": {
                symbol: len(history) for symbol, history in self.sentiment_history.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        } 