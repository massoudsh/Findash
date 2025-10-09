"""
Alternative Data Engine for Quantum Trading Matrix™
Comprehensive alternative data analysis including social sentiment, news, economics, and satellite data
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from textblob import TextBlob
import json
import requests
from collections import defaultdict
import yfinance as yf

logger = logging.getLogger(__name__)

class DataSource(Enum):
    SOCIAL = "social"
    NEWS = "news"
    ECONOMIC = "economic"
    SATELLITE = "satellite"
    ALTERNATIVE = "alternative"

@dataclass
class SocialSentimentData:
    symbol: str
    platform: str
    mentions: int
    positive_sentiment: float
    negative_sentiment: float
    neutral_sentiment: float
    overall_score: float
    confidence: float
    timestamp: datetime
    trending: bool = False
    
@dataclass
class NewsAnalyticsData:
    symbol: str
    headline: str
    sentiment_score: float
    impact_score: float
    source: str
    timestamp: datetime
    category: str
    confidence: float
    event_type: Optional[str] = None

@dataclass
class EconomicIndicator:
    indicator_name: str
    value: float
    previous_value: Optional[float]
    change: float
    impact_level: str  # "high", "medium", "low"
    market_correlation: float
    timestamp: datetime
    
@dataclass
class SatelliteData:
    data_type: str  # "parking", "oil_storage", "shipping"
    location: str
    metric_value: float
    change_from_baseline: float
    economic_impact: float
    timestamp: datetime
    confidence: float

@dataclass
class AlternativeDataScore:
    symbol: str
    overall_score: float
    social_score: float
    news_score: float
    economic_score: float
    satellite_score: float
    confidence: float
    recommendation: str
    risk_factors: List[str]
    timestamp: datetime
    trend: str  # "bullish", "bearish", "neutral"

class SocialSentimentAnalyzer:
    """Analyzes social media sentiment for trading symbols"""
    
    def __init__(self):
        self.platform_weights = {
            "twitter": 0.4,
            "reddit": 0.3,
            "discord": 0.2,
            "telegram": 0.1
        }
        
    async def analyze_social_sentiment(self, symbol: str) -> SocialSentimentData:
        """Analyze social sentiment for a given symbol"""
        try:
            # Simulated social data (in production, integrate with Twitter API, Reddit API, etc.)
            mentions_data = await self._fetch_social_mentions(symbol)
            sentiment_scores = await self._analyze_sentiment_scores(mentions_data)
            
            # Calculate weighted sentiment
            overall_score = sum(
                sentiment_scores[platform] * weight 
                for platform, weight in self.platform_weights.items()
                if platform in sentiment_scores
            )
            
            # Determine trending status
            trending = mentions_data.get("total_mentions", 0) > 1000
            
            return SocialSentimentData(
                symbol=symbol,
                platform="aggregated",
                mentions=mentions_data.get("total_mentions", 0),
                positive_sentiment=sentiment_scores.get("positive", 0.3),
                negative_sentiment=sentiment_scores.get("negative", 0.2),
                neutral_sentiment=sentiment_scores.get("neutral", 0.5),
                overall_score=overall_score,
                confidence=0.7 + (mentions_data.get("total_mentions", 0) / 10000) * 0.3,
                timestamp=datetime.now(),
                trending=trending
            )
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment for {symbol}: {e}")
            return self._get_default_social_sentiment(symbol)
    
    async def _fetch_social_mentions(self, symbol: str) -> Dict[str, Any]:
        """Fetch social media mentions (simulated)"""
        # Simulate API calls to social platforms
        base_mentions = hash(symbol) % 5000
        return {
            "total_mentions": base_mentions,
            "twitter_mentions": int(base_mentions * 0.4),
            "reddit_mentions": int(base_mentions * 0.3),
            "discord_mentions": int(base_mentions * 0.2),
            "telegram_mentions": int(base_mentions * 0.1)
        }
    
    async def _analyze_sentiment_scores(self, mentions_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze sentiment scores across platforms"""
        # Simulated sentiment analysis
        return {
            "positive": 0.35 + (mentions_data.get("total_mentions", 0) / 20000),
            "negative": 0.25 - (mentions_data.get("total_mentions", 0) / 30000),
            "neutral": 0.4,
            "twitter": 0.6,
            "reddit": 0.55,
            "discord": 0.65,
            "telegram": 0.5
        }
    
    def _get_default_social_sentiment(self, symbol: str) -> SocialSentimentData:
        """Return default sentiment data if analysis fails"""
        return SocialSentimentData(
            symbol=symbol,
            platform="aggregated",
            mentions=0,
            positive_sentiment=0.33,
            negative_sentiment=0.33,
            neutral_sentiment=0.34,
            overall_score=0.5,
            confidence=0.3,
            timestamp=datetime.now()
        )

class NewsAnalyzer:
    """Analyzes news sentiment and impact for trading symbols"""
    
    def __init__(self):
        self.impact_weights = {
            "earnings": 0.8,
            "merger": 0.9,
            "regulatory": 0.7,
            "product": 0.6,
            "management": 0.5,
            "general": 0.3
        }
        
    async def analyze_news_sentiment(self, symbol: str) -> List[NewsAnalyticsData]:
        """Analyze news sentiment for a given symbol"""
        try:
            news_articles = await self._fetch_news_articles(symbol)
            analyzed_news = []
            
            for article in news_articles:
                sentiment_score = self._analyze_article_sentiment(article["content"])
                impact_score = self._calculate_impact_score(article, symbol)
                
                news_data = NewsAnalyticsData(
                    symbol=symbol,
                    headline=article["headline"],
                    sentiment_score=sentiment_score,
                    impact_score=impact_score,
                    source=article.get("source", "unknown"),
                    timestamp=datetime.fromisoformat(article["timestamp"]),
                    category=article.get("category", "general"),
                    confidence=0.8,
                    event_type=self._classify_event_type(article["headline"])
                )
                analyzed_news.append(news_data)
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {e}")
            return []
    
    async def _fetch_news_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news articles for symbol (simulated)"""
        # Simulate news API response
        sample_articles = [
            {
                "headline": f"{symbol} Reports Strong Q3 Earnings",
                "content": f"Company {symbol} exceeded expectations with revenue growth",
                "source": "Financial Times",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "category": "earnings"
            },
            {
                "headline": f"{symbol} Announces New Product Launch",
                "content": f"{symbol} unveils innovative technology solution",
                "source": "TechCrunch",
                "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
                "category": "product"
            },
            {
                "headline": f"Analysts Upgrade {symbol} Rating",
                "content": f"Wall Street analysts raise price target for {symbol}",
                "source": "Bloomberg",
                "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
                "category": "analyst"
            }
        ]
        return sample_articles
    
    def _analyze_article_sentiment(self, content: str) -> float:
        """Analyze sentiment of article content"""
        try:
            blob = TextBlob(content)
            # Convert polarity from [-1, 1] to [0, 1]
            return (blob.sentiment.polarity + 1) / 2
        except Exception:
            return 0.5  # Neutral sentiment
    
    def _calculate_impact_score(self, article: Dict[str, Any], symbol: str) -> float:
        """Calculate potential market impact of news"""
        base_impact = self.impact_weights.get(article.get("category", "general"), 0.3)
        
        # Adjust based on source credibility
        source_multiplier = 1.0
        credible_sources = ["Bloomberg", "Reuters", "Financial Times", "Wall Street Journal"]
        if article.get("source") in credible_sources:
            source_multiplier = 1.2
        
        # Adjust based on headline keywords
        headline = article.get("headline", "").lower()
        keyword_multiplier = 1.0
        if any(word in headline for word in ["earnings", "merger", "acquisition"]):
            keyword_multiplier = 1.3
        elif any(word in headline for word in ["lawsuit", "investigation", "fine"]):
            keyword_multiplier = 1.2
        
        return min(base_impact * source_multiplier * keyword_multiplier, 1.0)
    
    def _classify_event_type(self, headline: str) -> Optional[str]:
        """Classify the type of news event"""
        headline_lower = headline.lower()
        
        if any(word in headline_lower for word in ["earnings", "revenue", "profit"]):
            return "earnings"
        elif any(word in headline_lower for word in ["merger", "acquisition", "buyout"]):
            return "m&a"
        elif any(word in headline_lower for word in ["fda", "approval", "regulatory"]):
            return "regulatory"
        elif any(word in headline_lower for word in ["product", "launch", "innovation"]):
            return "product"
        elif any(word in headline_lower for word in ["ceo", "management", "executive"]):
            return "management"
        else:
            return "general"

class EconomicDataProvider:
    """Provides economic indicators and their market impact"""
    
    def __init__(self):
        self.indicators = {
            "GDP": {"impact": "high", "correlation": 0.7},
            "CPI": {"impact": "high", "correlation": -0.6},
            "unemployment": {"impact": "medium", "correlation": -0.5},
            "interest_rates": {"impact": "high", "correlation": 0.8},
            "retail_sales": {"impact": "medium", "correlation": 0.4},
            "manufacturing_pmi": {"impact": "medium", "correlation": 0.6}
        }
    
    async def get_economic_indicators(self, symbol: str) -> List[EconomicIndicator]:
        """Get relevant economic indicators for symbol analysis"""
        try:
            indicators = []
            
            for indicator_name, properties in self.indicators.items():
                indicator_data = await self._fetch_indicator_data(indicator_name)
                
                indicator = EconomicIndicator(
                    indicator_name=indicator_name,
                    value=indicator_data["current"],
                    previous_value=indicator_data.get("previous"),
                    change=indicator_data["change"],
                    impact_level=properties["impact"],
                    market_correlation=properties["correlation"],
                    timestamp=datetime.now()
                )
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return []
    
    async def _fetch_indicator_data(self, indicator: str) -> Dict[str, float]:
        """Fetch economic indicator data (simulated)"""
        # Simulate economic data API
        base_values = {
            "GDP": 2.1,
            "CPI": 3.2,
            "unemployment": 3.8,
            "interest_rates": 5.25,
            "retail_sales": 4.1,
            "manufacturing_pmi": 48.7
        }
        
        current = base_values.get(indicator, 50.0)
        previous = current + np.random.normal(0, 0.5)
        change = current - previous
        
        return {
            "current": current,
            "previous": previous,
            "change": change
        }

class SatelliteAnalyzer:
    """Analyzes satellite data for economic insights"""
    
    def __init__(self):
        self.data_types = ["parking", "oil_storage", "shipping", "agriculture", "construction"]
    
    async def analyze_satellite_data(self, symbol: str) -> List[SatelliteData]:
        """Analyze satellite data relevant to the symbol"""
        try:
            satellite_data = []
            
            # Get sector-specific satellite data
            sector = await self._get_symbol_sector(symbol)
            relevant_data_types = self._get_relevant_data_types(sector)
            
            for data_type in relevant_data_types:
                data = await self._fetch_satellite_data(data_type, symbol)
                satellite_data.append(data)
            
            return satellite_data
            
        except Exception as e:
            logger.error(f"Error analyzing satellite data for {symbol}: {e}")
            return []
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get the sector for a given symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("sector", "Technology").lower()
        except:
            return "technology"
    
    def _get_relevant_data_types(self, sector: str) -> List[str]:
        """Get relevant satellite data types for a sector"""
        sector_mapping = {
            "energy": ["oil_storage", "shipping"],
            "retail": ["parking", "shipping"],
            "real estate": ["construction", "parking"],
            "agriculture": ["agriculture"],
            "technology": ["parking"],
            "transportation": ["shipping", "parking"],
            "consumer discretionary": ["parking", "shipping"]
        }
        
        return sector_mapping.get(sector, ["parking"])
    
    async def _fetch_satellite_data(self, data_type: str, symbol: str) -> SatelliteData:
        """Fetch specific satellite data (simulated)"""
        # Simulate satellite data
        base_value = hash(f"{symbol}{data_type}") % 1000 + 500
        baseline = base_value * 0.9
        change = (base_value - baseline) / baseline
        
        return SatelliteData(
            data_type=data_type,
            location=f"Global_{symbol}",
            metric_value=base_value,
            change_from_baseline=change,
            economic_impact=abs(change) * 0.5,
            timestamp=datetime.now(),
            confidence=0.75
        )

class AlternativeDataEngine:
    """Main engine for alternative data analysis and integration"""
    
    def __init__(self, cache=None):
        self.cache = cache
        self.social_analyzer = SocialSentimentAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.economic_provider = EconomicDataProvider()
        self.satellite_analyzer = SatelliteAnalyzer()
        
        # Scoring weights
        self.scoring_weights = {
            DataSource.SOCIAL: 0.25,
            DataSource.NEWS: 0.35,
            DataSource.ECONOMIC: 0.25,
            DataSource.SATELLITE: 0.15
        }
        
    async def analyze_symbol(self, symbol: str, sources: List[str] = None) -> AlternativeDataScore:
        """Comprehensive alternative data analysis for a symbol"""
        if sources is None:
            sources = ["social", "news", "economic", "satellite"]
        
        try:
            # Parallel data collection
            tasks = []
            
            if "social" in sources:
                tasks.append(self.social_analyzer.analyze_social_sentiment(symbol))
            if "news" in sources:
                tasks.append(self.news_analyzer.analyze_news_sentiment(symbol))
            if "economic" in sources:
                tasks.append(self.economic_provider.get_economic_indicators(symbol))
            if "satellite" in sources:
                tasks.append(self.satellite_analyzer.analyze_satellite_data(symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            social_data = results[0] if "social" in sources and len(results) > 0 else None
            news_data = results[1] if "news" in sources and len(results) > 1 else []
            economic_data = results[2] if "economic" in sources and len(results) > 2 else []
            satellite_data = results[3] if "satellite" in sources and len(results) > 3 else []
            
            # Calculate individual scores
            social_score = self._calculate_social_score(social_data) if social_data else 0.5
            news_score = self._calculate_news_score(news_data) if news_data else 0.5
            economic_score = self._calculate_economic_score(economic_data) if economic_data else 0.5
            satellite_score = self._calculate_satellite_score(satellite_data) if satellite_data else 0.5
            
            # Calculate overall score
            overall_score = (
                social_score * self.scoring_weights[DataSource.SOCIAL] +
                news_score * self.scoring_weights[DataSource.NEWS] +
                economic_score * self.scoring_weights[DataSource.ECONOMIC] +
                satellite_score * self.scoring_weights[DataSource.SATELLITE]
            )
            
            # Determine trend and recommendation
            trend = self._determine_trend(overall_score)
            recommendation = self._generate_recommendation(overall_score, trend)
            risk_factors = self._identify_risk_factors(social_data, news_data, economic_data, satellite_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(social_data, news_data, economic_data, satellite_data)
            
            return AlternativeDataScore(
                symbol=symbol,
                overall_score=overall_score,
                social_score=social_score,
                news_score=news_score,
                economic_score=economic_score,
                satellite_score=satellite_score,
                confidence=confidence,
                recommendation=recommendation,
                risk_factors=risk_factors,
                timestamp=datetime.now(),
                trend=trend
            )
            
        except Exception as e:
            logger.error(f"Error in alternative data analysis for {symbol}: {e}")
            return self._get_default_score(symbol)
    
    def _calculate_social_score(self, social_data: SocialSentimentData) -> float:
        """Calculate normalized social sentiment score"""
        if not social_data:
            return 0.5
        
        # Weight by confidence and mentions volume
        volume_factor = min(social_data.mentions / 1000, 1.0)  # Cap at 1000 mentions
        confidence_factor = social_data.confidence
        
        return social_data.overall_score * volume_factor * confidence_factor
    
    def _calculate_news_score(self, news_data: List[NewsAnalyticsData]) -> float:
        """Calculate normalized news sentiment score"""
        if not news_data:
            return 0.5
        
        # Weight by impact and recency
        total_weighted_score = 0
        total_weight = 0
        
        for news in news_data:
            # Recency weight (more recent = higher weight)
            hours_old = (datetime.now() - news.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - hours_old / 168)  # Decay over a week
            
            weight = news.impact_score * news.confidence * recency_weight
            total_weighted_score += news.sentiment_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_economic_score(self, economic_data: List[EconomicIndicator]) -> float:
        """Calculate normalized economic indicators score"""
        if not economic_data:
            return 0.5
        
        positive_indicators = 0
        total_weight = 0
        
        for indicator in economic_data:
            weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(indicator.impact_level, 0.3)
            
            # Positive change weighted by market correlation
            if indicator.change > 0 and indicator.market_correlation > 0:
                positive_indicators += weight
            elif indicator.change < 0 and indicator.market_correlation < 0:
                positive_indicators += weight
            
            total_weight += weight
        
        return positive_indicators / total_weight if total_weight > 0 else 0.5
    
    def _calculate_satellite_score(self, satellite_data: List[SatelliteData]) -> float:
        """Calculate normalized satellite data score"""
        if not satellite_data:
            return 0.5
        
        total_score = 0
        total_weight = 0
        
        for data in satellite_data:
            # Positive change indicates growth/activity
            score = 0.5 + (data.change_from_baseline * 0.5)
            score = max(0, min(1, score))  # Clamp between 0 and 1
            
            weight = data.confidence
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _determine_trend(self, overall_score: float) -> str:
        """Determine market trend based on overall score"""
        if overall_score >= 0.65:
            return "bullish"
        elif overall_score <= 0.35:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_recommendation(self, overall_score: float, trend: str) -> str:
        """Generate trading recommendation"""
        if trend == "bullish" and overall_score >= 0.7:
            return "strong_buy"
        elif trend == "bullish":
            return "buy"
        elif trend == "bearish" and overall_score <= 0.3:
            return "strong_sell"
        elif trend == "bearish":
            return "sell"
        else:
            return "hold"
    
    def _identify_risk_factors(self, social_data, news_data, economic_data, satellite_data) -> List[str]:
        """Identify potential risk factors from alternative data"""
        risk_factors = []
        
        if social_data and social_data.negative_sentiment > 0.4:
            risk_factors.append("High negative social sentiment")
        
        if news_data:
            negative_news = [n for n in news_data if n.sentiment_score < 0.3 and n.impact_score > 0.5]
            if negative_news:
                risk_factors.append("Negative high-impact news coverage")
        
        if economic_data:
            negative_indicators = [e for e in economic_data if e.change < 0 and e.impact_level == "high"]
            if negative_indicators:
                risk_factors.append("Deteriorating economic indicators")
        
        if satellite_data:
            declining_activity = [s for s in satellite_data if s.change_from_baseline < -0.1]
            if declining_activity:
                risk_factors.append("Declining economic activity indicators")
        
        return risk_factors
    
    def _calculate_confidence(self, social_data, news_data, economic_data, satellite_data) -> float:
        """Calculate overall confidence in the analysis"""
        confidences = []
        
        if social_data:
            confidences.append(social_data.confidence)
        
        if news_data:
            avg_news_confidence = sum(n.confidence for n in news_data) / len(news_data)
            confidences.append(avg_news_confidence)
        
        if economic_data:
            confidences.append(0.8)  # Economic data generally reliable
        
        if satellite_data:
            avg_satellite_confidence = sum(s.confidence for s in satellite_data) / len(satellite_data)
            confidences.append(avg_satellite_confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _get_default_score(self, symbol: str) -> AlternativeDataScore:
        """Return default score if analysis fails"""
        return AlternativeDataScore(
            symbol=symbol,
            overall_score=0.5,
            social_score=0.5,
            news_score=0.5,
            economic_score=0.5,
            satellite_score=0.5,
            confidence=0.3,
            recommendation="hold",
            risk_factors=["Insufficient data"],
            timestamp=datetime.now(),
            trend="neutral"
        )
    
    async def get_trending_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending symbols based on alternative data metrics"""
        try:
            # In production, this would query real alternative data sources
            trending_symbols = [
                {"symbol": "TSLA", "trend_score": 0.85, "social_mentions": 15000},
                {"symbol": "BTC-USD", "trend_score": 0.88, "social_mentions": 25000},
                {"symbol": "ETH-USD", "trend_score": 0.82, "social_mentions": 18000},
                {"symbol": "NVDA", "trend_score": 0.82, "social_mentions": 8000},
                {"symbol": "AAPL", "trend_score": 0.78, "social_mentions": 12000},
                {"symbol": "LINK-USD", "trend_score": 0.75, "social_mentions": 7500},
                {"symbol": "GOOGL", "trend_score": 0.72, "social_mentions": 5500},
                {"symbol": "GLD", "trend_score": 0.68, "social_mentions": 4200},
                {"symbol": "MSFT", "trend_score": 0.65, "social_mentions": 6000},
                {"symbol": "TRX-USD", "trend_score": 0.62, "social_mentions": 3800}
            ]
            
            # Sort by trend score and return top N
            trending_symbols.sort(key=lambda x: x["trend_score"], reverse=True)
            return trending_symbols[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending symbols: {e}")
            return []

# Cache integration if available
async def setup_alternative_data_cache(cache):
    """Setup caching for alternative data"""
    if cache:
        # Cache configuration for alternative data
        cache_config = {
            "social_sentiment": {"ttl": 300},  # 5 minutes
            "news_analysis": {"ttl": 900},     # 15 minutes
            "economic_data": {"ttl": 3600},    # 1 hour
            "satellite_data": {"ttl": 7200}    # 2 hours
        }
        
        for data_type, config in cache_config.items():
            await cache.set_config(f"alt_data_{data_type}", config) 