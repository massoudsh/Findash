"""
Analysis API Routes for FastAPI service
Handles fundamental analysis, technical analysis, and market intelligence
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import random
from datetime import datetime, timedelta

# from src.core.models import TradingSignal, SocialMetrics
# from src.database.postgres_connection import get_db

router = APIRouter()

# Pydantic models
class TechnicalIndicators(BaseModel):
    rsi: float = Field(..., ge=0, le=100)
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    moving_averages: Dict[str, float]
    volume_indicators: Dict[str, float]

class FundamentalMetrics(BaseModel):
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None

class AnalysisResponse(BaseModel):
    symbol: str
    analysis_type: str
    timestamp: str
    technical_indicators: Optional[TechnicalIndicators] = None
    fundamental_metrics: Optional[FundamentalMetrics] = None
    recommendation: str
    confidence_score: float
    target_price: Optional[float] = None
    support_levels: List[float]
    resistance_levels: List[float]

class MarketSentiment(BaseModel):
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_sources: Dict[str, float]
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    confidence: float
    timestamp: str

@router.get("/technical/{symbol}", response_model=AnalysisResponse)
async def get_technical_analysis(
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1m|5m|15m|1h|4h|1d|1w)$")
):
    """Get technical analysis for a symbol"""
    
    # Generate sample technical indicators
    rsi = random.uniform(20, 80)
    
    macd = {
        "macd_line": random.uniform(-2, 2),
        "signal_line": random.uniform(-2, 2),
        "histogram": random.uniform(-1, 1)
    }
    
    bollinger_bands = {
        "upper_band": 155.0 + random.uniform(5, 15),
        "middle_band": 150.0 + random.uniform(-5, 5),
        "lower_band": 145.0 - random.uniform(5, 15),
        "bandwidth": random.uniform(0.1, 0.3)
    }
    
    moving_averages = {
        "sma_20": 150.0 + random.uniform(-10, 10),
        "sma_50": 148.0 + random.uniform(-15, 15),
        "ema_12": 151.0 + random.uniform(-8, 8),
        "ema_26": 149.0 + random.uniform(-12, 12)
    }
    
    volume_indicators = {
        "volume_sma": random.uniform(1000000, 5000000),
        "on_balance_volume": random.uniform(-10000000, 10000000),
        "volume_rate_of_change": random.uniform(-50, 50)
    }
    
    technical_indicators = TechnicalIndicators(
        rsi=rsi,
        macd=macd,
        bollinger_bands=bollinger_bands,
        moving_averages=moving_averages,
        volume_indicators=volume_indicators
    )
    
    # Determine recommendation based on indicators
    if rsi < 30 and macd["macd_line"] > macd["signal_line"]:
        recommendation = "BUY"
        confidence = 0.85
    elif rsi > 70 and macd["macd_line"] < macd["signal_line"]:
        recommendation = "SELL"
        confidence = 0.80
    else:
        recommendation = "HOLD"
        confidence = 0.65
    
    # Generate support and resistance levels
    current_price = 150.0 + random.uniform(-20, 20)
    support_levels = [
        current_price - random.uniform(5, 10),
        current_price - random.uniform(10, 20),
        current_price - random.uniform(20, 30)
    ]
    resistance_levels = [
        current_price + random.uniform(5, 10),
        current_price + random.uniform(10, 20),
        current_price + random.uniform(20, 30)
    ]
    
    target_price = None
    if recommendation == "BUY":
        target_price = current_price + random.uniform(10, 25)
    elif recommendation == "SELL":
        target_price = current_price - random.uniform(10, 25)
    
    return AnalysisResponse(
        symbol=symbol,
        analysis_type="technical",
        timestamp=datetime.utcnow().isoformat(),
        technical_indicators=technical_indicators,
        fundamental_metrics=None,
        recommendation=recommendation,
        confidence_score=confidence,
        target_price=target_price,
        support_levels=sorted(support_levels, reverse=True),
        resistance_levels=sorted(resistance_levels)
    )

@router.get("/fundamental/{symbol}", response_model=AnalysisResponse)
async def get_fundamental_analysis(symbol: str):
    """Get fundamental analysis for a symbol"""
    
    # Generate sample fundamental metrics
    fundamental_metrics = FundamentalMetrics(
        pe_ratio=random.uniform(10, 40),
        pb_ratio=random.uniform(0.5, 5.0),
        debt_to_equity=random.uniform(0.1, 2.0),
        roe=random.uniform(0.05, 0.35),
        revenue_growth=random.uniform(-0.1, 0.5),
        earnings_growth=random.uniform(-0.2, 0.8),
        dividend_yield=random.uniform(0, 0.05)
    )
    
    # Determine recommendation based on fundamentals
    score = 0
    if fundamental_metrics.pe_ratio < 20:
        score += 1
    if fundamental_metrics.pb_ratio < 2.0:
        score += 1
    if fundamental_metrics.roe > 0.15:
        score += 1
    if fundamental_metrics.revenue_growth > 0.1:
        score += 1
    if fundamental_metrics.earnings_growth > 0.1:
        score += 1
    
    if score >= 4:
        recommendation = "BUY"
        confidence = 0.90
    elif score >= 2:
        recommendation = "HOLD"
        confidence = 0.70
    else:
        recommendation = "SELL"
        confidence = 0.75
    
    # Calculate target price based on fundamentals
    current_price = 150.0 + random.uniform(-30, 30)
    if recommendation == "BUY":
        target_price = current_price * (1 + random.uniform(0.15, 0.4))
    elif recommendation == "SELL":
        target_price = current_price * (1 - random.uniform(0.1, 0.3))
    else:
        target_price = current_price * (1 + random.uniform(-0.05, 0.05))
    
    return AnalysisResponse(
        symbol=symbol,
        analysis_type="fundamental",
        timestamp=datetime.utcnow().isoformat(),
        technical_indicators=None,
        fundamental_metrics=fundamental_metrics,
        recommendation=recommendation,
        confidence_score=confidence,
        target_price=round(target_price, 2),
        support_levels=[],
        resistance_levels=[]
    )

@router.get("/comprehensive/{symbol}", response_model=AnalysisResponse)
async def get_comprehensive_analysis(symbol: str):
    """Get comprehensive analysis combining technical and fundamental factors"""
    
    # Get both technical and fundamental analysis
    technical_response = await get_technical_analysis(symbol)
    fundamental_response = await get_fundamental_analysis(symbol)
    
    # Combine recommendations
    tech_weight = 0.6
    fund_weight = 0.4
    
    # Convert recommendations to scores
    rec_scores = {"BUY": 1, "HOLD": 0, "SELL": -1}
    tech_score = rec_scores[technical_response.recommendation]
    fund_score = rec_scores[fundamental_response.recommendation]
    
    combined_score = (tech_score * tech_weight) + (fund_score * fund_weight)
    combined_confidence = (technical_response.confidence_score * tech_weight) + \
                         (fundamental_response.confidence_score * fund_weight)
    
    if combined_score > 0.3:
        final_recommendation = "BUY"
    elif combined_score < -0.3:
        final_recommendation = "SELL"
    else:
        final_recommendation = "HOLD"
    
    # Average target prices
    target_price = None
    if technical_response.target_price and fundamental_response.target_price:
        target_price = (technical_response.target_price + fundamental_response.target_price) / 2
    elif technical_response.target_price:
        target_price = technical_response.target_price
    elif fundamental_response.target_price:
        target_price = fundamental_response.target_price
    
    return AnalysisResponse(
        symbol=symbol,
        analysis_type="comprehensive",
        timestamp=datetime.utcnow().isoformat(),
        technical_indicators=technical_response.technical_indicators,
        fundamental_metrics=fundamental_response.fundamental_metrics,
        recommendation=final_recommendation,
        confidence_score=round(combined_confidence, 2),
        target_price=round(target_price, 2) if target_price else None,
        support_levels=technical_response.support_levels,
        resistance_levels=technical_response.resistance_levels
    )

@router.get("/sentiment/{symbol}", response_model=MarketSentiment)
async def get_market_sentiment(symbol: str):
    """Get market sentiment analysis for a symbol"""
    
    # Generate sample sentiment scores
    news_sentiment = random.uniform(-0.5, 0.8)
    social_sentiment = random.uniform(-0.7, 0.9)
    analyst_sentiment = random.uniform(-0.3, 0.6)
    
    # Calculate overall sentiment (weighted average)
    overall_sentiment = (news_sentiment * 0.4) + (social_sentiment * 0.3) + (analyst_sentiment * 0.3)
    
    sentiment_sources = {
        "news_articles": news_sentiment,
        "social_media": social_sentiment,
        "analyst_reports": analyst_sentiment,
        "earnings_calls": random.uniform(-0.2, 0.5),
        "sec_filings": random.uniform(-0.1, 0.3)
    }
    
    # Calculate confidence based on consistency
    sentiments = list(sentiment_sources.values())
    sentiment_std = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
    confidence = max(0.5, 1.0 - sentiment_std)
    
    return MarketSentiment(
        symbol=symbol,
        overall_sentiment=round(overall_sentiment, 3),
        sentiment_sources={k: round(v, 3) for k, v in sentiment_sources.items()},
        news_sentiment=round(news_sentiment, 3),
        social_sentiment=round(social_sentiment, 3),
        analyst_sentiment=round(analyst_sentiment, 3),
        confidence=round(confidence, 3),
        timestamp=datetime.utcnow().isoformat()
    )

@router.get("/screener", response_model=List[Dict[str, Any]])
async def stock_screener(
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_volume: Optional[int] = Query(None, ge=0),
    min_market_cap: Optional[float] = Query(None, ge=0),
    sector: Optional[str] = Query(None),
    recommendation: Optional[str] = Query(None, pattern="^(BUY|SELL|HOLD)$"),
    limit: int = Query(20, le=100)
):
    """Screen stocks based on specified criteria"""
    
    # Sample stock universe
    stocks = [
        {"symbol": "AAPL", "price": 175.50, "volume": 50000000, "market_cap": 2800000000000, "sector": "Technology"},
        {"symbol": "TSLA", "price": 240.80, "volume": 45000000, "market_cap": 760000000000, "sector": "Consumer Discretionary"},
        {"symbol": "MSFT", "price": 378.90, "volume": 25000000, "market_cap": 2800000000000, "sector": "Technology"},
        {"symbol": "GOOGL", "price": 140.20, "volume": 28000000, "market_cap": 1750000000000, "sector": "Communication Services"},
        {"symbol": "AMZN", "price": 155.30, "volume": 40000000, "market_cap": 1600000000000, "sector": "Consumer Discretionary"},
        {"symbol": "NVDA", "price": 485.60, "volume": 35000000, "market_cap": 1200000000000, "sector": "Technology"},
        {"symbol": "META", "price": 325.40, "volume": 20000000, "market_cap": 850000000000, "sector": "Communication Services"},
        {"symbol": "JPM", "price": 155.80, "volume": 15000000, "market_cap": 450000000000, "sector": "Financial"},
        {"symbol": "JNJ", "price": 162.30, "volume": 8000000, "market_cap": 420000000000, "sector": "Healthcare"},
        {"symbol": "PG", "price": 156.90, "volume": 6000000, "market_cap": 370000000000, "sector": "Consumer Staples"}
    ]
    
    # Apply filters
    filtered_stocks = []
    
    for stock in stocks:
        # Apply filters
        if min_price and stock["price"] < min_price:
            continue
        if max_price and stock["price"] > max_price:
            continue
        if min_volume and stock["volume"] < min_volume:
            continue
        if min_market_cap and stock["market_cap"] < min_market_cap:
            continue
        if sector and stock["sector"] != sector:
            continue
        
        # Generate analysis data for the stock
        tech_rec = random.choice(["BUY", "SELL", "HOLD"])
        fund_rec = random.choice(["BUY", "SELL", "HOLD"])
        
        # Simple scoring for overall recommendation
        rec_scores = {"BUY": 1, "HOLD": 0, "SELL": -1}
        avg_score = (rec_scores[tech_rec] + rec_scores[fund_rec]) / 2
        
        if avg_score > 0.3:
            overall_rec = "BUY"
        elif avg_score < -0.3:
            overall_rec = "SELL"
        else:
            overall_rec = "HOLD"
        
        if recommendation and overall_rec != recommendation:
            continue
        
        # Add to results
        filtered_stocks.append({
            **stock,
            "technical_recommendation": tech_rec,
            "fundamental_recommendation": fund_rec,
            "overall_recommendation": overall_rec,
            "rsi": random.uniform(20, 80),
            "pe_ratio": random.uniform(10, 40),
            "revenue_growth": random.uniform(-0.1, 0.5),
            "price_change_1d": random.uniform(-5, 5),
            "price_change_1w": random.uniform(-10, 10),
            "target_price": stock["price"] * (1 + random.uniform(-0.2, 0.3))
        })
    
    # Sort by market cap (largest first) and limit results
    filtered_stocks.sort(key=lambda x: x["market_cap"], reverse=True)
    return filtered_stocks[:limit]

@router.get("/comparison", response_model=Dict[str, Any])
async def compare_stocks(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Compare multiple stocks across key metrics"""
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    if len(symbol_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed for comparison")
    
    comparison_data = {}
    
    for symbol in symbol_list:
        # Generate sample data for each symbol
        comparison_data[symbol] = {
            "price": 150.0 + random.uniform(-50, 100),
            "market_cap": random.uniform(100000000000, 3000000000000),
            "pe_ratio": random.uniform(10, 40),
            "pb_ratio": random.uniform(0.5, 5.0),
            "debt_to_equity": random.uniform(0.1, 2.0),
            "roe": random.uniform(0.05, 0.35),
            "revenue_growth": random.uniform(-0.1, 0.5),
            "earnings_growth": random.uniform(-0.2, 0.8),
            "dividend_yield": random.uniform(0, 0.05),
            "beta": random.uniform(0.5, 2.0),
            "rsi": random.uniform(20, 80),
            "recommendation": random.choice(["BUY", "SELL", "HOLD"]),
            "target_price": 150.0 + random.uniform(-30, 50),
            "analyst_count": random.randint(5, 25),
            "price_change_ytd": random.uniform(-0.3, 0.8)
        }
    
    # Calculate relative rankings
    metrics = ["market_cap", "pe_ratio", "roe", "revenue_growth", "earnings_growth"]
    rankings = {}
    
    for metric in metrics:
        sorted_symbols = sorted(symbol_list, 
                              key=lambda s: comparison_data[s][metric], 
                              reverse=(metric in ["market_cap", "roe", "revenue_growth", "earnings_growth"]))
        rankings[metric] = {symbol: idx + 1 for idx, symbol in enumerate(sorted_symbols)}
    
    return {
        "symbols": symbol_list,
        "comparison_data": comparison_data,
        "rankings": rankings,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health", response_model=Dict[str, Any])
async def analysis_health():
    """Health check for analysis service"""
    return {
        "status": "healthy",
        "service": "analysis",
        "timestamp": datetime.utcnow().isoformat(),
        "available_analyses": ["technical", "fundamental", "comprehensive", "sentiment"],
        "screener_stocks": 500,
        "last_update": datetime.utcnow().isoformat()
    } 