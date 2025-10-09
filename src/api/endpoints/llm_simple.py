"""
Simplified LLM endpoints for development/demo purposes
Provides mock data without requiring heavy ML dependencies
"""

from typing import List, Dict, Any
from fastapi import APIRouter
from datetime import datetime

llm_router = APIRouter()

@llm_router.get("/reports/analysis-status")
async def get_analysis_status():
    """
    Get the status of various analysis components
    """
    return {
        "sections": [
            {"id": "market", "name": "Market Analysis", "status": "completed", "progress": 100, "insights": 8, "lastUpdated": "2 min ago"},
            {"id": "portfolio", "name": "Portfolio Performance", "status": "completed", "progress": 100, "insights": 12, "lastUpdated": "5 min ago"},
            {"id": "risk", "name": "Risk Assessment", "status": "processing", "progress": 75, "insights": 6, "lastUpdated": "1 min ago"},
            {"id": "sentiment", "name": "Sentiment Analysis", "status": "completed", "progress": 100, "insights": 15, "lastUpdated": "3 min ago"},
            {"id": "technical", "name": "Technical Signals", "status": "processing", "progress": 60, "insights": 9, "lastUpdated": "Now"},
            {"id": "macro", "name": "Macro Environment", "status": "pending", "progress": 0, "insights": 0, "lastUpdated": "Pending"}
        ]
    }

@llm_router.get("/reports/data-sources")
async def get_data_sources():
    """
    Get the status of data sources feeding into AI analysis
    """
    return {
        "sources": [
            {"id": "market", "name": "Market Data Feed", "type": "market_data", "status": "active", "lastSync": "30s ago", "recordsProcessed": 15420},
            {"id": "news", "name": "News Analytics", "type": "news", "status": "active", "lastSync": "1m ago", "recordsProcessed": 2847},
            {"id": "social", "name": "Social Sentiment", "type": "social", "status": "active", "lastSync": "45s ago", "recordsProcessed": 8932},
            {"id": "technical", "name": "Technical Indicators", "type": "technical", "status": "active", "lastSync": "15s ago", "recordsProcessed": 5621},
            {"id": "fundamental", "name": "Fundamental Data", "type": "fundamental", "status": "active", "lastSync": "2m ago", "recordsProcessed": 1205},
            {"id": "sentiment", "name": "Sentiment Engine", "type": "sentiment", "status": "active", "lastSync": "20s ago", "recordsProcessed": 3847}
        ]
    }

@llm_router.post("/reports/generate-insights")
async def generate_ai_insights():
    """
    Generate AI-powered insights (mock version)
    """
    insights = [
        {
            "id": "tech_momentum",
            "type": "bullish",
            "title": "Strong Momentum in Tech Sector",
            "summary": "AI analysis indicates sustained upward pressure in NVDA (+12.34), MSFT, and GOOGL based on earnings momentum, institutional flow, and technical patterns. Expected 12-15% upside over next 30 days.",
            "confidence": 87,
            "impact": "high",
            "sources": ["Technical Analysis", "Market Data", "News Sentiment", "Options Flow"],
            "timestamp": datetime.now().isoformat(),
            "ai_analysis": "Technical momentum indicators show strong bullish signals across major tech stocks with institutional accumulation patterns."
        },
        {
            "id": "crypto_volatility",
            "type": "warning",
            "title": "Crypto Volatility Alert",
            "summary": "Unusual options activity and social sentiment divergence detected in BTC-USD and ETH-USD. Potential 20-25% volatility spike expected within 48-72 hours.",
            "confidence": 78,
            "impact": "medium",
            "sources": ["Social Sentiment", "Options Data", "Technical Indicators"],
            "timestamp": datetime.now().isoformat(),
            "ai_analysis": "AI detects mixed signals in crypto markets with institutional accumulation conflicting with retail sentiment."
        },
        {
            "id": "commodities_rotation",
            "type": "opportunity",
            "title": "Commodities Rotation Signal",
            "summary": "Inflation hedge rotation detected. GLD and SLV showing strong accumulation patterns while institutional money rotates from growth to value. 8-12% upside potential.",
            "confidence": 82,
            "impact": "medium",
            "sources": ["Institutional Flow", "Macro Analysis", "Technical Patterns"],
            "timestamp": datetime.now().isoformat(),
            "ai_analysis": "Macro environment favors precious metals as inflation hedge with central bank policy uncertainty."
        }
    ]
    
    return {
        "insights": insights,
        "market_summary": {
            "total_assets_analyzed": 17,
            "bullish_signals": 12,
            "bearish_signals": 4,
            "neutral_signals": 1,
            "analysis_timestamp": datetime.now().isoformat(),
            "ai_model": "Octopus-Financial-Analyst"
        },
        "raw_ai_response": "Market analysis shows strong bullish momentum in tech sector with emerging opportunities in commodities. Risk management protocols recommend maintaining diversified exposure while monitoring crypto volatility signals."
    }

@llm_router.post("/finetune")
async def start_finetuning(output_dir: str = "./peft-output"):
    """
    Mock fine-tuning endpoint
    """
    return {
        "task_id": f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "started",
        "message": "Fine-tuning job initiated (mock mode)",
        "output_dir": output_dir
    }

@llm_router.get("/finetune/{task_id}")
async def get_finetuning_status(task_id: str):
    """
    Mock fine-tuning status endpoint
    """
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100,
        "message": "Fine-tuning completed successfully (mock mode)",
        "model_path": "./peft-output",
        "metrics": {
            "loss": 0.234,
            "eval_loss": 0.267,
            "perplexity": 12.45
        }
    }

@llm_router.post("/predict")
async def predict(text: str):
    """
    Mock prediction endpoint
    """
    return {
        "prediction": f"AI Analysis: Based on the input '{text[:50]}...', the market sentiment appears optimistic with moderate confidence.",
        "confidence": 0.75,
        "model": "octopus-financial-llm-mock"
    }

@llm_router.post("/llama/predict")
async def predict_llama(text: str):
    """
    Mock Llama prediction endpoint
    """
    return {
        "response": f"Financial Analysis: {text[:100]}... Based on current market conditions, I recommend a balanced approach with emphasis on risk management and diversification.",
        "model": "llama-financial-mock",
        "timestamp": datetime.now().isoformat()
    } 