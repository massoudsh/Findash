"""
Agent Panels Stub API (Phase 2 – Issue #3)
Stub endpoints for Trading Center agent panels: M1 Data Collector, M4 Strategy, M9 Sentiment, M11 Insights.
Frontend can fetch from these when backend is available; replace with real data sources later.
"""

from fastapi import APIRouter
from typing import List

router = APIRouter(prefix="/api/agent-panels", tags=["Agent Panels"])


# ---------- M1 Data Collector ----------
def _data_collector_sources() -> List[dict]:
    return [
        {"id": "market", "name": "Market Data", "type": "market_data", "status": "active", "lastSync": "12s ago", "recordsToday": 15420},
        {"id": "news", "name": "News Feed", "type": "news", "status": "active", "lastSync": "1m ago", "recordsToday": 2847},
        {"id": "social", "name": "Social Sentiment", "type": "social", "status": "degraded", "lastSync": "5m ago", "recordsToday": 892},
        {"id": "fundamental", "name": "Fundamental", "type": "fundamental", "status": "active", "lastSync": "2m ago", "recordsToday": 1205},
        {"id": "onchain", "name": "On-chain", "type": "on_chain", "status": "active", "lastSync": "45s ago", "recordsToday": 3421},
    ]


@router.get("/data-collector")
async def get_data_collector_sources():
    """M1 Data Collector: pipeline status and ingestion health (stub)."""
    return {"sources": _data_collector_sources()}


# ---------- M4 Strategy signals ----------
def _strategy_signals() -> List[dict]:
    return [
        {"id": "1", "strategy": "momentum", "symbol": "NVDA", "side": "long", "strength": 0.82, "timestamp": "2m ago"},
        {"id": "2", "strategy": "mean_reversion", "symbol": "SPY", "side": "long", "strength": 0.61, "timestamp": "5m ago"},
        {"id": "3", "strategy": "trend_following", "symbol": "ETH-USD", "side": "short", "strength": 0.55, "timestamp": "8m ago"},
        {"id": "4", "strategy": "momentum", "symbol": "AAPL", "side": "long", "strength": 0.71, "timestamp": "12m ago"},
    ]


@router.get("/strategy-signals")
async def get_strategy_signals():
    """M4 Strategy Agent: active signals and execution feed (stub)."""
    return {"signals": _strategy_signals()}


# ---------- M9 Sentiment ----------
def _sentiment_buckets() -> List[dict]:
    return [
        {"symbol": "BTC-USD", "sentiment": "positive", "score": 0.72, "source": "social"},
        {"symbol": "NVDA", "sentiment": "positive", "score": 0.68, "source": "news"},
        {"symbol": "TSLA", "sentiment": "neutral", "score": 0.48, "source": "social"},
        {"symbol": "AAPL", "sentiment": "positive", "score": 0.61, "source": "news"},
    ]


@router.get("/sentiment")
async def get_sentiment():
    """M9 Sentiment Agent: news and social sentiment by asset (stub)."""
    return {"items": _sentiment_buckets()}


# ---------- M11 Analysis insights ----------
def _insights() -> List[dict]:
    return [
        {"id": "insight-1", "source": "technical", "title": "RSI divergence", "summary": "AAPL 4H RSI divergence suggests near-term pullback.", "signal": "bearish", "symbol": "AAPL", "timestamp": "Just now"},
        {"id": "insight-2", "source": "fundamental", "title": "Earnings beat", "summary": "NVDA forward P/E supportive; institutional flow positive.", "signal": "bullish", "symbol": "NVDA", "timestamp": "2m ago"},
        {"id": "insight-3", "source": "macro", "title": "Rates hold", "summary": "Fed hold priced in; DXY weakness supports risk assets.", "signal": "bullish", "symbol": None, "timestamp": "4m ago"},
        {"id": "insight-4", "source": "on-chain", "title": "Whale accumulation", "summary": "Large ETH wallets adding; exchange outflow rising.", "signal": "bullish", "symbol": "ETH", "timestamp": "6m ago"},
        {"id": "insight-5", "source": "social", "title": "Sentiment shift", "summary": "Twitter/X sentiment for BTC turned positive (7d).", "signal": "bullish", "symbol": "BTC", "timestamp": "8m ago"},
        {"id": "insight-6", "source": "ai-models", "title": "Regime: risk-on", "summary": "Ensemble model assigns 0.78 to risk-on regime next 5d.", "signal": "bullish", "symbol": None, "timestamp": "10m ago"},
        {"id": "insight-7", "source": "technical", "title": "Support level", "summary": "SPY holding above 580; volume declining on pullbacks.", "signal": "neutral", "symbol": "SPY", "timestamp": "12m ago"},
        {"id": "insight-8", "source": "fundamental", "title": "Sector rotation", "summary": "Fund flows into tech; defensives outflow.", "signal": "bullish", "symbol": None, "timestamp": "14m ago"},
    ]


@router.get("/insights")
async def get_insights():
    """M11 Analysis Agent: real-time insights from technical, fundamental, macro, on-chain, social, AI (stub)."""
    return {"insights": _insights()}
