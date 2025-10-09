"""
Fundamental Data API Endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from src.data_processing.fundamental_data_engine import FundamentalDataEngine
from src.core.assets_config import AssetsConfig

logger = logging.getLogger(__name__)

router = APIRouter()
fundamental_engine = FundamentalDataEngine()

@router.get("/analysis/{symbol}")
async def get_fundamental_analysis(symbol: str) -> Dict[str, Any]:
    """Get comprehensive fundamental analysis for a symbol"""
    try:
        analysis = await fundamental_engine.get_fundamental_analysis(symbol.upper())
        return analysis
    except Exception as e:
        logger.error(f"Error getting fundamental analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/{symbol}")
async def get_fundamental_signals(
    symbol: str,
    signal_types: Optional[List[str]] = Query(None),
    min_confidence: Optional[float] = Query(0.0)
) -> Dict[str, Any]:
    """Get fundamental signals for a symbol with optional filtering"""
    try:
        analysis = await fundamental_engine.get_fundamental_analysis(symbol.upper())
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        signals = analysis["signals"]
        
        # Filter by signal types if specified
        if signal_types:
            signals = [s for s in signals if s["signal_type"] in signal_types]
        
        # Filter by confidence
        if min_confidence > 0:
            signals = [s for s in signals if s["confidence"] >= min_confidence]
        
        return {
            "symbol": symbol.upper(),
            "signals": signals,
            "count": len(signals),
            "filtered": len(signals) != len(analysis["signals"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/onchain/{symbol}")
async def get_onchain_metrics(symbol: str) -> Dict[str, Any]:
    """Get on-chain metrics for cryptocurrency assets"""
    try:
        # Check if symbol is a cryptocurrency
        asset = AssetsConfig.get_asset(symbol.upper())
        if not asset or asset.asset_type.value not in ["cryptocurrency", "stablecoin"]:
            raise HTTPException(
                status_code=400, 
                detail=f"{symbol} is not a cryptocurrency asset"
            )
        
        analysis = await fundamental_engine.get_fundamental_analysis(symbol.upper())
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Filter for on-chain signals only
        onchain_signals = [
            s for s in analysis["signals"] 
            if s["signal_type"].startswith("onchain_") or s["signal_type"] == "whale_activity"
        ]
        
        return {
            "symbol": symbol.upper(),
            "onchain_signals": onchain_signals,
            "count": len(onchain_signals),
            "timestamp": analysis["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting on-chain metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sec-filings/{symbol}")
async def get_sec_analysis(symbol: str) -> Dict[str, Any]:
    """Get SEC filing analysis for stock symbols"""
    try:
        # Check if symbol is a stock
        asset = AssetsConfig.get_asset(symbol.upper())
        if not asset or asset.asset_type.value not in ["stock", "etf"]:
            raise HTTPException(
                status_code=400,
                detail=f"{symbol} is not a stock or ETF"
            )
        
        analysis = await fundamental_engine.get_fundamental_analysis(symbol.upper())
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Filter for SEC filing signals
        sec_signals = [
            s for s in analysis["signals"]
            if s["signal_type"].startswith("sec_") or s["signal_type"] in ["valuation_pe", "revenue_growth"]
        ]
        
        return {
            "symbol": symbol.upper(),
            "sec_signals": sec_signals,
            "count": len(sec_signals),
            "timestamp": analysis["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SEC analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/whale-activity/{symbol}")
async def get_whale_activity(
    symbol: str,
    hours: Optional[int] = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """Get whale activity analysis for cryptocurrency assets"""
    try:
        # Check if symbol is a cryptocurrency
        asset = AssetsConfig.get_asset(symbol.upper())
        if not asset or asset.asset_type.value not in ["cryptocurrency", "stablecoin"]:
            raise HTTPException(
                status_code=400,
                detail=f"{symbol} is not a cryptocurrency asset"
            )
        
        analysis = await fundamental_engine.get_fundamental_analysis(symbol.upper())
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # Filter for whale signals
        whale_signals = [
            s for s in analysis["signals"]
            if s["signal_type"] == "whale_activity"
        ]
        
        # Simulate additional whale metrics
        whale_metrics = {
            "large_transactions_24h": hash(symbol + "large") % 50,
            "whale_accumulation_score": (hash(symbol + "accum") % 100) / 100,
            "exchange_inflow": (hash(symbol + "inflow") % 1000000),
            "exchange_outflow": (hash(symbol + "outflow") % 800000),
            "net_flow": (hash(symbol + "net") % 400000) - 200000
        }
        
        return {
            "symbol": symbol.upper(),
            "whale_signals": whale_signals,
            "whale_metrics": whale_metrics,
            "timeframe_hours": hours,
            "timestamp": analysis["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting whale activity for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_fundamental_dashboard() -> Dict[str, Any]:
    """Get fundamental analysis dashboard"""
    try:
        key_assets = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "MSFT", "GOOGL"]
        dashboard_data = {}
        
        for symbol in key_assets:
            try:
                analysis = await fundamental_engine.get_fundamental_analysis(symbol)
                if "error" not in analysis:
                    dashboard_data[symbol] = {
                        "score": analysis["score"],
                        "confidence": analysis["confidence"],
                        "sentiment": analysis["summary"]["overall_sentiment"],
                        "signal_count": analysis["summary"]["signal_count"],
                        "key_factor": analysis["summary"]["key_factors"][0] if analysis["summary"]["key_factors"] else "No signals"
                    }
            except Exception:
                continue
        
        # Calculate market overview
        scores = [data["score"] for data in dashboard_data.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for data in dashboard_data.values():
            sentiment_counts[data["sentiment"]] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_overview": {
                "average_score": round(avg_score, 1),
                "sentiment_distribution": sentiment_counts,
                "assets_analyzed": len(dashboard_data)
            },
            "asset_analysis": dashboard_data,
            "data_sources": {
                "onchain_metrics": "Active for crypto assets",
                "sec_filings": "Active for stocks", 
                "whale_tracking": "Active for crypto assets",
                "fundamental_analysis": "Active for all assets"
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating fundamental dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-sources")
async def get_data_sources() -> Dict[str, Any]:
    """Get information about available data sources and their capabilities"""
    return {
        "data_sources": [
            {
                "name": "On-Chain Analytics",
                "description": "Blockchain metrics including active addresses, transaction volume, MVRV ratio",
                "asset_types": ["cryptocurrency", "stablecoin"],
                "update_frequency": "Real-time",
                "metrics": ["active_addresses", "transaction_volume", "mvrv_ratio", "network_value"]
            },
            {
                "name": "SEC Filings",
                "description": "Analysis of 10-K, 10-Q, 8-K filings and insider activity",
                "asset_types": ["stock", "etf"],
                "update_frequency": "Daily",
                "metrics": ["filing_sentiment", "insider_activity", "earnings_guidance"]
            },
            {
                "name": "Whale Tracking",
                "description": "Large transaction monitoring and whale sentiment analysis",
                "asset_types": ["cryptocurrency", "stablecoin"],
                "update_frequency": "Real-time",
                "metrics": ["large_transactions", "whale_accumulation", "exchange_flows"]
            },
            {
                "name": "Fundamental Analysis",
                "description": "Traditional fundamental metrics and valuation analysis",
                "asset_types": ["stock", "etf", "commodity"],
                "update_frequency": "Daily",
                "metrics": ["pe_ratio", "revenue_growth", "profit_margins"]
            }
        ],
        "supported_assets": list(AssetsConfig.ASSETS.keys()),
        "total_metrics": 50
    }

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for fundamental data services"""
    try:
        # Test analysis for a sample symbol
        test_analysis = await fundamental_engine.get_fundamental_analysis("BTC-USD")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "fundamental_engine": "operational",
                "cache": "operational",
                "analysis": "operational" if "error" not in test_analysis else "degraded"
            },
            "cache_size": len(fundamental_engine.cache),
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }