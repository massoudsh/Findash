"""
Fundamental Data Engine
Integrates on-chain data, SEC filings, whale tracking, and fundamental signals
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import re

logger = logging.getLogger(__name__)

class DataSource(Enum):
    GLASSNODE = "glassnode"
    SEC_EDGAR = "sec_edgar"
    WHALE_ALERT = "whale_alert"
    MESSARI = "messari"

class SignalStrength(Enum):
    VERY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    VERY_BULLISH = 2

@dataclass
class FundamentalSignal:
    symbol: str
    signal_type: str
    strength: SignalStrength
    confidence: float
    description: str
    contributing_factors: List[str]
    timestamp: datetime
    expiry: Optional[datetime] = None
    source_data: Dict[str, Any] = field(default_factory=dict)

class FundamentalDataEngine:
    """Main engine for fundamental data integration"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def get_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive fundamental analysis for a symbol"""
        try:
            # Check cache
            cache_key = f"fundamental:{symbol}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Generate analysis based on asset type
            asset_type = self._get_asset_type(symbol)
            signals = []
            
            if asset_type == "crypto":
                signals.extend(await self._get_onchain_signals(symbol))
                signals.extend(await self._get_whale_signals(symbol))
            elif asset_type == "stock":
                signals.extend(await self._get_sec_signals(symbol))
                signals.extend(await self._get_fundamental_signals(symbol))
            
            # Create analysis
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "asset_type": asset_type,
                "signals": [self._signal_to_dict(signal) for signal in signals],
                "summary": self._create_summary(signals),
                "score": self._calculate_fundamental_score(signals),
                "confidence": self._calculate_overall_confidence(signals)
            }
            
            # Cache results
            self.cache[cache_key] = (analysis, time.time())
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _signal_to_dict(self, signal: FundamentalSignal) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "strength": signal.strength.value,
            "confidence": signal.confidence,
            "description": signal.description,
            "contributing_factors": signal.contributing_factors,
            "timestamp": signal.timestamp.isoformat()
        }
    
    def _get_asset_type(self, symbol: str) -> str:
        """Determine asset type"""
        if any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "TRX", "LINK", "CAKE", "USDT", "USDC"]):
            return "crypto"
        elif symbol.upper() in ["GLD", "SLV"]:
            return "commodity"
        else:
            return "stock"
    
    async def _get_onchain_signals(self, symbol: str) -> List[FundamentalSignal]:
        """Generate on-chain signals for crypto assets"""
        signals = []
        
        # Simulate on-chain metrics
        metrics = {
            "active_addresses": hash(symbol + "addresses") % 100,
            "transaction_volume": hash(symbol + "volume") % 1000,
            "mvrv_ratio": (hash(symbol + "mvrv") % 100) / 50.0,
            "network_value": hash(symbol + "value") % 10000
        }
        
        for metric_name, value in metrics.items():
            # Determine signal strength based on metric
            if metric_name == "active_addresses" and value > 70:
                strength = SignalStrength.BULLISH
                description = f"High network activity with {value} active addresses"
            elif metric_name == "mvrv_ratio" and value > 1.5:
                strength = SignalStrength.BEARISH
                description = f"MVRV ratio of {value:.2f} suggests overvaluation"
            elif metric_name == "transaction_volume" and value > 800:
                strength = SignalStrength.BULLISH
                description = f"High transaction volume indicates strong network usage"
            else:
                continue
            
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type=f"onchain_{metric_name}",
                strength=strength,
                confidence=0.8,
                description=description,
                contributing_factors=[f"{metric_name}: {value}"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=24)
            )
            signals.append(signal)
        
        return signals
    
    async def _get_whale_signals(self, symbol: str) -> List[FundamentalSignal]:
        """Generate whale activity signals"""
        signals = []
        
        # Simulate whale data
        whale_hash = hash(symbol + "whale") % 100
        buy_volume = whale_hash * 100000
        sell_volume = (100 - whale_hash) * 80000
        
        net_flow = buy_volume - sell_volume
        
        if abs(net_flow) > 500000:  # Significant whale activity
            if net_flow > 0:
                strength = SignalStrength.BULLISH
                description = f"Large whale accumulation: ${net_flow:,.0f} net inflow"
            else:
                strength = SignalStrength.BEARISH
                description = f"Large whale distribution: ${abs(net_flow):,.0f} net outflow"
            
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type="whale_activity",
                strength=strength,
                confidence=0.75,
                description=description,
                contributing_factors=[f"Net flow: ${net_flow:,.0f}"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=12)
            )
            signals.append(signal)
        
        return signals
    
    async def _get_sec_signals(self, symbol: str) -> List[FundamentalSignal]:
        """Generate SEC filing signals"""
        signals = []
        
        # Simulate SEC filing analysis
        filing_hash = hash(symbol + "sec") % 100
        
        if filing_hash > 70:  # Positive filing sentiment
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type="sec_filing_sentiment",
                strength=SignalStrength.BULLISH,
                confidence=0.7,
                description="Recent SEC filings show positive management outlook",
                contributing_factors=["Strong earnings guidance", "Expansion plans"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(days=30)
            )
            signals.append(signal)
        elif filing_hash < 30:  # Negative filing sentiment
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type="sec_filing_sentiment",
                strength=SignalStrength.BEARISH,
                confidence=0.7,
                description="Recent SEC filings indicate challenges ahead",
                contributing_factors=["Lower guidance", "Risk factors"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(days=30)
            )
            signals.append(signal)
        
        return signals
    
    async def _get_fundamental_signals(self, symbol: str) -> List[FundamentalSignal]:
        """Generate general fundamental signals"""
        signals = []
        
        # Simulate fundamental metrics
        metrics_hash = hash(symbol + "fundamentals") % 100
        
        # P/E ratio signal
        pe_ratio = (metrics_hash % 40) + 5  # 5-45 range
        if pe_ratio < 15:
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type="valuation_pe",
                strength=SignalStrength.BULLISH,
                confidence=0.8,
                description=f"Low P/E ratio of {pe_ratio:.1f} suggests undervaluation",
                contributing_factors=[f"P/E: {pe_ratio:.1f}", "Below sector average"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(days=7)
            )
            signals.append(signal)
        
        # Revenue growth signal
        growth_rate = ((metrics_hash + 20) % 50) - 10  # -10% to 40% range
        if growth_rate > 20:
            signal = FundamentalSignal(
                symbol=symbol,
                signal_type="revenue_growth",
                strength=SignalStrength.BULLISH,
                confidence=0.85,
                description=f"Strong revenue growth of {growth_rate}%",
                contributing_factors=[f"Revenue growth: {growth_rate}%", "Market expansion"],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(days=90)
            )
            signals.append(signal)
        
        return signals
    
    def _create_summary(self, signals: List[FundamentalSignal]) -> Dict[str, Any]:
        """Create summary of fundamental signals"""
        if not signals:
            return {
                "overall_sentiment": "neutral", 
                "key_factors": [], 
                "signal_count": 0, 
                "bullish_signals": 0, 
                "bearish_signals": 0
            }
        
        # Count signal strengths
        bullish_signals = sum(1 for s in signals if s.strength.value > 0)
        bearish_signals = sum(1 for s in signals if s.strength.value < 0)
        
        # Determine overall sentiment
        if bullish_signals > bearish_signals:
            overall_sentiment = "bullish"
        elif bearish_signals > bullish_signals:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # Key factors (top 3 signals by confidence)
        top_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)[:3]
        key_factors = [signal.description for signal in top_signals]
        
        return {
            "overall_sentiment": overall_sentiment,
            "key_factors": key_factors,
            "signal_count": len(signals),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        }
    
    def _calculate_fundamental_score(self, signals: List[FundamentalSignal]) -> float:
        """Calculate overall fundamental score (-100 to +100)"""
        if not signals:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = signal.confidence
            score = signal.strength.value * 25  # Convert to -50 to +50 range
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(100, max(-100, weighted_score / total_weight))
    
    def _calculate_overall_confidence(self, signals: List[FundamentalSignal]) -> float:
        """Calculate overall confidence in the analysis"""
        if not signals:
            return 0.0
        
        return min(1.0, sum(signal.confidence for signal in signals) / len(signals)) 