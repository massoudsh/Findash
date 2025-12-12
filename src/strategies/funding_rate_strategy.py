"""
🐙 Octopus Trading Platform - Funding Rate Strategy
Cryptocurrency funding rate arbitrage and signal generation strategy.

This strategy monitors funding rates across crypto exchanges and generates
trading signals based on funding rate anomalies, trends, and arbitrage opportunities.
"""

import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from fastapi import BackgroundTasks

from .base import BaseStrategy, StrategyType
from ..core.cache import TradingCache
from ..core.config import get_settings
from ..core.exceptions import TradingError, ExternalServiceError

logger = logging.getLogger(__name__)

class FundingSignalType(Enum):
    """Types of funding rate signals"""
    EXTREME_POSITIVE = "extreme_positive"  # Very high funding rate (short bias)
    EXTREME_NEGATIVE = "extreme_negative"  # Very low funding rate (long bias)
    TRENDING_UP = "trending_up"           # Funding rate increasing
    TRENDING_DOWN = "trending_down"       # Funding rate decreasing
    MEAN_REVERSION = "mean_reversion"     # Rate moving back to average
    ARBITRAGE_OPPORTUNITY = "arbitrage"   # Cross-exchange opportunity

@dataclass
class FundingRateData:
    """Funding rate data structure"""
    symbol: str
    exchange: str
    funding_rate: float
    funding_time: int
    timestamp: datetime
    next_funding_time: Optional[int] = None
    predicted_rate: Optional[float] = None

@dataclass
class FundingAnalysis:
    """Funding rate analysis results"""
    symbol: str
    current_rate: float
    historical_avg: float
    volatility: float
    percentile_rank: float  # Where current rate sits in historical distribution
    trend_direction: str    # "up", "down", "stable"
    signal_strength: float  # 0-1
    confidence: float       # 0-1
    time_to_next: int      # Minutes until next funding
    arbitrage_score: float  # Cross-exchange arbitrage potential

class FundingRateStrategy(BaseStrategy):
    """
    🐙 Cryptocurrency Funding Rate Strategy
    
    Monitors funding rates across exchanges and generates signals based on:
    1. Extreme funding rates (contrarian signals)
    2. Funding rate trends and momentum
    3. Cross-exchange arbitrage opportunities
    4. Mean reversion patterns
    5. Correlation with price movements
    """
    
    def __init__(self, cache_manager: TradingCache):
        super().__init__()
        self.cache_manager = cache_manager
        self.settings = get_settings()
        
        # Exchange configurations
        self.exchanges = {
            "binance": {
                "funding_endpoint": "https://fapi.binance.com/fapi/v1/fundingRate",
                "current_endpoint": "https://fapi.binance.com/fapi/v1/premiumIndex",
                "enabled": True
            },
            "bybit": {
                "funding_endpoint": "https://api.bybit.com/v2/public/funding/prev-funding-rate",
                "current_endpoint": "https://api.bybit.com/v2/public/tickers",
                "enabled": False  # Enable when API keys available
            }
        }
        
        # Strategy parameters
        self.lookback_periods = [24, 72, 168]  # Hours: 1d, 3d, 1w
        self.extreme_thresholds = {
            "high": 0.01,      # 1% daily rate (very high)
            "moderate": 0.005,  # 0.5% daily rate
            "low": -0.005,     # -0.5% daily rate
            "very_low": -0.01  # -1% daily rate
        }
        
        # Cache TTL settings
        self.cache_ttl = {
            "funding_rate": 3600,      # 1 hour
            "historical": 7200,        # 2 hours
            "analysis": 1800           # 30 minutes
        }
    
    async def generate_signal(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on funding rate analysis.
        
        Args:
            market_data: Market price data
            parameters: Strategy parameters
            
        Returns:
            Trading signal dictionary or None
        """
        try:
            symbol = parameters.get('symbol', 'BTCUSDT')
            timeframe = parameters.get('timeframe', '1h')
            
            logger.info(f"🔍 Generating funding rate signal for {symbol}")
            
            # 1. Fetch current funding rate data
            funding_data = await self._fetch_current_funding_rate(symbol)
            if not funding_data:
                logger.warning(f"No funding rate data available for {symbol}")
                return None
            
            # 2. Get historical funding rate data
            historical_data = await self._fetch_historical_funding_rates(symbol)
            
            # 3. Perform funding rate analysis
            analysis = await self._analyze_funding_rates(symbol, funding_data, historical_data)
            
            # 4. Generate signal based on analysis
            signal = await self._generate_funding_signal(analysis, market_data, parameters)
            
            # 5. Cache the analysis results
            cache_key = f"funding_analysis:{symbol}:{datetime.utcnow().hour}"
            await self.cache_manager.set(cache_key, analysis.__dict__, ttl=self.cache_ttl["analysis"])
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating funding rate signal: {e}")
            return None
    
    async def _fetch_current_funding_rate(self, symbol: str) -> Optional[FundingRateData]:
        """Fetch current funding rate from primary exchange (Binance)"""
        try:
            # Check cache first
            cache_key = f"funding_current:{symbol}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                return FundingRateData(**cached_data)
            
            # Fetch from Binance
            exchange_config = self.exchanges["binance"]
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get current funding rate
                params = {"symbol": symbol.upper()}
                response = await client.get(exchange_config["current_endpoint"], params=params)
                response.raise_for_status()
                
                premium_data = response.json()
                if not premium_data:
                    logger.warning(f"No data returned from Binance for {symbol}")
                    return None
                
                # Binance premiumIndex returns a single object, not an array
                if isinstance(premium_data, list):
                    premium_data = premium_data[0] if premium_data else None
                
                if not premium_data:
                    logger.warning(f"Empty premium data for {symbol}")
                    return None
                
                current_rate = float(premium_data.get('lastFundingRate', 0))
                next_funding_time = int(premium_data.get('nextFundingTime', 0))
                
                funding_data = FundingRateData(
                    symbol=symbol,
                    exchange="binance",
                    funding_rate=current_rate,
                    funding_time=int(datetime.utcnow().timestamp() * 1000),
                    timestamp=datetime.utcnow(),
                    next_funding_time=next_funding_time
                )
                
                # Cache the result
                await self.cache_manager.set(
                    cache_key, 
                    funding_data.__dict__, 
                    ttl=self.cache_ttl["funding_rate"]
                )
                
                logger.info(f"💰 Current funding rate for {symbol}: {current_rate:.6f}")
                return funding_data
                
        except Exception as e:
            logger.error(f"Error fetching current funding rate for {symbol}: {e}")
            raise ExternalServiceError(f"Failed to fetch funding rate: {e}")
    
    async def _fetch_historical_funding_rates(self, symbol: str, limit: int = 500) -> List[FundingRateData]:
        """Fetch historical funding rates"""
        try:
            # Check cache first
            cache_key = f"funding_historical:{symbol}:{limit}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                return [FundingRateData(**item) for item in cached_data]
            
            exchange_config = self.exchanges["binance"]
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {
                    "symbol": symbol.upper(),
                    "limit": limit
                }
                response = await client.get(exchange_config["funding_endpoint"], params=params)
                response.raise_for_status()
                
                historical_data = response.json()
                
                funding_rates = []
                for item in historical_data:
                    funding_data = FundingRateData(
                        symbol=symbol,
                        exchange="binance",
                        funding_rate=float(item['fundingRate']),
                        funding_time=int(item['fundingTime']),
                        timestamp=datetime.fromtimestamp(int(item['fundingTime']) / 1000)
                    )
                    funding_rates.append(funding_data)
                
                # Cache the historical data
                cached_items = [item.__dict__ for item in funding_rates]
                await self.cache_manager.set(
                    cache_key, 
                    cached_items, 
                    ttl=self.cache_ttl["historical"]
                )
                
                logger.info(f"📊 Fetched {len(funding_rates)} historical funding rates for {symbol}")
                return funding_rates
                
        except Exception as e:
            logger.error(f"Error fetching historical funding rates for {symbol}: {e}")
            return []
    
    async def _analyze_funding_rates(self, symbol: str, current: FundingRateData, 
                                   historical: List[FundingRateData]) -> FundingAnalysis:
        """Perform comprehensive funding rate analysis"""
        try:
            if not historical:
                logger.warning(f"No historical data available for {symbol}")
                return self._create_default_analysis(symbol, current)
            
            # Convert to pandas for analysis
            rates = [item.funding_rate for item in historical]
            rates_df = pd.Series(rates)
            
            # Statistical analysis
            historical_avg = rates_df.mean()
            volatility = rates_df.std()
            percentile_rank = (rates_df <= current.funding_rate).mean()
            
            # Trend analysis (last 24 hours)
            recent_rates = rates_df.tail(8) if len(rates_df) >= 8 else rates_df  # Last 8 funding periods
            trend_direction = self._calculate_trend(recent_rates)
            
            # Signal strength calculation
            signal_strength = self._calculate_signal_strength(current.funding_rate, historical_avg, volatility)
            
            # Confidence calculation
            confidence = self._calculate_confidence(current, historical, volatility)
            
            # Time to next funding
            time_to_next = self._calculate_time_to_next_funding(current.next_funding_time)
            
            # Arbitrage score (placeholder for cross-exchange analysis)
            arbitrage_score = 0.0  # Would require multiple exchange data
            
            analysis = FundingAnalysis(
                symbol=symbol,
                current_rate=current.funding_rate,
                historical_avg=historical_avg,
                volatility=volatility,
                percentile_rank=percentile_rank,
                trend_direction=trend_direction,
                signal_strength=signal_strength,
                confidence=confidence,
                time_to_next=time_to_next,
                arbitrage_score=arbitrage_score
            )
            
            logger.info(f"📈 Funding analysis for {symbol}: Rate={current.funding_rate:.6f}, "
                       f"Percentile={percentile_rank:.2f}, Strength={signal_strength:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing funding rates for {symbol}: {e}")
            return self._create_default_analysis(symbol, current)
    
    def _calculate_trend(self, recent_rates: pd.Series) -> str:
        """Calculate funding rate trend direction"""
        if len(recent_rates) < 3:
            return "stable"
        
        # Linear regression to determine trend
        x = np.arange(len(recent_rates))
        slope = np.polyfit(x, recent_rates.values, 1)[0]
        
        if slope > 0.0001:  # 0.01% threshold
            return "up"
        elif slope < -0.0001:
            return "down"
        else:
            return "stable"
    
    def _calculate_signal_strength(self, current_rate: float, avg_rate: float, volatility: float) -> float:
        """Calculate signal strength based on deviation from average"""
        if volatility == 0:
            return 0.0
        
        # Z-score based strength
        z_score = abs(current_rate - avg_rate) / volatility
        
        # Normalize to 0-1 range
        strength = min(z_score / 3.0, 1.0)  # 3 standard deviations = max strength
        
        return strength
    
    def _calculate_confidence(self, current: FundingRateData, historical: List[FundingRateData], 
                            volatility: float) -> float:
        """Calculate confidence in the signal"""
        base_confidence = 0.5
        
        # Higher confidence with more data
        data_factor = min(len(historical) / 100, 1.0)  # Max confidence with 100+ data points
        
        # Lower confidence with high volatility
        volatility_factor = max(0.3, 1.0 - (volatility * 100))  # Adjust for volatility scale
        
        # Time-based confidence (higher closer to funding time)
        time_factor = 1.0
        if current.next_funding_time:
            time_to_funding = self._calculate_time_to_next_funding(current.next_funding_time)
            time_factor = max(0.5, 1.0 - (time_to_funding / 480))  # Lower confidence if >8h away
        
        confidence = base_confidence * data_factor * volatility_factor * time_factor
        return min(confidence, 0.95)  # Cap at 95%
    
    def _calculate_time_to_next_funding(self, next_funding_time: Optional[int]) -> int:
        """Calculate minutes until next funding"""
        if not next_funding_time:
            return 480  # Default 8 hours if unknown
        
        current_timestamp = int(datetime.utcnow().timestamp() * 1000)
        time_diff_ms = next_funding_time - current_timestamp
        
        return max(0, int(time_diff_ms / (1000 * 60)))  # Convert to minutes
    
    def _create_default_analysis(self, symbol: str, current: FundingRateData) -> FundingAnalysis:
        """Create default analysis when historical data is unavailable"""
        return FundingAnalysis(
            symbol=symbol,
            current_rate=current.funding_rate,
            historical_avg=0.0,
            volatility=0.001,  # Default volatility
            percentile_rank=0.5,
            trend_direction="stable",
            signal_strength=0.1,
            confidence=0.3,  # Low confidence without historical data
            time_to_next=self._calculate_time_to_next_funding(current.next_funding_time),
            arbitrage_score=0.0
        )
    
    async def _generate_funding_signal(self, analysis: FundingAnalysis, market_data: pd.DataFrame,
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on funding analysis"""
        try:
            signal_type = "hold"
            action_confidence = 0.0
            signal_strength = 0.0
            reasoning = []
            
            current_rate = analysis.current_rate
            percentile = analysis.percentile_rank
            
            # Extreme funding rate signals (contrarian approach)
            if current_rate > self.extreme_thresholds["high"]:
                signal_type = "sell"  # High funding = many longs = potential reversal
                action_confidence = min(0.8, analysis.confidence + 0.2)
                signal_strength = analysis.signal_strength
                reasoning.append(f"Extremely high funding rate: {current_rate:.4f} (>{self.extreme_thresholds['high']:.4f})")
                
            elif current_rate < self.extreme_thresholds["very_low"]:
                signal_type = "buy"   # Very low funding = many shorts = potential reversal
                action_confidence = min(0.8, analysis.confidence + 0.2)
                signal_strength = analysis.signal_strength
                reasoning.append(f"Extremely low funding rate: {current_rate:.4f} (<{self.extreme_thresholds['very_low']:.4f})")
            
            # Moderate extremes
            elif current_rate > self.extreme_thresholds["moderate"] and percentile > 0.8:
                signal_type = "sell"
                action_confidence = analysis.confidence
                signal_strength = analysis.signal_strength * 0.7
                reasoning.append(f"High funding rate in 80th percentile: {current_rate:.4f}")
                
            elif current_rate < self.extreme_thresholds["low"] and percentile < 0.2:
                signal_type = "buy"
                action_confidence = analysis.confidence
                signal_strength = analysis.signal_strength * 0.7
                reasoning.append(f"Low funding rate in 20th percentile: {current_rate:.4f}")
            
            # Trend-based signals
            if analysis.trend_direction == "up" and signal_type == "hold":
                signal_type = "sell"  # Rising funding might indicate overheated longs
                action_confidence = analysis.confidence * 0.6
                signal_strength = 0.4
                reasoning.append("Funding rate trending upward (long bias increasing)")
                
            elif analysis.trend_direction == "down" and signal_type == "hold":
                signal_type = "buy"   # Falling funding might indicate oversold
                action_confidence = analysis.confidence * 0.6
                signal_strength = 0.4
                reasoning.append("Funding rate trending downward (short bias increasing)")
            
            # Time-based urgency
            if analysis.time_to_next < 60:  # Less than 1 hour to funding
                action_confidence *= 1.2  # Increase confidence near funding time
                reasoning.append(f"Funding payment in {analysis.time_to_next} minutes")
            
            # Minimum thresholds
            if action_confidence < 0.3 or signal_strength < 0.2:
                signal_type = "hold"
                action_confidence = 0.1
                signal_strength = 0.1
                reasoning = ["Insufficient signal strength for action"]
            
            return {
                "action": signal_type,
                "confidence": min(action_confidence, 0.95),
                "strength": min(signal_strength, 1.0),
                "features": {
                    "funding_rate": current_rate,
                    "funding_percentile": percentile,
                    "historical_avg": analysis.historical_avg,
                    "volatility": analysis.volatility,
                    "trend": analysis.trend_direction,
                    "time_to_funding": analysis.time_to_next
                },
                "metadata": {
                    "strategy_type": "funding_rate",
                    "signal_reasoning": reasoning,
                    "funding_analysis": analysis.__dict__,
                    "exchange": "binance"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating funding signal: {e}")
            return {
                "action": "hold",
                "confidence": 0.1,
                "strength": 0.1,
                "features": {},
                "metadata": {"error": str(e)}
            }
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate strategy parameters"""
        required = ["symbol"]
        for param in required:
            if param not in parameters:
                logger.error(f"Missing required parameter: {param}")
                return False
        
        symbol = parameters["symbol"]
        if not symbol.endswith("USDT"):
            logger.warning(f"Symbol {symbol} might not be supported for funding rates")
        
        return True
    
    async def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and status"""
        return {
            "name": "Funding Rate Strategy",
            "type": "funding_rate",
            "description": "Cryptocurrency funding rate arbitrage and contrarian signals",
            "supported_exchanges": list(self.exchanges.keys()),
            "enabled_exchanges": [k for k, v in self.exchanges.items() if v["enabled"]],
            "parameters": {
                "symbol": "Trading symbol (e.g., BTCUSDT)",
                "timeframe": "Analysis timeframe",
                "extreme_thresholds": self.extreme_thresholds
            },
            "features": [
                "Real-time funding rate monitoring",
                "Historical analysis and percentile ranking",
                "Trend detection and momentum signals",
                "Contrarian signals based on extreme rates",
                "Time-based urgency adjustments"
            ]
        }


# FastAPI endpoints for funding rate access (standalone usage)
from fastapi import FastAPI, BackgroundTasks, HTTPException

def create_funding_rate_api(cache_manager: TradingCache) -> FastAPI:
    """Create FastAPI app for funding rate endpoints"""
    
    app = FastAPI(title="🐙 Octopus Funding Rate API", version="1.0.0")
    strategy = FundingRateStrategy(cache_manager)
    
    @app.get("/funding/{symbol}")
    async def get_funding_rate(symbol: str, background_tasks: BackgroundTasks):
        """Get current funding rate for a symbol"""
        try:
            cache_key = f"funding_current:{symbol}"
            cached_data = await cache_manager.get(cache_key)
            
            if cached_data:
                return cached_data
            
            # Fetch in background if not cached
            background_tasks.add_task(strategy._fetch_current_funding_rate, symbol)
            return {"status": "Fetching funding rate in background. Try again shortly."}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/funding/{symbol}/refresh")
    async def refresh_funding_rate(symbol: str):
        """Force refresh funding rate for a symbol"""
        try:
            funding_data = await strategy._fetch_current_funding_rate(symbol)
            if funding_data:
                return funding_data.__dict__
            else:
                raise HTTPException(status_code=404, detail="No funding data found")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/funding/{symbol}/analysis")
    async def get_funding_analysis(symbol: str, limit: int = 100):
        """Get comprehensive funding rate analysis"""
        try:
            current = await strategy._fetch_current_funding_rate(symbol)
            if not current:
                raise HTTPException(status_code=404, detail="No current funding data")
            
            historical = await strategy._fetch_historical_funding_rates(symbol, limit)
            analysis = await strategy._analyze_funding_rates(symbol, current, historical)
            
            return analysis.__dict__
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/funding/{symbol}/signal")
    async def get_funding_signal(symbol: str, timeframe: str = "1h"):
        """Generate trading signal based on funding rates"""
        try:
            # Mock market data for signal generation
            market_data = pd.DataFrame()  # Would normally fetch real market data
            parameters = {"symbol": symbol, "timeframe": timeframe}
            
            signal = await strategy.generate_signal(market_data, parameters)
            if signal:
                return signal
            else:
                raise HTTPException(status_code=404, detail="Could not generate signal")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app