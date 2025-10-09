"""
M4 - Strategy Agent
Central orchestrator for trading strategies, signal fusion, and decision making.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from .signal_fusion import SignalFusionEngine, TradingSignal, FusedSignal, SignalType
from .base import BaseStrategy
from .momentum import MomentumStrategy
from .technical_analysis import TechnicalAnalysisStrategy
from .risk_aware import RiskAwareStrategy
from .funding_rate_strategy import FundingRateStrategy
from ..core.cache import TradingCache
from ..core.exceptions import TradingError, StrategyError
from ..prediction.prophet_service import ProphetService
from ..training.distillation_trainer import DistillationTrainer


class StrategyType(Enum):
    """Available strategy types."""
    MOMENTUM = "momentum"
    TECHNICAL = "technical"
    RISK_AWARE = "risk_aware"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SENTIMENT_DRIVEN = "sentiment_driven"
    FUNDING_RATE = "funding_rate"


class MarketRegime(Enum):
    """Market regime types for strategy selection."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"


@dataclass
class StrategyAllocation:
    """Strategy allocation with weights and parameters."""
    strategy_type: StrategyType
    weight: float
    parameters: Dict[str, Any]
    expected_return: float
    risk_score: float
    active: bool = True


@dataclass
class TradingDecision:
    """Final trading decision from the Strategy Agent."""
    symbol: str
    action: SignalType
    confidence: float
    position_size: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    time_horizon: timedelta
    strategy_allocation: Dict[StrategyType, float]
    risk_metrics: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "time_horizon_minutes": self.time_horizon.total_seconds() / 60,
            "strategy_allocation": {k.value: v for k, v in self.strategy_allocation.items()},
            "risk_metrics": self.risk_metrics,
            "timestamp": self.timestamp.isoformat()
        }


class StrategyAgent:
    """
    M4 - Strategy Agent
    Central orchestrator that combines multiple strategies and intelligence signals
    to make optimal trading decisions.
    """
    
    def __init__(self, cache: TradingCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Initialize signal fusion engine
        self.signal_fusion = SignalFusionEngine(cache)
        
        # Initialize strategies
        self.strategies: Dict[StrategyType, BaseStrategy] = {
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.TECHNICAL: TechnicalAnalysisStrategy(),
            StrategyType.RISK_AWARE: RiskAwareStrategy(),
            StrategyType.FUNDING_RATE: FundingRateStrategy(cache)
        }
        
        # Current market regime and strategy allocations
        self.current_regime = MarketRegime.UNCERTAIN
        self.strategy_allocations: Dict[StrategyType, StrategyAllocation] = {}
        
        # Performance tracking
        self.strategy_performance: Dict[StrategyType, Dict[str, float]] = {}
        
        # Initialize default allocations
        self._initialize_default_allocations()
        
    def _initialize_default_allocations(self):
        """Initialize default strategy allocations."""
        self.strategy_allocations = {
            StrategyType.MOMENTUM: StrategyAllocation(
                strategy_type=StrategyType.MOMENTUM,
                weight=0.25,
                parameters={"lookback_period": 20, "threshold": 0.02},
                expected_return=0.015,
                risk_score=0.6
            ),
            StrategyType.TECHNICAL: StrategyAllocation(
                strategy_type=StrategyType.TECHNICAL,
                weight=0.3,
                parameters={"rsi_period": 14, "bb_period": 20},
                expected_return=0.012,
                risk_score=0.5
            ),
            StrategyType.RISK_AWARE: StrategyAllocation(
                strategy_type=StrategyType.RISK_AWARE,
                weight=0.25,
                parameters={"max_drawdown": 0.1, "var_confidence": 0.95},
                expected_return=0.008,
                risk_score=0.3
            ),
            StrategyType.FUNDING_RATE: StrategyAllocation(
                strategy_type=StrategyType.FUNDING_RATE,
                weight=0.2,
                parameters={"symbol": "BTCUSDT", "timeframe": "1h"},
                expected_return=0.010,
                risk_score=0.4
            )
        }
    
    async def analyze_market_regime(self, symbol: str, timeframe: str = "1h") -> MarketRegime:
        """
        Analyze current market regime to optimize strategy selection.
        """
        try:
            # Get market data from cache
            cache_key = f"market_data:{symbol}:{timeframe}"
            market_data = await self.cache.get(cache_key)
            
            if not market_data:
                self.logger.warning(f"No market data available for regime analysis: {symbol}")
                return MarketRegime.UNCERTAIN
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(market_data)
            
            if len(df) < 50:
                return MarketRegime.UNCERTAIN
            
            # Calculate regime indicators
            regime_indicators = await self._calculate_regime_indicators(df)
            
            # Determine regime based on indicators
            regime = await self._classify_regime(regime_indicators)
            
            # Cache the regime
            regime_cache_key = f"market_regime:{symbol}:{timeframe}"
            await self.cache.set(regime_cache_key, regime.value, ttl=1800)  # 30 minutes
            
            self.current_regime = regime
            self.logger.info(f"Market regime for {symbol}: {regime.value}")
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return MarketRegime.UNCERTAIN
    
    async def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators for regime classification."""
        indicators = {}
        
        # Price trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Trend strength
        indicators['trend_strength'] = (df['close'].iloc[-1] - df['sma_50'].iloc[-1]) / df['sma_50'].iloc[-1]
        indicators['short_vs_long_ma'] = (df['sma_20'].iloc[-1] - df['sma_50'].iloc[-1]) / df['sma_50'].iloc[-1]
        
        # Volatility indicators
        df['returns'] = df['close'].pct_change()
        indicators['volatility'] = df['returns'].rolling(20).std().iloc[-1]
        indicators['volatility_percentile'] = (df['returns'].rolling(20).std().iloc[-1] > 
                                             df['returns'].rolling(20).std().quantile(0.8))
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        indicators['rsi'] = df['rsi'].iloc[-1]
        
        # Volume trend (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            indicators['volume_trend'] = df['volume'].iloc[-1] / df['volume_ma'].iloc[-1]
        else:
            indicators['volume_trend'] = 1.0
        
        # Price range indicators
        indicators['price_range'] = (df['high'].rolling(20).max().iloc[-1] - 
                                   df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1]
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _classify_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """Classify market regime based on indicators."""
        
        # Extract key indicators
        trend_strength = indicators.get('trend_strength', 0)
        volatility = indicators.get('volatility', 0)
        rsi = indicators.get('rsi', 50)
        volume_trend = indicators.get('volume_trend', 1)
        
        # High volatility regime
        if volatility > 0.03:  # 3% daily volatility
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if volatility < 0.01:  # 1% daily volatility
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if abs(trend_strength) > 0.05:  # 5% trend
            if trend_strength > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Sideways regime
        if abs(trend_strength) < 0.02 and 30 < rsi < 70:
            return MarketRegime.SIDEWAYS
        
        return MarketRegime.UNCERTAIN
    
    async def optimize_strategy_allocation(self, regime: MarketRegime) -> None:
        """Optimize strategy allocation based on market regime."""
        
        # Regime-based allocation templates
        regime_allocations = {
            MarketRegime.TRENDING_UP: {
                StrategyType.MOMENTUM: 0.4,
                StrategyType.TECHNICAL: 0.25,
                StrategyType.RISK_AWARE: 0.2,
                StrategyType.FUNDING_RATE: 0.15
            },
            MarketRegime.TRENDING_DOWN: {
                StrategyType.RISK_AWARE: 0.4,
                StrategyType.TECHNICAL: 0.25,
                StrategyType.MOMENTUM: 0.15,
                StrategyType.FUNDING_RATE: 0.2
            },
            MarketRegime.SIDEWAYS: {
                StrategyType.TECHNICAL: 0.35,
                StrategyType.FUNDING_RATE: 0.3,
                StrategyType.RISK_AWARE: 0.2,
                StrategyType.MOMENTUM: 0.15
            },
            MarketRegime.HIGH_VOLATILITY: {
                StrategyType.RISK_AWARE: 0.45,
                StrategyType.FUNDING_RATE: 0.25,
                StrategyType.TECHNICAL: 0.2,
                StrategyType.MOMENTUM: 0.1
            },
            MarketRegime.LOW_VOLATILITY: {
                StrategyType.MOMENTUM: 0.35,
                StrategyType.TECHNICAL: 0.3,
                StrategyType.RISK_AWARE: 0.2,
                StrategyType.FUNDING_RATE: 0.15
            }
        }
        
        # Get allocation for current regime
        target_allocation = regime_allocations.get(regime, {
            StrategyType.MOMENTUM: 0.25,
            StrategyType.TECHNICAL: 0.25,
            StrategyType.RISK_AWARE: 0.25,
            StrategyType.FUNDING_RATE: 0.25
        })
        
        # Update strategy weights
        for strategy_type, weight in target_allocation.items():
            if strategy_type in self.strategy_allocations:
                self.strategy_allocations[strategy_type].weight = weight
        
        self.logger.info(f"Optimized strategy allocation for regime {regime.value}: {target_allocation}")
    
    async def generate_trading_decision(self, symbol: str, timeframe: str = "1h") -> Optional[TradingDecision]:
        """
        Generate a comprehensive trading decision by combining all intelligence sources.
        """
        try:
            # 1. Analyze market regime
            regime = await self.analyze_market_regime(symbol, timeframe)
            
            # 2. Optimize strategy allocation
            await self.optimize_strategy_allocation(regime)
            
            # 3. Collect signals from all strategies
            strategy_signals = await self._collect_strategy_signals(symbol, timeframe)
            
            # 4. Get fused signal from signal fusion engine
            fused_signal = await self.signal_fusion.fuse_signals(symbol, timeframe)
            
            if not fused_signal and not strategy_signals:
                self.logger.warning(f"No signals available for {symbol}")
                return None
            
            # 5. Combine strategy and fused signals
            final_decision = await self._combine_signals_into_decision(
                symbol, strategy_signals, fused_signal, regime, timeframe
            )
            
            # 6. Apply risk management
            final_decision = await self._apply_risk_management(final_decision)
            
            # 7. Cache the decision
            cache_key = f"trading_decision:{symbol}:{timeframe}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, final_decision.to_dict(), ttl=1800)  # 30 minutes
            
            self.logger.info(f"Generated trading decision for {symbol}: {final_decision.action.value} "
                           f"(confidence: {final_decision.confidence:.2f})")
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error generating trading decision for {symbol}: {e}")
            raise TradingError(f"Failed to generate trading decision: {e}")
    
    async def _collect_strategy_signals(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Collect signals from all active strategies."""
        strategy_signals = []
        
        for strategy_type, allocation in self.strategy_allocations.items():
            if not allocation.active or allocation.weight <= 0:
                continue
            
            try:
                strategy = self.strategies.get(strategy_type)
                if strategy:
                    # Get market data for strategy
                    cache_key = f"market_data:{symbol}:{timeframe}"
                    market_data = await self.cache.get(cache_key)
                    
                    if market_data:
                        # Generate signal from strategy
                        signal_data = await strategy.generate_signal(market_data, allocation.parameters)
                        
                        if signal_data:
                            # Convert to TradingSignal
                            signal = TradingSignal(
                                agent_id=f"strategy_{strategy_type.value}",
                                symbol=symbol,
                                signal_type=SignalType(signal_data['action']),
                                confidence=signal_data['confidence'],
                                strength=allocation.weight,
                                timestamp=datetime.utcnow(),
                                timeframe=timeframe,
                                features=signal_data.get('features', {}),
                                metadata={
                                    'strategy_type': strategy_type.value,
                                    'parameters': allocation.parameters
                                }
                            )
                            
                            strategy_signals.append(signal)
                            
                            # Add to signal fusion engine
                            await self.signal_fusion.add_signal(signal)
                
            except Exception as e:
                self.logger.error(f"Error collecting signal from {strategy_type.value}: {e}")
                continue
        
        return strategy_signals
    
    async def _combine_signals_into_decision(self, symbol: str, strategy_signals: List[TradingSignal],
                                           fused_signal: Optional[FusedSignal], regime: MarketRegime,
                                           timeframe: str) -> TradingDecision:
        """Combine all signals into a final trading decision."""
        
        # Determine primary action
        if fused_signal:
            primary_action = fused_signal.signal_type
            base_confidence = fused_signal.confidence
            base_strength = fused_signal.strength
        elif strategy_signals:
            # Fallback to strategy consensus
            action_votes = {}
            total_weight = 0
            
            for signal in strategy_signals:
                weight = self.strategy_allocations.get(
                    StrategyType(signal.metadata['strategy_type']), 
                    StrategyAllocation(StrategyType.TECHNICAL, 0.1, {}, 0, 0)
                ).weight
                
                action_votes[signal.signal_type] = action_votes.get(signal.signal_type, 0) + weight
                total_weight += weight
            
            primary_action = max(action_votes.keys(), key=lambda k: action_votes[k])
            base_confidence = action_votes[primary_action] / total_weight if total_weight > 0 else 0.5
            base_strength = base_confidence
        else:
            primary_action = SignalType.HOLD
            base_confidence = 0.5
            base_strength = 0.5
        
        # Calculate position size based on confidence and risk
        position_size = await self._calculate_position_size(base_confidence, base_strength, regime)
        
        # Calculate risk metrics
        risk_metrics = await self._calculate_risk_metrics(symbol, primary_action, position_size)
        
        # Determine entry levels
        entry_price, stop_loss, take_profit = await self._calculate_entry_levels(
            symbol, primary_action, base_confidence
        )
        
        # Determine time horizon
        time_horizon = await self._determine_time_horizon(base_confidence, regime, timeframe)
        
        # Create strategy allocation summary
        strategy_allocation = {
            strategy_type: allocation.weight 
            for strategy_type, allocation in self.strategy_allocations.items()
        }
        
        return TradingDecision(
            symbol=symbol,
            action=primary_action,
            confidence=base_confidence,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_horizon=time_horizon,
            strategy_allocation=strategy_allocation,
            risk_metrics=risk_metrics,
            timestamp=datetime.utcnow()
        )
    
    async def _calculate_position_size(self, confidence: float, strength: float, regime: MarketRegime) -> float:
        """Calculate optimal position size based on confidence and regime."""
        
        # Base position size from confidence
        base_size = confidence * strength
        
        # Regime adjustments
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.UNCERTAIN: 0.5
        }
        
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        # Final position size (as fraction of available capital)
        position_size = min(0.25, max(0.01, base_size * regime_multiplier))  # 1% to 25%
        
        return position_size
    
    async def _calculate_risk_metrics(self, symbol: str, action: SignalType, position_size: float) -> Dict[str, float]:
        """Calculate risk metrics for the trading decision."""
        
        # Get volatility estimate
        cache_key = f"market_data:{symbol}:1h"
        market_data = await self.cache.get(cache_key)
        
        volatility = 0.02  # Default 2% daily volatility
        if market_data:
            df = pd.DataFrame(market_data)
            if len(df) > 20:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # Annualized hourly volatility
        
        # Calculate risk metrics
        var_95 = position_size * volatility * 1.645  # 95% VaR
        expected_loss = position_size * volatility * 0.5  # Expected daily loss
        
        return {
            "var_95": var_95,
            "expected_loss": expected_loss,
            "volatility": volatility,
            "position_size": position_size,
            "leverage": 1.0  # No leverage for now
        }
    
    async def _calculate_entry_levels(self, symbol: str, action: SignalType, 
                                    confidence: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, stop loss, and take profit levels."""
        
        # Get current price
        cache_key = f"market_data:{symbol}:1m"
        market_data = await self.cache.get(cache_key)
        
        if not market_data:
            return None, None, None
        
        current_price = market_data[-1]['close'] if isinstance(market_data, list) else market_data.get('close')
        if not current_price:
            return None, None, None
        
        # Calculate levels based on action and confidence
        if action in [SignalType.BUY, SignalType.STRONG_BUY]:
            # For buy signals
            entry_price = current_price * (1 + 0.001)  # Slightly above current price
            stop_loss = current_price * (1 - 0.02 / confidence)  # Tighter stops for higher confidence
            take_profit = current_price * (1 + 0.04 * confidence)  # Higher targets for higher confidence
            
        elif action in [SignalType.SELL, SignalType.STRONG_SELL]:
            # For sell signals
            entry_price = current_price * (1 - 0.001)  # Slightly below current price
            stop_loss = current_price * (1 + 0.02 / confidence)
            take_profit = current_price * (1 - 0.04 * confidence)
            
        else:  # HOLD
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        return entry_price, stop_loss, take_profit
    
    async def _determine_time_horizon(self, confidence: float, regime: MarketRegime, timeframe: str) -> timedelta:
        """Determine optimal time horizon for the trade."""
        
        # Base time horizons by timeframe
        base_horizons = {
            "1m": timedelta(minutes=15),
            "5m": timedelta(hours=1),
            "15m": timedelta(hours=4),
            "1h": timedelta(hours=12),
            "4h": timedelta(days=2),
            "1d": timedelta(days=7)
        }
        
        base_horizon = base_horizons.get(timeframe, timedelta(hours=12))
        
        # Adjust based on confidence (higher confidence = longer holding)
        confidence_multiplier = 0.5 + confidence
        
        # Adjust based on regime
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.5,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.LOW_VOLATILITY: 1.3,
            MarketRegime.UNCERTAIN: 0.5
        }
        
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        # Calculate final time horizon
        final_seconds = base_horizon.total_seconds() * confidence_multiplier * regime_multiplier
        return timedelta(seconds=final_seconds)
    
    async def _apply_risk_management(self, decision: TradingDecision) -> TradingDecision:
        """Apply final risk management checks and adjustments."""
        
        # Check position size limits
        max_position_size = 0.2  # 20% max position
        if decision.position_size > max_position_size:
            decision.position_size = max_position_size
            self.logger.warning(f"Position size capped at {max_position_size} for {decision.symbol}")
        
        # Check confidence thresholds
        min_confidence = 0.55  # Minimum 55% confidence for trades
        if decision.confidence < min_confidence and decision.action != SignalType.HOLD:
            decision.action = SignalType.HOLD
            decision.position_size = 0.0
            self.logger.warning(f"Trade blocked due to low confidence: {decision.confidence}")
        
        # Check risk metrics
        max_var = 0.05  # Maximum 5% VaR
        if decision.risk_metrics.get("var_95", 0) > max_var:
            decision.position_size *= 0.5  # Reduce position size
            self.logger.warning(f"Position size reduced due to high VaR for {decision.symbol}")
        
        return decision
    
    async def update_strategy_performance(self, strategy_type: StrategyType, symbol: str,
                                        actual_return: float, predicted_return: float) -> None:
        """Update strategy performance tracking."""
        
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = {
                "total_trades": 0,
                "successful_trades": 0,
                "average_return": 0.0,
                "sharpe_ratio": 0.0,
                "last_updated": datetime.utcnow().isoformat()
            }
        
        perf = self.strategy_performance[strategy_type]
        
        # Update metrics
        perf["total_trades"] += 1
        
        # Check if trade was successful
        if actual_return > 0:
            perf["successful_trades"] += 1
        
        # Update average return with exponential moving average
        alpha = 0.1
        perf["average_return"] = alpha * actual_return + (1 - alpha) * perf["average_return"]
        perf["last_updated"] = datetime.utcnow().isoformat()
        
        # Adjust strategy weight based on performance
        success_rate = perf["successful_trades"] / perf["total_trades"]
        if success_rate > 0.6:  # Good performance
            current_weight = self.strategy_allocations[strategy_type].weight
            self.strategy_allocations[strategy_type].weight = min(0.5, current_weight * 1.05)
        elif success_rate < 0.4:  # Poor performance
            current_weight = self.strategy_allocations[strategy_type].weight
            self.strategy_allocations[strategy_type].weight = max(0.1, current_weight * 0.95)
        
        self.logger.info(f"Updated performance for {strategy_type.value}: "
                        f"success_rate={success_rate:.2f}, "
                        f"weight={self.strategy_allocations[strategy_type].weight:.3f}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the Strategy Agent."""
        
        # Get fusion engine statistics
        fusion_stats = await self.signal_fusion.get_fusion_statistics()
        
        return {
            "agent_id": "M4_strategy_agent",
            "current_regime": self.current_regime.value,
            "strategy_allocations": {
                k.value: {
                    "weight": v.weight,
                    "active": v.active,
                    "expected_return": v.expected_return,
                    "risk_score": v.risk_score
                }
                for k, v in self.strategy_allocations.items()
            },
            "strategy_performance": self.strategy_performance,
            "fusion_engine_stats": fusion_stats,
            "last_updated": datetime.utcnow().isoformat()
        } 