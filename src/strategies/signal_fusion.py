"""
Signal Fusion Engine - Core of M4 Strategy Agent
Combines signals from multiple intelligence agents into unified trading decisions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor
import joblib

from ..core.cache import TradingCache
from ..core.exceptions import TradingError, MLModelError


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConfidenceLevel(Enum):
    """Signal confidence levels."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class TradingSignal:
    """Individual trading signal from an agent."""
    agent_id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    timeframe: str  # "1m", "5m", "1h", "1d"
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")


@dataclass
class FusedSignal:
    """Final fused trading signal with combined intelligence."""
    symbol: str
    signal_type: SignalType
    confidence: float
    strength: float
    timestamp: datetime
    contributing_signals: List[TradingSignal]
    fusion_weights: Dict[str, float]
    risk_score: float
    expected_return: float
    time_horizon: timedelta
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "timestamp": self.timestamp.isoformat(),
            "risk_score": self.risk_score,
            "expected_return": self.expected_return,
            "time_horizon_minutes": self.time_horizon.total_seconds() / 60,
            "contributing_agents": [s.agent_id for s in self.contributing_signals],
            "fusion_weights": self.fusion_weights
        }


class SignalFusionEngine:
    """
    Advanced signal fusion engine that combines predictions from multiple AI agents.
    Uses ensemble methods, confidence weighting, and temporal consistency.
    """
    
    def __init__(self, cache: TradingCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
        # Agent weights and performance tracking
        self.agent_weights: Dict[str, float] = {
            "M5_deep_learning": 0.25,
            "M7_price_prediction": 0.25,
            "M9_sentiment": 0.20,
            "technical_analysis": 0.15,
            "fundamental_analysis": 0.10,
            "risk_adjusted": 0.05
        }
        
        # Performance tracking for dynamic weight adjustment
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
        # Signal history for temporal consistency
        self.signal_history: Dict[str, List[TradingSignal]] = {}
        
        # Fusion models
        self.classification_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        
    async def add_signal(self, signal: TradingSignal) -> None:
        """Add a new signal to the fusion engine."""
        try:
            # Store in cache
            cache_key = f"signal:{signal.agent_id}:{signal.symbol}:{signal.timestamp.isoformat()}"
            await self.cache.set(cache_key, signal.__dict__, ttl=3600)
            
            # Add to history
            if signal.symbol not in self.signal_history:
                self.signal_history[signal.symbol] = []
            
            self.signal_history[signal.symbol].append(signal)
            
            # Keep only recent signals (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.signal_history[signal.symbol] = [
                s for s in self.signal_history[signal.symbol] 
                if s.timestamp > cutoff_time
            ]
            
            self.logger.info(f"Added signal from {signal.agent_id} for {signal.symbol}: {signal.signal_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {e}")
            raise TradingError(f"Failed to add signal: {e}")
    
    async def fuse_signals(self, symbol: str, timeframe: str = "1h") -> Optional[FusedSignal]:
        """
        Fuse all available signals for a symbol into a unified decision.
        """
        try:
            # Get recent signals for the symbol
            recent_signals = await self._get_recent_signals(symbol, timeframe)
            
            if not recent_signals:
                self.logger.warning(f"No signals available for {symbol}")
                return None
            
            # Perform signal fusion
            fused_signal = await self._perform_fusion(recent_signals, symbol, timeframe)
            
            # Cache the result
            cache_key = f"fused_signal:{symbol}:{timeframe}:{datetime.utcnow().isoformat()}"
            await self.cache.set(cache_key, fused_signal.to_dict(), ttl=300)  # 5 minutes
            
            return fused_signal
            
        except Exception as e:
            self.logger.error(f"Error fusing signals for {symbol}: {e}")
            raise TradingError(f"Signal fusion failed: {e}")
    
    async def _get_recent_signals(self, symbol: str, timeframe: str) -> List[TradingSignal]:
        """Get recent signals for a symbol within the specified timeframe."""
        if symbol not in self.signal_history:
            return []
        
        # Define timeframe windows
        timeframe_windows = {
            "1m": timedelta(minutes=5),
            "5m": timedelta(minutes=15),
            "15m": timedelta(minutes=30),
            "1h": timedelta(hours=2),
            "4h": timedelta(hours=8),
            "1d": timedelta(days=2)
        }
        
        window = timeframe_windows.get(timeframe, timedelta(hours=1))
        cutoff_time = datetime.utcnow() - window
        
        recent_signals = [
            signal for signal in self.signal_history[symbol]
            if signal.timestamp > cutoff_time and signal.timeframe == timeframe
        ]
        
        return recent_signals
    
    async def _perform_fusion(self, signals: List[TradingSignal], symbol: str, timeframe: str) -> FusedSignal:
        """Perform the actual signal fusion using multiple methods."""
        
        # 1. Confidence-weighted voting
        weighted_signals = await self._confidence_weighted_fusion(signals)
        
        # 2. Temporal consistency check
        temporal_weight = await self._check_temporal_consistency(signals, symbol)
        
        # 3. Cross-validation with ensemble models
        ensemble_prediction = await self._ensemble_prediction(signals)
        
        # 4. Risk assessment
        risk_score = await self._calculate_risk_score(signals, symbol)
        
        # 5. Combine all methods
        final_signal_type, final_confidence, final_strength = await self._combine_fusion_methods(
            weighted_signals, ensemble_prediction, temporal_weight
        )
        
        # 6. Calculate expected return
        expected_return = await self._calculate_expected_return(signals, final_signal_type, final_confidence)
        
        # 7. Determine time horizon
        time_horizon = await self._determine_time_horizon(signals, timeframe)
        
        # Create fusion weights
        fusion_weights = {signal.agent_id: self.agent_weights.get(signal.agent_id, 0.1) for signal in signals}
        total_weight = sum(fusion_weights.values())
        fusion_weights = {k: v/total_weight for k, v in fusion_weights.items()}
        
        return FusedSignal(
            symbol=symbol,
            signal_type=final_signal_type,
            confidence=final_confidence,
            strength=final_strength,
            timestamp=datetime.utcnow(),
            contributing_signals=signals,
            fusion_weights=fusion_weights,
            risk_score=risk_score,
            expected_return=expected_return,
            time_horizon=time_horizon
        )
    
    async def _confidence_weighted_fusion(self, signals: List[TradingSignal]) -> Dict[SignalType, float]:
        """Perform confidence-weighted signal fusion."""
        signal_scores = {signal_type: 0.0 for signal_type in SignalType}
        total_weight = 0.0
        
        for signal in signals:
            # Calculate weight based on confidence and agent performance
            agent_weight = self.agent_weights.get(signal.agent_id, 0.1)
            confidence_weight = signal.confidence
            strength_weight = signal.strength
            
            combined_weight = agent_weight * confidence_weight * strength_weight
            
            signal_scores[signal.signal_type] += combined_weight
            total_weight += combined_weight
        
        # Normalize scores
        if total_weight > 0:
            signal_scores = {k: v/total_weight for k, v in signal_scores.items()}
        
        return signal_scores
    
    async def _check_temporal_consistency(self, signals: List[TradingSignal], symbol: str) -> float:
        """Check temporal consistency of signals to avoid whipsawing."""
        if len(signals) < 2:
            return 1.0
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.timestamp)
        
        # Check for signal consistency over time
        consistency_score = 0.0
        total_comparisons = 0
        
        for i in range(1, len(sorted_signals)):
            prev_signal = sorted_signals[i-1]
            curr_signal = sorted_signals[i]
            
            # Calculate time decay factor
            time_diff = (curr_signal.timestamp - prev_signal.timestamp).total_seconds() / 3600  # hours
            time_decay = np.exp(-time_diff / 24)  # Decay over 24 hours
            
            # Check signal agreement
            if prev_signal.signal_type == curr_signal.signal_type:
                consistency_score += time_decay
            elif self._are_signals_compatible(prev_signal.signal_type, curr_signal.signal_type):
                consistency_score += 0.5 * time_decay
            
            total_comparisons += time_decay
        
        return consistency_score / total_comparisons if total_comparisons > 0 else 0.5
    
    def _are_signals_compatible(self, signal1: SignalType, signal2: SignalType) -> bool:
        """Check if two signals are compatible (not contradictory)."""
        compatible_pairs = [
            (SignalType.BUY, SignalType.STRONG_BUY),
            (SignalType.SELL, SignalType.STRONG_SELL),
            (SignalType.HOLD, SignalType.BUY),
            (SignalType.HOLD, SignalType.SELL)
        ]
        
        return (signal1, signal2) in compatible_pairs or (signal2, signal1) in compatible_pairs
    
    async def _ensemble_prediction(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Use ensemble models for additional prediction validation."""
        if not signals:
            return {"confidence": 0.5, "strength": 0.5}
        
        try:
            # Prepare features from signals
            features = await self._extract_features_from_signals(signals)
            
            if features is None or len(features) == 0:
                return {"confidence": 0.5, "strength": 0.5}
            
            # Use pre-trained ensemble models if available
            if self.classification_model and self.regression_model:
                # Predict signal type (classification)
                signal_proba = self.classification_model.predict_proba([features])[0]
                
                # Predict confidence (regression)
                confidence_pred = self.regression_model.predict([features])[0]
                
                return {
                    "confidence": max(0.1, min(0.9, confidence_pred)),
                    "signal_probabilities": signal_proba.tolist()
                }
            
            # Fallback: simple ensemble
            return await self._simple_ensemble(signals)
            
        except Exception as e:
            self.logger.warning(f"Ensemble prediction failed: {e}")
            return {"confidence": 0.5, "strength": 0.5}
    
    async def _extract_features_from_signals(self, signals: List[TradingSignal]) -> Optional[List[float]]:
        """Extract numerical features from signals for ML models."""
        if not signals:
            return None
        
        features = []
        
        # Agent distribution
        agent_counts = {}
        for signal in signals:
            agent_counts[signal.agent_id] = agent_counts.get(signal.agent_id, 0) + 1
        
        # Signal type distribution
        signal_type_counts = {}
        for signal in signals:
            signal_type_counts[signal.signal_type] = signal_type_counts.get(signal.signal_type, 0) + 1
        
        # Aggregate statistics
        confidences = [s.confidence for s in signals]
        strengths = [s.strength for s in signals]
        
        features.extend([
            len(signals),  # Number of signals
            np.mean(confidences),  # Average confidence
            np.std(confidences),   # Confidence std
            np.mean(strengths),    # Average strength
            np.std(strengths),     # Strength std
            signal_type_counts.get(SignalType.BUY, 0),
            signal_type_counts.get(SignalType.SELL, 0),
            signal_type_counts.get(SignalType.HOLD, 0),
        ])
        
        return features
    
    async def _simple_ensemble(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Simple ensemble when ML models are not available."""
        if not signals:
            return {"confidence": 0.5, "strength": 0.5}
        
        # Weighted average of confidences and strengths
        total_weight = 0.0
        weighted_confidence = 0.0
        weighted_strength = 0.0
        
        for signal in signals:
            weight = self.agent_weights.get(signal.agent_id, 0.1)
            weighted_confidence += signal.confidence * weight
            weighted_strength += signal.strength * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_confidence /= total_weight
            weighted_strength /= total_weight
        
        return {
            "confidence": weighted_confidence,
            "strength": weighted_strength
        }
    
    async def _combine_fusion_methods(self, weighted_signals: Dict[SignalType, float], 
                                   ensemble_prediction: Dict[str, float],
                                   temporal_weight: float) -> Tuple[SignalType, float, float]:
        """Combine all fusion methods into final decision."""
        
        # Get the strongest signal from weighted voting
        best_signal_type = max(weighted_signals.keys(), key=lambda k: weighted_signals[k])
        best_signal_score = weighted_signals[best_signal_type]
        
        # Adjust confidence based on ensemble prediction and temporal consistency
        base_confidence = ensemble_prediction.get("confidence", 0.5)
        temporal_adjustment = temporal_weight * 0.2  # Up to 20% adjustment
        
        final_confidence = min(0.95, max(0.05, base_confidence + temporal_adjustment))
        
        # Calculate final strength
        final_strength = min(0.95, max(0.05, best_signal_score * temporal_weight))
        
        return best_signal_type, final_confidence, final_strength
    
    async def _calculate_risk_score(self, signals: List[TradingSignal], symbol: str) -> float:
        """Calculate risk score for the fused signal."""
        if not signals:
            return 0.5
        
        # Factors contributing to risk
        signal_divergence = await self._calculate_signal_divergence(signals)
        confidence_variance = np.var([s.confidence for s in signals])
        agent_diversity = len(set(s.agent_id for s in signals)) / len(self.agent_weights)
        
        # Risk score (0 = low risk, 1 = high risk)
        risk_score = (
            signal_divergence * 0.4 +
            confidence_variance * 0.3 +
            (1 - agent_diversity) * 0.3
        )
        
        return min(0.95, max(0.05, risk_score))
    
    async def _calculate_signal_divergence(self, signals: List[TradingSignal]) -> float:
        """Calculate how much signals diverge from each other."""
        if len(signals) < 2:
            return 0.0
        
        signal_values = []
        for signal in signals:
            # Convert signal type to numerical value
            signal_map = {
                SignalType.STRONG_SELL: -2,
                SignalType.SELL: -1,
                SignalType.HOLD: 0,
                SignalType.BUY: 1,
                SignalType.STRONG_BUY: 2
            }
            signal_values.append(signal_map[signal.signal_type] * signal.strength)
        
        # Calculate variance as a measure of divergence
        return min(1.0, np.var(signal_values) / 2)  # Normalize to 0-1
    
    async def _calculate_expected_return(self, signals: List[TradingSignal], 
                                       signal_type: SignalType, confidence: float) -> float:
        """Calculate expected return based on signals and historical performance."""
        if not signals:
            return 0.0
        
        # Base expected return based on signal type
        base_returns = {
            SignalType.STRONG_BUY: 0.03,
            SignalType.BUY: 0.015,
            SignalType.HOLD: 0.0,
            SignalType.SELL: -0.015,
            SignalType.STRONG_SELL: -0.03
        }
        
        base_return = base_returns[signal_type]
        
        # Adjust based on confidence and signal strength
        confidence_adjustment = (confidence - 0.5) * 2  # -1 to 1
        avg_strength = np.mean([s.strength for s in signals])
        
        expected_return = base_return * confidence_adjustment * avg_strength
        
        return expected_return
    
    async def _determine_time_horizon(self, signals: List[TradingSignal], timeframe: str) -> timedelta:
        """Determine appropriate time horizon for the signal."""
        # Base time horizons by timeframe
        base_horizons = {
            "1m": timedelta(minutes=5),
            "5m": timedelta(minutes=15),
            "15m": timedelta(minutes=30),
            "1h": timedelta(hours=2),
            "4h": timedelta(hours=8),
            "1d": timedelta(days=2)
        }
        
        base_horizon = base_horizons.get(timeframe, timedelta(hours=1))
        
        # Adjust based on signal confidence
        if signals:
            avg_confidence = np.mean([s.confidence for s in signals])
            # Higher confidence = longer holding period
            confidence_multiplier = 0.5 + avg_confidence
            base_horizon = timedelta(seconds=base_horizon.total_seconds() * confidence_multiplier)
        
        return base_horizon
    
    async def update_agent_performance(self, agent_id: str, symbol: str, 
                                     actual_return: float, predicted_return: float) -> None:
        """Update agent performance tracking for dynamic weight adjustment."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "average_error": 0.0,
                "last_updated": datetime.utcnow().isoformat()
            }
        
        perf = self.agent_performance[agent_id]
        
        # Update metrics
        perf["total_predictions"] += 1
        
        # Check if prediction was correct (within 20% tolerance)
        error = abs(actual_return - predicted_return)
        if error <= 0.2 * abs(predicted_return) if predicted_return != 0 else error <= 0.01:
            perf["correct_predictions"] += 1
        
        # Update average error with exponential moving average
        alpha = 0.1  # Learning rate
        perf["average_error"] = alpha * error + (1 - alpha) * perf["average_error"]
        perf["last_updated"] = datetime.utcnow().isoformat()
        
        # Adjust agent weight based on performance
        accuracy = perf["correct_predictions"] / perf["total_predictions"]
        if accuracy > 0.7:  # Good performance
            self.agent_weights[agent_id] = min(0.4, self.agent_weights.get(agent_id, 0.1) * 1.1)
        elif accuracy < 0.3:  # Poor performance
            self.agent_weights[agent_id] = max(0.05, self.agent_weights.get(agent_id, 0.1) * 0.9)
        
        self.logger.info(f"Updated performance for {agent_id}: accuracy={accuracy:.2f}, weight={self.agent_weights[agent_id]:.3f}")
    
    async def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fusion engine."""
        total_signals = sum(len(signals) for signals in self.signal_history.values())
        
        return {
            "total_signals_processed": total_signals,
            "symbols_tracked": len(self.signal_history),
            "agent_weights": self.agent_weights.copy(),
            "agent_performance": self.agent_performance.copy(),
            "signals_by_symbol": {k: len(v) for k, v in self.signal_history.items()},
            "last_updated": datetime.utcnow().isoformat()
        } 