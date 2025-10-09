"""
Autonomous Trading Pods (ATP) System
Self-evolving AI trading agents with genetic algorithms and swarm intelligence

Features:
- Independent AI agents with unique strategies
- Genetic algorithm evolution of trading parameters
- Swarm intelligence for coordinated actions
- Performance-based natural selection
- Risk budget allocation and management
"""

import asyncio
import logging
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

class PodStrategy(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    PAIRS_TRADING = "pairs_trading"

class PodStatus(Enum):
    ACTIVE = "active"
    LEARNING = "learning"
    HIBERNATING = "hibernating"
    EVOLVING = "evolving"
    RETIRED = "retired"

@dataclass
class TradingGenes:
    """Genetic representation of trading strategy parameters"""
    risk_tolerance: float  # 0.0 - 1.0
    holding_period: int    # days
    position_size: float   # 0.0 - 1.0 of allocated capital
    stop_loss: float       # 0.0 - 0.2 (20% max loss)
    take_profit: float     # 1.1 - 3.0 (profit multiplier)
    momentum_threshold: float  # 0.0 - 1.0
    volume_sensitivity: float  # 0.0 - 1.0
    correlation_sensitivity: float  # 0.0 - 1.0
    news_weight: float     # 0.0 - 1.0
    technical_weight: float # 0.0 - 1.0
    
    def mutate(self, mutation_rate: float = 0.1) -> 'TradingGenes':
        """Create mutated version of genes"""
        def mutate_value(value: float, min_val: float, max_val: float) -> float:
            if random.random() < mutation_rate:
                mutation = random.gauss(0, 0.1)  # 10% std deviation
                return max(min_val, min(max_val, value + mutation))
            return value
        
        return TradingGenes(
            risk_tolerance=mutate_value(self.risk_tolerance, 0.0, 1.0),
            holding_period=max(1, int(self.holding_period + random.gauss(0, 2))),
            position_size=mutate_value(self.position_size, 0.01, 1.0),
            stop_loss=mutate_value(self.stop_loss, 0.01, 0.2),
            take_profit=mutate_value(self.take_profit, 1.1, 3.0),
            momentum_threshold=mutate_value(self.momentum_threshold, 0.0, 1.0),
            volume_sensitivity=mutate_value(self.volume_sensitivity, 0.0, 1.0),
            correlation_sensitivity=mutate_value(self.correlation_sensitivity, 0.0, 1.0),
            news_weight=mutate_value(self.news_weight, 0.0, 1.0),
            technical_weight=mutate_value(self.technical_weight, 0.0, 1.0)
        )
    
    @classmethod
    def crossover(cls, parent1: 'TradingGenes', parent2: 'TradingGenes') -> 'TradingGenes':
        """Create offspring from two parent gene sets"""
        return cls(
            risk_tolerance=(parent1.risk_tolerance + parent2.risk_tolerance) / 2,
            holding_period=random.choice([parent1.holding_period, parent2.holding_period]),
            position_size=(parent1.position_size + parent2.position_size) / 2,
            stop_loss=(parent1.stop_loss + parent2.stop_loss) / 2,
            take_profit=(parent1.take_profit + parent2.take_profit) / 2,
            momentum_threshold=(parent1.momentum_threshold + parent2.momentum_threshold) / 2,
            volume_sensitivity=random.choice([parent1.volume_sensitivity, parent2.volume_sensitivity]),
            correlation_sensitivity=(parent1.correlation_sensitivity + parent2.correlation_sensitivity) / 2,
            news_weight=random.choice([parent1.news_weight, parent2.news_weight]),
            technical_weight=(parent1.technical_weight + parent2.technical_weight) / 2
        )
    
    @classmethod
    def random(cls) -> 'TradingGenes':
        """Generate random genes"""
        return cls(
            risk_tolerance=random.uniform(0.1, 0.8),
            holding_period=random.randint(1, 30),
            position_size=random.uniform(0.01, 0.5),
            stop_loss=random.uniform(0.02, 0.15),
            take_profit=random.uniform(1.2, 2.5),
            momentum_threshold=random.uniform(0.1, 0.9),
            volume_sensitivity=random.uniform(0.1, 0.9),
            correlation_sensitivity=random.uniform(0.1, 0.9),
            news_weight=random.uniform(0.1, 0.9),
            technical_weight=random.uniform(0.1, 0.9)
        )

@dataclass
class PodPerformance:
    """Performance metrics for a trading pod"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_executed: int = 0
    avg_holding_period: float = 0.0
    fitness_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SwarmMessage:
    """Communication message between pods"""
    sender_id: str
    message_type: str  # "signal", "warning", "opportunity", "coordination"
    content: Dict[str, Any]
    timestamp: datetime
    priority: int  # 1-10, higher = more important

class TradingPod:
    """Individual autonomous trading agent"""
    
    def __init__(
        self,
        pod_id: str,
        strategy: PodStrategy,
        genes: TradingGenes,
        initial_capital: float = 10000.0
    ):
        self.pod_id = pod_id
        self.strategy = strategy
        self.genes = genes
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.performance = PodPerformance()
        self.status = PodStatus.LEARNING
        self.generation = 1
        self.age = 0  # trading days
        self.trades_history: List[Dict] = []
        self.message_queue: List[SwarmMessage] = []
        self.last_evolution = datetime.now()
        
        # Strategy-specific parameters
        self.strategy_params = self._initialize_strategy_params()
        
        # Learning and adaptation
        self.learning_rate = 0.01
        self.confidence = 0.5
        self.stress_level = 0.0
    
    def _initialize_strategy_params(self) -> Dict[str, Any]:
        """Initialize strategy-specific parameters"""
        if self.strategy == PodStrategy.MOMENTUM:
            return {
                "lookback_period": int(5 + self.genes.momentum_threshold * 15),
                "momentum_factor": self.genes.momentum_threshold,
                "volume_threshold": self.genes.volume_sensitivity
            }
        elif self.strategy == PodStrategy.MEAN_REVERSION:
            return {
                "reversion_period": int(10 + (1 - self.genes.momentum_threshold) * 20),
                "z_score_threshold": 1.5 + self.genes.risk_tolerance,
                "correlation_period": int(20 + self.genes.correlation_sensitivity * 30)
            }
        elif self.strategy == PodStrategy.VOLATILITY:
            return {
                "volatility_window": int(10 + self.genes.risk_tolerance * 20),
                "volatility_threshold": 0.2 + self.genes.risk_tolerance * 0.3,
                "gamma_exposure": self.genes.risk_tolerance
            }
        else:
            return {
                "default_param": 1.0,
                "adaptation_rate": self.genes.risk_tolerance
            }
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Process market data and generate trading signals"""
        try:
            if self.status != PodStatus.ACTIVE:
                return None
            
            # Strategy-specific signal generation
            signal = await self._generate_trading_signal(market_data)
            
            if signal:
                # Apply risk management
                signal = self._apply_risk_management(signal, market_data)
                
                # Log decision
                await self._log_decision(signal, market_data)
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Pod {self.pod_id} error processing market data: {e}")
            self.stress_level = min(1.0, self.stress_level + 0.1)
            return None
    
    async def _generate_trading_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Generate trading signal based on strategy"""
        symbol = market_data.get("symbol")
        price = market_data.get("price", 0)
        volume = market_data.get("volume", 0)
        
        if not symbol or not price:
            return None
        
        if self.strategy == PodStrategy.MOMENTUM:
            return await self._momentum_signal(market_data)
        elif self.strategy == PodStrategy.MEAN_REVERSION:
            return await self._mean_reversion_signal(market_data)
        elif self.strategy == PodStrategy.VOLATILITY:
            return await self._volatility_signal(market_data)
        elif self.strategy == PodStrategy.SENTIMENT:
            return await self._sentiment_signal(market_data)
        else:
            return await self._default_signal(market_data)
    
    async def _momentum_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Generate momentum-based trading signal"""
        symbol = market_data["symbol"]
        price = market_data["price"]
        
        # Simulate momentum calculation
        momentum_score = random.uniform(-1, 1) * self.genes.momentum_threshold
        
        if abs(momentum_score) > self.strategy_params["momentum_factor"]:
            side = "buy" if momentum_score > 0 else "sell"
            confidence = min(0.95, abs(momentum_score) * self.confidence)
            
            return {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": self._calculate_position_size(price, confidence),
                "strategy": "momentum",
                "confidence": confidence,
                "reasoning": f"Momentum score: {momentum_score:.3f}"
            }
        
        return None
    
    async def _mean_reversion_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Generate mean reversion trading signal"""
        symbol = market_data["symbol"]
        price = market_data["price"]
        
        # Simulate z-score calculation
        z_score = random.gauss(0, 1.5)
        
        threshold = self.strategy_params["z_score_threshold"]
        
        if abs(z_score) > threshold:
            side = "sell" if z_score > 0 else "buy"  # Reversion logic
            confidence = min(0.95, abs(z_score) / 3.0)
            
            return {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": self._calculate_position_size(price, confidence),
                "strategy": "mean_reversion",
                "confidence": confidence,
                "reasoning": f"Z-score: {z_score:.3f}"
            }
        
        return None
    
    async def _volatility_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Generate volatility-based trading signal"""
        symbol = market_data["symbol"]
        price = market_data["price"]
        
        # Simulate volatility analysis
        volatility = random.uniform(0.1, 0.8)
        
        if volatility > self.strategy_params["volatility_threshold"]:
            # High volatility - potential options play
            side = "buy" if self.genes.risk_tolerance > 0.5 else "sell"
            confidence = min(0.9, volatility * self.confidence)
            
            return {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": self._calculate_position_size(price, confidence * 0.5),  # Reduced size for vol plays
                "strategy": "volatility",
                "confidence": confidence,
                "reasoning": f"High volatility: {volatility:.3f}"
            }
        
        return None
    
    async def _sentiment_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Generate sentiment-based trading signal"""
        symbol = market_data["symbol"]
        price = market_data["price"]
        
        # Simulate sentiment analysis
        sentiment = random.uniform(-1, 1) * self.genes.news_weight
        
        if abs(sentiment) > 0.3:
            side = "buy" if sentiment > 0 else "sell"
            confidence = min(0.8, abs(sentiment) * self.confidence)
            
            return {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": self._calculate_position_size(price, confidence),
                "strategy": "sentiment",
                "confidence": confidence,
                "reasoning": f"Sentiment: {sentiment:.3f}"
            }
        
        return None
    
    async def _default_signal(self, market_data: Dict[str, Any]) -> Optional[Dict]:
        """Default signal generation"""
        if random.random() < 0.1:  # 10% chance of random signal
            return {
                "symbol": market_data["symbol"],
                "side": random.choice(["buy", "sell"]),
                "price": market_data["price"],
                "quantity": self._calculate_position_size(market_data["price"], 0.3),
                "strategy": "random",
                "confidence": 0.3,
                "reasoning": "Random exploration"
            }
        return None
    
    def _calculate_position_size(self, price: float, confidence: float) -> int:
        """Calculate position size based on genes and confidence"""
        available_capital = self.capital * self.genes.position_size
        risk_adjusted_size = available_capital * confidence * self.genes.risk_tolerance
        
        max_shares = int(risk_adjusted_size / price)
        return max(1, max_shares)
    
    def _apply_risk_management(self, signal: Dict, market_data: Dict) -> Dict:
        """Apply risk management rules to trading signal"""
        # Position sizing limits
        current_position_value = sum(abs(pos * market_data.get("price", 100)) 
                                   for pos in self.positions.values())
        
        if current_position_value > self.capital * 0.8:  # 80% position limit
            signal["quantity"] = int(signal["quantity"] * 0.5)  # Reduce size
        
        # Stop loss and take profit
        signal["stop_loss"] = signal["price"] * (1 - self.genes.stop_loss) if signal["side"] == "buy" else signal["price"] * (1 + self.genes.stop_loss)
        signal["take_profit"] = signal["price"] * self.genes.take_profit if signal["side"] == "buy" else signal["price"] / self.genes.take_profit
        
        # Stress level adjustment
        if self.stress_level > 0.5:
            signal["quantity"] = int(signal["quantity"] * (1 - self.stress_level))
        
        return signal
    
    async def _log_decision(self, signal: Dict, market_data: Dict):
        """Log trading decision for learning"""
        trade_log = {
            "timestamp": datetime.now(),
            "signal": signal,
            "market_data": market_data,
            "confidence": self.confidence,
            "stress_level": self.stress_level,
            "genes_snapshot": vars(self.genes)
        }
        
        self.trades_history.append(trade_log)
        
        # Keep only recent history
        if len(self.trades_history) > 1000:
            self.trades_history = self.trades_history[-500:]
    
    def update_performance(self, trade_result: Dict):
        """Update performance metrics after trade execution"""
        try:
            pnl = trade_result.get("pnl", 0)
            self.capital += pnl
            
            # Update performance metrics
            self.performance.trades_executed += 1
            self.performance.total_return = (self.capital - self.initial_capital) / self.initial_capital
            
            # Update win rate
            if pnl > 0:
                wins = sum(1 for trade in self.trades_history[-100:] 
                          if trade.get("pnl", 0) > 0)
                self.performance.win_rate = wins / min(100, len(self.trades_history))
            
            # Calculate fitness score
            self.performance.fitness_score = self._calculate_fitness()
            
            # Adjust confidence based on recent performance
            recent_trades = self.trades_history[-10:]
            if recent_trades:
                recent_performance = sum(t.get("pnl", 0) for t in recent_trades) / len(recent_trades)
                if recent_performance > 0:
                    self.confidence = min(0.95, self.confidence * 1.01)
                else:
                    self.confidence = max(0.1, self.confidence * 0.99)
            
            # Reduce stress on successful trades
            if pnl > 0:
                self.stress_level = max(0, self.stress_level - 0.05)
            else:
                self.stress_level = min(1.0, self.stress_level + 0.02)
            
            self.age += 1
            
        except Exception as e:
            logger.error(f"Error updating performance for pod {self.pod_id}: {e}")
    
    def _calculate_fitness(self) -> float:
        """Calculate overall fitness score"""
        if self.performance.trades_executed < 5:
            return 0.5  # Neutral fitness for new pods
        
        # Weighted fitness calculation
        return_score = max(0, min(2, self.performance.total_return + 1))
        win_rate_score = self.performance.win_rate
        risk_score = max(0, 1 - abs(self.performance.max_drawdown))
        
        # Penalize high stress
        stress_penalty = 1 - self.stress_level * 0.5
        
        fitness = (return_score * 0.4 + win_rate_score * 0.3 + risk_score * 0.2 + stress_penalty * 0.1)
        
        return max(0, min(1, fitness))
    
    def should_evolve(self) -> bool:
        """Determine if pod should undergo evolution"""
        time_since_evolution = datetime.now() - self.last_evolution
        
        # Evolution triggers
        if time_since_evolution.days > 30:  # Monthly evolution
            return True
        
        if self.performance.fitness_score < 0.3 and self.age > 20:  # Poor performance
            return True
        
        if self.stress_level > 0.8:  # High stress
            return True
        
        return False
    
    def evolve(self, mutation_rate: float = 0.1):
        """Evolve pod's genes through mutation"""
        old_genes = self.genes
        self.genes = self.genes.mutate(mutation_rate)
        self.generation += 1
        self.last_evolution = datetime.now()
        self.status = PodStatus.EVOLVING
        
        # Reset some metrics
        self.stress_level *= 0.5
        self.confidence = (self.confidence + 0.5) / 2  # Move toward neutral
        
        logger.info(f"Pod {self.pod_id} evolved to generation {self.generation}")
    
    def receive_message(self, message: SwarmMessage):
        """Receive message from swarm"""
        self.message_queue.append(message)
        
        # Keep only recent messages
        if len(self.message_queue) > 50:
            self.message_queue = sorted(
                self.message_queue, 
                key=lambda x: (x.priority, x.timestamp)
            )[-25:]
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            "pod_id": self.pod_id,
            "strategy": self.strategy.value,
            "status": self.status.value,
            "generation": self.generation,
            "age": self.age,
            "capital": self.capital,
            "performance": {
                "total_return": self.performance.total_return,
                "win_rate": self.performance.win_rate,
                "trades_executed": self.performance.trades_executed,
                "fitness_score": self.performance.fitness_score
            },
            "health": {
                "confidence": self.confidence,
                "stress_level": self.stress_level
            },
            "genes": vars(self.genes),
            "active_positions": len(self.positions),
            "pending_messages": len(self.message_queue)
        }

class SwarmIntelligence:
    """Coordinates communication and collective behavior between pods"""
    
    def __init__(self):
        self.pods: Dict[str, TradingPod] = {}
        self.global_signals: List[Dict] = []
        self.coordination_rules: List[Callable] = []
        self.market_regime: str = "normal"  # normal, volatile, trending, ranging
        
    def add_pod(self, pod: TradingPod):
        """Add pod to swarm"""
        self.pods[pod.pod_id] = pod
        logger.info(f"Added pod {pod.pod_id} to swarm")
    
    def remove_pod(self, pod_id: str):
        """Remove pod from swarm"""
        if pod_id in self.pods:
            del self.pods[pod_id]
            logger.info(f"Removed pod {pod_id} from swarm")
    
    async def coordinate_pods(self, market_data: Dict[str, Any]) -> List[Dict]:
        """Coordinate pod activities and generate collective signals"""
        all_signals = []
        
        # Update market regime
        await self._update_market_regime(market_data)
        
        # Collect individual pod signals
        pod_signals = await asyncio.gather(*[
            pod.process_market_data(market_data) 
            for pod in self.pods.values() 
            if pod.status == PodStatus.ACTIVE
        ], return_exceptions=True)
        
        # Filter valid signals
        valid_signals = [s for s in pod_signals if isinstance(s, dict) and s]
        
        # Apply swarm intelligence
        coordinated_signals = await self._apply_swarm_intelligence(valid_signals, market_data)
        
        # Generate collective signals
        collective_signals = await self._generate_collective_signals(coordinated_signals, market_data)
        
        all_signals.extend(coordinated_signals)
        all_signals.extend(collective_signals)
        
        # Share information between pods
        await self._share_swarm_information(valid_signals)
        
        return all_signals
    
    async def _update_market_regime(self, market_data: Dict[str, Any]):
        """Update current market regime assessment"""
        # Simulate regime detection
        volatility = market_data.get("volatility", 0.2)
        volume = market_data.get("volume", 1000000)
        
        if volatility > 0.4:
            self.market_regime = "volatile"
        elif volatility < 0.1:
            self.market_regime = "ranging"
        else:
            self.market_regime = "normal"
    
    async def _apply_swarm_intelligence(
        self, 
        signals: List[Dict], 
        market_data: Dict[str, Any]
    ) -> List[Dict]:
        """Apply swarm intelligence to modify individual signals"""
        
        if not signals:
            return []
        
        coordinated_signals = []
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in signals:
            symbol = signal.get("symbol")
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            symbol_signals[symbol].append(signal)
        
        # Apply coordination for each symbol
        for symbol, symbol_signal_list in symbol_signals.items():
            coordinated = await self._coordinate_symbol_signals(symbol_signal_list, market_data)
            coordinated_signals.extend(coordinated)
        
        return coordinated_signals
    
    async def _coordinate_symbol_signals(
        self, 
        signals: List[Dict], 
        market_data: Dict[str, Any]
    ) -> List[Dict]:
        """Coordinate signals for a specific symbol"""
        
        if len(signals) <= 1:
            return signals
        
        # Calculate consensus
        buy_signals = [s for s in signals if s.get("side") == "buy"]
        sell_signals = [s for s in signals if s.get("side") == "sell"]
        
        # Apply swarm rules
        coordinated = []
        
        # Rule 1: Consensus amplification
        if len(buy_signals) >= len(sell_signals) * 2:  # Strong buy consensus
            # Amplify strongest buy signal
            best_buy = max(buy_signals, key=lambda x: x.get("confidence", 0))
            best_buy["quantity"] = int(best_buy["quantity"] * 1.5)
            best_buy["swarm_coordination"] = "consensus_amplification"
            coordinated.append(best_buy)
            
        elif len(sell_signals) >= len(buy_signals) * 2:  # Strong sell consensus
            # Amplify strongest sell signal
            best_sell = max(sell_signals, key=lambda x: x.get("confidence", 0))
            best_sell["quantity"] = int(best_sell["quantity"] * 1.5)
            best_sell["swarm_coordination"] = "consensus_amplification"
            coordinated.append(best_sell)
            
        # Rule 2: Conflict resolution
        elif len(buy_signals) == len(sell_signals):
            # High-confidence signal wins
            all_signals = buy_signals + sell_signals
            highest_confidence = max(all_signals, key=lambda x: x.get("confidence", 0))
            if highest_confidence.get("confidence", 0) > 0.7:
                highest_confidence["quantity"] = int(highest_confidence["quantity"] * 0.7)
                highest_confidence["swarm_coordination"] = "conflict_resolution"
                coordinated.append(highest_confidence)
        
        # Rule 3: Portfolio diversification
        else:
            # Include top signals with reduced size
            all_signals = sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)
            for signal in all_signals[:2]:  # Top 2 signals
                signal["quantity"] = int(signal["quantity"] * 0.8)
                signal["swarm_coordination"] = "diversification"
                coordinated.append(signal)
        
        return coordinated
    
    async def _generate_collective_signals(
        self, 
        coordinated_signals: List[Dict], 
        market_data: Dict[str, Any]
    ) -> List[Dict]:
        """Generate signals based on collective swarm behavior"""
        
        collective_signals = []
        
        # Detect market opportunities that individual pods might miss
        if self.market_regime == "volatile":
            # Generate volatility arbitrage signal
            collective_signals.append({
                "symbol": market_data.get("symbol", "SPY"),
                "side": "buy",
                "price": market_data.get("price", 400),
                "quantity": 10,
                "strategy": "volatility_arbitrage",
                "confidence": 0.6,
                "reasoning": "Swarm detected high volatility opportunity",
                "swarm_coordination": "collective_opportunity"
            })
        
        elif len(coordinated_signals) == 0 and len(self.pods) > 5:
            # Generate exploration signal when all pods are idle
            collective_signals.append({
                "symbol": market_data.get("symbol", "SPY"),
                "side": random.choice(["buy", "sell"]),
                "price": market_data.get("price", 400),
                "quantity": 5,
                "strategy": "swarm_exploration",
                "confidence": 0.4,
                "reasoning": "Swarm exploration to find new opportunities",
                "swarm_coordination": "exploration"
            })
        
        return collective_signals
    
    async def _share_swarm_information(self, signals: List[Dict]):
        """Share information between pods based on signals"""
        
        if not signals:
            return
        
        # Create information sharing messages
        for signal in signals:
            if signal.get("confidence", 0) > 0.8:  # High confidence signals
                message = SwarmMessage(
                    sender_id="swarm_coordinator",
                    message_type="signal",
                    content={
                        "signal": signal,
                        "market_regime": self.market_regime,
                        "swarm_consensus": len([s for s in signals if s.get("side") == signal.get("side")])
                    },
                    timestamp=datetime.now(),
                    priority=8
                )
                
                # Send to relevant pods
                for pod in self.pods.values():
                    if pod.strategy.value in signal.get("strategy", ""):
                        pod.receive_message(message)
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        active_pods = [pod for pod in self.pods.values() if pod.status == PodStatus.ACTIVE]
        
        return {
            "total_pods": len(self.pods),
            "active_pods": len(active_pods),
            "market_regime": self.market_regime,
            "average_fitness": np.mean([pod.performance.fitness_score for pod in active_pods]) if active_pods else 0,
            "total_capital": sum(pod.capital for pod in self.pods.values()),
            "strategy_distribution": {
                strategy.value: len([pod for pod in self.pods.values() if pod.strategy == strategy])
                for strategy in PodStrategy
            },
            "performance_summary": {
                "best_performer": max(self.pods.values(), key=lambda x: x.performance.fitness_score).pod_id if self.pods else None,
                "worst_performer": min(self.pods.values(), key=lambda x: x.performance.fitness_score).pod_id if self.pods else None,
                "average_return": np.mean([pod.performance.total_return for pod in self.pods.values()]) if self.pods else 0
            }
        }

class GeneticAlgorithm:
    """Manages genetic evolution of trading pod populations"""
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.generation = 0
    
    async def evolve_population(
        self, 
        current_pods: List[TradingPod],
        swarm: SwarmIntelligence
    ) -> List[TradingPod]:
        """Evolve pod population using genetic algorithm"""
        
        if len(current_pods) < 5:  # Not enough pods for evolution
            return current_pods
        
        self.generation += 1
        logger.info(f"Starting genetic evolution - Generation {self.generation}")
        
        # Evaluate fitness and sort by performance
        pods_with_fitness = [(pod, pod.performance.fitness_score) for pod in current_pods]
        pods_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select elite pods (survivors)
        elite_pods = [pod for pod, _ in pods_with_fitness[:self.elite_size]]
        
        # Generate new population
        new_pods = []
        new_pods.extend(elite_pods)  # Keep elite
        
        # Generate offspring through crossover and mutation
        while len(new_pods) < self.population_size:
            # Selection for reproduction
            parent1 = self._tournament_selection(pods_with_fitness)
            parent2 = self._tournament_selection(pods_with_fitness)
            
            if parent1 != parent2 and random.random() < self.crossover_rate:
                # Crossover
                child_genes = TradingGenes.crossover(parent1.genes, parent2.genes)
                child_strategy = random.choice([parent1.strategy, parent2.strategy])
            else:
                # Clone better parent
                better_parent = parent1 if parent1.performance.fitness_score > parent2.performance.fitness_score else parent2
                child_genes = better_parent.genes
                child_strategy = better_parent.strategy
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_genes = child_genes.mutate(self.mutation_rate)
            
            # Create new pod
            child_pod = TradingPod(
                pod_id=f"pod_{self.generation}_{len(new_pods)}",
                strategy=child_strategy,
                genes=child_genes,
                initial_capital=10000.0
            )
            child_pod.generation = self.generation
            
            new_pods.append(child_pod)
        
        # Retire old pods and add new ones to swarm
        for pod in current_pods:
            if pod not in elite_pods:
                pod.status = PodStatus.RETIRED
                swarm.remove_pod(pod.pod_id)
        
        for new_pod in new_pods:
            if new_pod not in elite_pods:
                new_pod.status = PodStatus.ACTIVE
                swarm.add_pod(new_pod)
        
        logger.info(f"Evolution complete - {len(new_pods)} pods in new generation")
        
        return new_pods
    
    def _tournament_selection(
        self, 
        pods_with_fitness: List[Tuple[TradingPod, float]], 
        tournament_size: int = 3
    ) -> TradingPod:
        """Select pod using tournament selection"""
        tournament = random.sample(pods_with_fitness, min(tournament_size, len(pods_with_fitness)))
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

class AutonomousTradingPodSystem:
    """Main system managing autonomous trading pods"""
    
    def __init__(self, initial_pods: int = 10):
        self.swarm = SwarmIntelligence()
        self.genetic_algorithm = GeneticAlgorithm()
        self.pods: List[TradingPod] = []
        self.total_capital = 0.0
        self.system_start_time = datetime.now()
        
        # Initialize pod population
        self._initialize_pod_population(initial_pods)
    
    def _initialize_pod_population(self, count: int):
        """Initialize the initial pod population"""
        strategies = list(PodStrategy)
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            genes = TradingGenes.random()
            
            pod = TradingPod(
                pod_id=f"genesis_pod_{i}",
                strategy=strategy,
                genes=genes,
                initial_capital=10000.0
            )
            pod.status = PodStatus.ACTIVE
            
            self.pods.append(pod)
            self.swarm.add_pod(pod)
            self.total_capital += pod.capital
        
        logger.info(f"Initialized {count} trading pods")
    
    async def process_market_update(self, market_data: Dict[str, Any]) -> List[Dict]:
        """Process market update through the pod system"""
        try:
            # Coordinate pods and generate signals
            signals = await self.swarm.coordinate_pods(market_data)
            
            # Check for evolution triggers
            await self._check_evolution_triggers()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error processing market update: {e}")
            return []
    
    async def _check_evolution_triggers(self):
        """Check if population should evolve"""
        active_pods = [pod for pod in self.pods if pod.status == PodStatus.ACTIVE]
        
        # Evolution triggers
        should_evolve = False
        
        # Time-based evolution (weekly)
        time_since_start = datetime.now() - self.system_start_time
        if time_since_start.days > 0 and time_since_start.days % 7 == 0:
            should_evolve = True
        
        # Performance-based evolution
        if active_pods:
            avg_fitness = np.mean([pod.performance.fitness_score for pod in active_pods])
            if avg_fitness < 0.4:  # Poor overall performance
                should_evolve = True
            
            # Individual pod evolution checks
            for pod in active_pods:
                if pod.should_evolve():
                    pod.evolve(self.genetic_algorithm.mutation_rate)
        
        # Population evolution
        if should_evolve and len(active_pods) >= 5:
            self.pods = await self.genetic_algorithm.evolve_population(active_pods, self.swarm)
    
    def update_pod_performance(self, pod_id: str, trade_result: Dict):
        """Update performance for specific pod"""
        pod = next((p for p in self.pods if p.pod_id == pod_id), None)
        if pod:
            pod.update_performance(trade_result)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_info": {
                "total_pods": len(self.pods),
                "total_capital": sum(pod.capital for pod in self.pods),
                "system_uptime": str(datetime.now() - self.system_start_time),
                "generation": self.genetic_algorithm.generation
            },
            "swarm_status": self.swarm.get_swarm_status(),
            "top_performers": [
                pod.get_status_report() 
                for pod in sorted(self.pods, key=lambda x: x.performance.fitness_score, reverse=True)[:5]
            ],
            "strategy_performance": self._get_strategy_performance(),
            "genetic_diversity": self._calculate_genetic_diversity()
        }
    
    def _get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance by strategy type"""
        strategy_performance = {}
        
        for strategy in PodStrategy:
            strategy_pods = [pod for pod in self.pods if pod.strategy == strategy]
            if strategy_pods:
                strategy_performance[strategy.value] = {
                    "count": len(strategy_pods),
                    "avg_return": np.mean([pod.performance.total_return for pod in strategy_pods]),
                    "avg_fitness": np.mean([pod.performance.fitness_score for pod in strategy_pods]),
                    "best_performer": max(strategy_pods, key=lambda x: x.performance.fitness_score).pod_id
                }
        
        return strategy_performance
    
    def _calculate_genetic_diversity(self) -> float:
        """Calculate genetic diversity of current population"""
        if len(self.pods) < 2:
            return 0.0
        
        # Sample genetic parameters for diversity calculation
        gene_vectors = []
        for pod in self.pods:
            vector = [
                pod.genes.risk_tolerance,
                pod.genes.position_size,
                pod.genes.momentum_threshold,
                pod.genes.news_weight,
                pod.genes.technical_weight
            ]
            gene_vectors.append(vector)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(gene_vectors)):
            for j in range(i + 1, len(gene_vectors)):
                distance = np.linalg.norm(np.array(gene_vectors[i]) - np.array(gene_vectors[j]))
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0 