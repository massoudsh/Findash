"""
Intelligence Orchestrator for the Octopus Trading Platform
Coordinates AI agents and manages the flow of information between different modules
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntelligenceReport:
    """Comprehensive intelligence report combining all agent analyses"""
    symbol: str
    timeframe: str
    strategy_decision: Any
    ml_predictions: Dict[str, Any]
    price_forecasts: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    final_recommendation: str
    confidence_score: float
    risk_assessment: Dict[str, Any]
    expected_return: float
    time_horizon: Any
    agent_consensus: Dict[str, Any]
    uncertainty_factors: List[str]
    market_conditions: Dict[str, Any]
    timestamp: datetime

@dataclass
class AgentTask:
    """Represents a task for an AI agent"""
    agent_name: str
    task_type: str
    priority: int
    data: Dict[str, Any]
    created_at: datetime
    status: str = "pending"

@dataclass
class AgentResult:
    """Represents the result from an AI agent"""
    agent_name: str
    task_id: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None

class IntelligenceOrchestrator:
    """
    Orchestrates the 11 AI agents in the Octopus Trading Platform
    Manages task distribution, priority, and coordination between agents
    """
    
    def __init__(self, cache=None):
        self.cache = cache
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.active_tasks = {}
        self.agent_status = {}
        
        # Placeholder agent references (set by initialize_agents)
        self.strategy_agent = None
        self.ml_agent = None
        self.prediction_agent = None
        self.sentiment_agent = None
        self.agent_weights = {
            "strategy": 0.30,
            "ml": 0.25,
            "prediction": 0.25,
            "sentiment": 0.20
        }
        self.active_symbols: Dict[str, Dict[str, Any]] = {}
        self.intelligence_history: Dict[str, List[IntelligenceReport]] = {}
        
        # Initialize agent registry
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize the registry of available agents"""
        self.agents = {
            "M1_data_collector": {
                "name": "Data Collection Agent",
                "status": "active",
                "capabilities": ["web_scraping", "api_fetching", "market_data"],
                "priority": 1
            },
            "M2_data_warehouse": {
                "name": "Data Warehouse Agent", 
                "status": "active",
                "capabilities": ["data_storage", "data_retrieval", "data_validation"],
                "priority": 2
            },
            "M3_realtime_processor": {
                "name": "Real-time Processing Agent",
                "status": "active", 
                "capabilities": ["stream_processing", "real_time_analysis", "alerts"],
                "priority": 1
            },
            "M4_strategy_agent": {
                "name": "Strategy Agent",
                "status": "active",
                "capabilities": ["strategy_execution", "signal_generation", "backtesting"],
                "priority": 3
            },
            "M5_ml_models": {
                "name": "ML Models Agent",
                "status": "active",
                "capabilities": ["prediction", "classification", "deep_learning"],
                "priority": 2
            },
            "M6_risk_manager": {
                "name": "Risk Management Agent",
                "status": "active",
                "capabilities": ["risk_assessment", "portfolio_optimization", "compliance"],
                "priority": 1
            },
            "M7_price_predictor": {
                "name": "Price Prediction Agent",
                "status": "active",
                "capabilities": ["time_series_forecasting", "prophet", "neural_networks"],
                "priority": 2
            },
            "M8_paper_trader": {
                "name": "Paper Trading Agent",
                "status": "active",
                "capabilities": ["simulated_trading", "execution_simulation", "performance_tracking"],
                "priority": 3
            },
            "M9_sentiment_analyzer": {
                "name": "Market Sentiment Agent",
                "status": "active",
                "capabilities": ["sentiment_analysis", "news_analysis", "social_media_monitoring"],
                "priority": 2
            },
            "M10_backtester": {
                "name": "Backtesting Agent",
                "status": "active",
                "capabilities": ["historical_testing", "performance_analysis", "strategy_validation"],
                "priority": 3
            },
            "M11_visualizer": {
                "name": "Visualization Agent",
                "status": "active",
                "capabilities": ["chart_generation", "dashboard_updates", "reporting"],
                "priority": 4
            }
        }
        
        logger.info(f"Initialized {len(self.agents)} AI agents")
    
    async def submit_task(self, agent_name: str, task_type: str, data: Dict[str, Any], priority: int = 5) -> str:
        """Submit a task to be processed by a specific agent"""
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        if self.agents[agent_name]["status"] != "active":
            raise ValueError(f"Agent {agent_name} is not active")
        
        task_id = f"{agent_name}_{task_type}_{int(datetime.utcnow().timestamp())}"
        
        task = AgentTask(
            agent_name=agent_name,
            task_type=task_type,
            priority=priority,
            data=data,
            created_at=datetime.utcnow()
        )
        
        # Add to task queue
        await self.task_queue.put((priority, task_id, task))
        self.active_tasks[task_id] = task
        
        logger.info(f"Submitted task {task_id} to agent {agent_name}")
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[AgentResult]:
        """Get the result of a completed task"""
        return self.results.get(task_id)
    
    async def coordinate_pipeline(self, symbol: str, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Coordinate a full analysis pipeline across multiple agents
        """
        pipeline_id = f"pipeline_{symbol}_{int(datetime.utcnow().timestamp())}"
        results = {}
        
        try:
            logger.info(f"Starting coordination pipeline {pipeline_id} for {symbol}")
            
            # Stage 1: Data Collection (M1)
            data_task = await self.submit_task(
                "M1_data_collector", 
                "fetch_market_data", 
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=1
            )
            
            # Stage 2: Real-time Processing (M3) - parallel with data collection
            realtime_task = await self.submit_task(
                "M3_realtime_processor",
                "process_realtime",
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=1
            )
            
            # Stage 3: Sentiment Analysis (M9) - can run in parallel
            sentiment_task = await self.submit_task(
                "M9_sentiment_analyzer",
                "analyze_sentiment", 
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=2
            )
            
            # Wait for data collection to complete before proceeding
            # In a real implementation, this would wait for actual task completion
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Stage 4: ML Prediction (M5 & M7) - depends on data
            ml_task = await self.submit_task(
                "M5_ml_models",
                "generate_prediction",
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=2
            )
            
            price_prediction_task = await self.submit_task(
                "M7_price_predictor",
                "predict_price",
                {"symbol": symbol, "pipeline_id": pipeline_id}, 
                priority=2
            )
            
            # Stage 5: Risk Analysis (M6)
            risk_task = await self.submit_task(
                "M6_risk_manager",
                "assess_risk",
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=1
            )
            
            # Stage 6: Strategy Generation (M4)
            strategy_task = await self.submit_task(
                "M4_strategy_agent", 
                "generate_signals",
                {"symbol": symbol, "pipeline_id": pipeline_id},
                priority=3
            )
            
            # If full analysis requested, include backtesting and visualization
            if analysis_type == "full":
                backtest_task = await self.submit_task(
                    "M10_backtester",
                    "run_backtest",
                    {"symbol": symbol, "pipeline_id": pipeline_id},
                    priority=3
                )
                
                viz_task = await self.submit_task(
                    "M11_visualizer",
                    "generate_charts",
                    {"symbol": symbol, "pipeline_id": pipeline_id},
                    priority=4
                )
            
            # Simulate pipeline completion
            results = {
                "pipeline_id": pipeline_id,
                "symbol": symbol,
                "status": "completed",
                "tasks_submitted": 8 if analysis_type == "full" else 6,
                "completion_time": datetime.utcnow().isoformat(),
                "stages": {
                    "data_collection": "completed",
                    "realtime_processing": "completed", 
                    "sentiment_analysis": "completed",
                    "ml_prediction": "completed",
                    "price_prediction": "completed",
                    "risk_analysis": "completed",
                    "strategy_generation": "completed"
                }
            }
            
            if analysis_type == "full":
                results["stages"]["backtesting"] = "completed"
                results["stages"]["visualization"] = "completed"
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
            results = {
                "pipeline_id": pipeline_id,
                "symbol": symbol,
                "status": "failed",
                "error": str(e),
                "completion_time": datetime.utcnow().isoformat()
            }
        
        return results
    
    async def get_agent_status(self, agent_name: str = None) -> Dict[str, Any]:
        """Get status of one or all agents"""
        
        if agent_name:
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            return self.agents[agent_name]
        
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a["status"] == "active"]),
            "agents": self.agents,
            "task_queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.results)
        }
    
    async def set_agent_status(self, agent_name: str, status: str) -> bool:
        """Set the status of an agent (active, inactive, maintenance)"""
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        valid_statuses = ["active", "inactive", "maintenance"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        self.agents[agent_name]["status"] = status
        logger.info(f"Agent {agent_name} status changed to {status}")
        
        return True
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a["status"] == "active"])
        
        health_percentage = (active_agents / total_agents) * 100 if total_agents > 0 else 0
        
        if health_percentage >= 90:
            overall_status = "healthy"
        elif health_percentage >= 70:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "health_percentage": round(health_percentage, 1),
            "total_agents": total_agents,
            "active_agents": active_agents,
            "inactive_agents": total_agents - active_agents,
            "task_queue_size": self.task_queue.qsize(),
            "active_pipelines": len([t for t in self.active_tasks.values() if "pipeline" in t.task_type]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Intelligence Orchestrator...")
        
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            del self.active_tasks[task_id]
        
        # Clear task queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("Intelligence Orchestrator shutdown complete")

    async def initialize_agents(self):
        pass

    async def cleanup(self):
        """Cleanup method for graceful shutdown"""
        pass

    async def add_symbol_for_analysis(self, symbol: str, timeframes: List[str]):
        """Register a symbol for intelligence analysis."""
        self.active_symbols[symbol] = {
            "timeframes": timeframes,
            "analysis_count": 0
        }

    async def generate_intelligence_report(self, symbol: str, timeframe: str) -> IntelligenceReport:
        """Generate a comprehensive intelligence report for a symbol."""
        intelligence = await asyncio.gather(
            self._get_strategy_intelligence(symbol, timeframe),
            self._get_ml_intelligence(symbol, timeframe),
            self._get_prediction_intelligence(symbol, timeframe),
            self._get_sentiment_intelligence(symbol),
            return_exceptions=True
        )
        strategy_result, ml_result, prediction_result, sentiment_result = intelligence

        if isinstance(strategy_result, Exception):
            strategy_result = None
        if isinstance(ml_result, Exception):
            ml_result = {}
        if isinstance(prediction_result, Exception):
            prediction_result = {}
        if isinstance(sentiment_result, Exception):
            sentiment_result = {}

        agent_recommendations = {}
        agent_confidences = {}
        final_recommendation = "HOLD"
        confidence_score = 0.5

        if strategy_result is not None:
            agent_recommendations["strategy"] = strategy_result.action.value
            agent_confidences["strategy"] = strategy_result.confidence

        try:
            final_recommendation, confidence_score = await self._build_consensus(
                agent_recommendations, agent_confidences
            )
        except Exception:
            pass

        risk_assessment = await self._calculate_unified_risk(
            symbol, strategy_result, ml_result, prediction_result, sentiment_result
        )

        uncertainty_factors = await self._identify_uncertainty_factors(
            agent_recommendations, agent_confidences, ml_result, sentiment_result
        )

        market_conditions = await self._analyze_market_conditions(symbol, timeframe)

        report = IntelligenceReport(
            symbol=symbol,
            timeframe=timeframe,
            strategy_decision=strategy_result,
            ml_predictions=ml_result,
            price_forecasts=prediction_result,
            sentiment_analysis=sentiment_result,
            final_recommendation=final_recommendation,
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            expected_return=0.0,
            time_horizon=timedelta(hours=1),
            agent_consensus={"recommendations": agent_recommendations, "confidences": agent_confidences},
            uncertainty_factors=uncertainty_factors,
            market_conditions=market_conditions,
            timestamp=datetime.utcnow()
        )

        self.intelligence_history.setdefault(symbol, []).append(report)
        self._cleanup_old_history(symbol)
        if symbol in self.active_symbols:
            self.active_symbols[symbol]["analysis_count"] += 1

        return report

    def _cleanup_old_history(self, symbol: str, max_age_hours: int = 24):
        """Remove reports older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        if symbol in self.intelligence_history:
            self.intelligence_history[symbol] = [
                r for r in self.intelligence_history[symbol]
                if r.timestamp > cutoff
            ]

    async def _get_strategy_intelligence(self, symbol: str, timeframe: str):
        """Collect intelligence from the strategy agent."""
        return None

    async def _get_ml_intelligence(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Collect intelligence from the ML agent."""
        return {}

    async def _get_prediction_intelligence(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Collect intelligence from the prediction agent."""
        return {}

    async def _get_sentiment_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Collect intelligence from the sentiment agent."""
        return {}

    async def _build_consensus(
        self,
        agent_recommendations: Dict[str, str],
        agent_confidences: Dict[str, float]
    ) -> tuple:
        """Build consensus from agent recommendations."""
        return "HOLD", 0.5

    async def _calculate_unified_risk(
        self,
        symbol: str,
        strategy_result,
        ml_result: Dict[str, Any],
        prediction_result: Dict[str, Any],
        sentiment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate unified risk assessment."""
        return {
            "var_95": 0.05,
            "volatility": 0.02,
            "anomaly_risk": 0.0,
            "predicted_volatility": 0.025,
            "sentiment_uncertainty": 0.0,
            "overall_risk": 0.1
        }

    async def _identify_uncertainty_factors(
        self,
        agent_recommendations: Dict[str, str],
        agent_confidences: Dict[str, float],
        ml_result: Dict[str, Any],
        sentiment_result: Dict[str, Any]
    ) -> List[str]:
        """Identify uncertainty factors in the analysis."""
        factors = []
        return factors

    async def _analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze current market conditions."""
        return {
            "volatility_regime": "normal",
            "trend_strength": 0.5,
            "trend_direction": "sideways",
            "momentum": 0.0
        }

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status report."""
        return {
            "orchestrator_id": "main",
            "active_symbols": list(self.active_symbols.keys()),
            "agent_weights": self.agent_weights,
            "agent_statuses": {
                "strategy": {"agent_id": "M4_strategy_agent", "status": "active"},
                "ml": {"agent_id": "M5_ml_models", "status": "active"},
                "prediction": {"agent_id": "M7_price_predictor", "status": "active"},
                "sentiment": {"agent_id": "M9_sentiment_analyzer", "status": "active"}
            },
            "intelligence_history_counts": {
                sym: len(hist) for sym, hist in self.intelligence_history.items()
            }
        }

# Global instance
intelligence_orchestrator = IntelligenceOrchestrator() 