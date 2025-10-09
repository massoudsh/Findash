"""
Tests for the Intelligence Orchestrator and Phase 3 integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.core.cache import TradingCache
from src.core.intelligence_orchestrator import IntelligenceOrchestrator, IntelligenceReport
from src.strategies.strategy_agent import TradingDecision, SignalType
from src.strategies.signal_fusion import TradingSignal


@pytest.fixture
async def mock_cache():
    """Create a mock cache for testing."""
    cache = MagicMock(spec=TradingCache)
    cache.get = AsyncMock()
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    return cache


@pytest.fixture
async def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq='H')
    
    # Generate realistic price data
    base_price = 150.0
    prices = []
    
    for i in range(100):
        # Add some trend and noise
        trend = i * 0.1
        noise = np.random.normal(0, 1)
        price = base_price + trend + noise
        prices.append(price)
    
    return [
        {
            "timestamp": dates[i].isoformat(),
            "open": prices[i] * 0.999,
            "high": prices[i] * 1.002,
            "low": prices[i] * 0.998,
            "close": prices[i],
            "volume": np.random.randint(1000000, 5000000)
        }
        for i in range(100)
    ]


@pytest.fixture
async def intelligence_orchestrator(mock_cache):
    """Create an intelligence orchestrator for testing."""
    orchestrator = IntelligenceOrchestrator(mock_cache)
    
    # Mock the agent initialization
    with patch.object(orchestrator, 'initialize_agents', new_callable=AsyncMock) as mock_init:
        mock_init.return_value = {
            "strategy_agent": True,
            "ml_agent": True,
            "prediction_agent": True,
            "sentiment_agent": True
        }
        await orchestrator.initialize_agents()
    
    return orchestrator


class TestIntelligenceOrchestrator:
    """Test suite for Intelligence Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_cache):
        """Test orchestrator initialization."""
        orchestrator = IntelligenceOrchestrator(mock_cache)
        
        # Check that all agents are initialized
        assert orchestrator.strategy_agent is not None
        assert orchestrator.ml_agent is not None
        assert orchestrator.prediction_agent is not None
        assert orchestrator.sentiment_agent is not None
        
        # Check default weights
        assert orchestrator.agent_weights["strategy"] == 0.30
        assert orchestrator.agent_weights["ml"] == 0.25
        assert orchestrator.agent_weights["prediction"] == 0.25
        assert orchestrator.agent_weights["sentiment"] == 0.20
    
    @pytest.mark.asyncio
    async def test_add_symbol_for_analysis(self, intelligence_orchestrator):
        """Test adding symbols for analysis."""
        symbol = "AAPL"
        timeframes = ["1h", "4h", "1d"]
        
        await intelligence_orchestrator.add_symbol_for_analysis(symbol, timeframes)
        
        assert symbol in intelligence_orchestrator.active_symbols
        assert intelligence_orchestrator.active_symbols[symbol]["timeframes"] == timeframes
        assert intelligence_orchestrator.active_symbols[symbol]["analysis_count"] == 0
    
    @pytest.mark.asyncio
    async def test_strategy_intelligence_collection(self, intelligence_orchestrator, sample_market_data):
        """Test strategy intelligence collection."""
        symbol = "AAPL"
        timeframe = "1h"
        
        # Mock strategy decision
        mock_decision = TradingDecision(
            symbol=symbol,
            action=SignalType.BUY,
            confidence=0.75,
            position_size=0.1,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            time_horizon=timedelta(hours=6),
            strategy_allocation={"momentum": 0.5, "technical": 0.3, "risk_aware": 0.2},
            risk_metrics={"var_95": 0.05, "volatility": 0.02},
            timestamp=datetime.utcnow()
        )
        
        with patch.object(intelligence_orchestrator.strategy_agent, 'generate_trading_decision', 
                         new_callable=AsyncMock) as mock_strategy:
            mock_strategy.return_value = mock_decision
            
            result = await intelligence_orchestrator._get_strategy_intelligence(symbol, timeframe)
            
            assert result is not None
            assert result.symbol == symbol
            assert result.action == SignalType.BUY
            assert result.confidence == 0.75
    
    @pytest.mark.asyncio
    async def test_ml_intelligence_collection(self, intelligence_orchestrator):
        """Test ML intelligence collection."""
        symbol = "AAPL"
        timeframe = "1h"
        
        # Mock ML predictions
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {
            "predicted_price": 155.0,
            "confidence_score": 0.8,
            "model_name": "ensemble"
        }
        
        mock_anomaly = {
            "anomaly_detected": False,
            "reconstruction_error": 0.05
        }
        
        with patch.object(intelligence_orchestrator.ml_agent, 'ensemble_predict', 
                         new_callable=AsyncMock) as mock_ensemble, \
             patch.object(intelligence_orchestrator.ml_agent, 'detect_anomalies', 
                         new_callable=AsyncMock) as mock_anomaly_detect, \
             patch.object(intelligence_orchestrator.ml_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_status:
            
            mock_ensemble.return_value = mock_prediction
            mock_anomaly_detect.return_value = mock_anomaly
            mock_status.return_value = {"agent_id": "M5_deep_learning_agent"}
            
            result = await intelligence_orchestrator._get_ml_intelligence(symbol, timeframe)
            
            assert result is not None
            assert "ensemble_prediction" in result
            assert "anomaly_detection" in result
            assert result["ensemble_prediction"]["predicted_price"] == 155.0
    
    @pytest.mark.asyncio
    async def test_prediction_intelligence_collection(self, intelligence_orchestrator):
        """Test prediction intelligence collection."""
        symbol = "AAPL"
        timeframe = "1h"
        
        # Mock comprehensive prediction
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {
            "trend_direction": "up",
            "trend_strength": 0.7,
            "predictions": {"1h_24h": 155.0}
        }
        
        mock_patterns = []
        
        with patch.object(intelligence_orchestrator.prediction_agent, 'generate_comprehensive_prediction', 
                         new_callable=AsyncMock) as mock_comprehensive, \
             patch.object(intelligence_orchestrator.prediction_agent, 'get_pattern_signals', 
                         new_callable=AsyncMock) as mock_patterns_func, \
             patch.object(intelligence_orchestrator.prediction_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_status:
            
            mock_comprehensive.return_value = mock_prediction
            mock_patterns_func.return_value = mock_patterns
            mock_status.return_value = {"agent_id": "M7_advanced_prediction_agent"}
            
            result = await intelligence_orchestrator._get_prediction_intelligence(symbol, timeframe)
            
            assert result is not None
            assert "comprehensive_prediction" in result
            assert "pattern_signals" in result
            assert result["comprehensive_prediction"]["trend_direction"] == "up"
    
    @pytest.mark.asyncio
    async def test_sentiment_intelligence_collection(self, intelligence_orchestrator):
        """Test sentiment intelligence collection."""
        symbol = "AAPL"
        
        # Mock sentiment summary
        mock_sentiment = {
            "current_sentiment": {
                "sentiment_score": 0.6,
                "confidence": 0.8,
                "overall_sentiment": "positive"
            }
        }
        
        with patch.object(intelligence_orchestrator.sentiment_agent, 'get_sentiment_summary', 
                         new_callable=AsyncMock) as mock_sentiment_func:
            
            mock_sentiment_func.return_value = {"sentiment_summary": mock_sentiment}
            
            result = await intelligence_orchestrator._get_sentiment_intelligence(symbol)
            
            assert result is not None
            assert "sentiment_summary" in result
            assert result["sentiment_summary"]["current_sentiment"]["sentiment_score"] == 0.6
    
    @pytest.mark.asyncio
    async def test_consensus_building(self, intelligence_orchestrator):
        """Test consensus building from agent recommendations."""
        
        # Test case 1: Strong buy consensus
        agent_recommendations = {
            "strategy": "buy",
            "ml": "buy",
            "prediction": "buy",
            "sentiment": "hold"
        }
        
        agent_confidences = {
            "strategy": 0.8,
            "ml": 0.7,
            "prediction": 0.75,
            "sentiment": 0.6
        }
        
        final_rec, confidence = await intelligence_orchestrator._build_consensus(
            agent_recommendations, agent_confidences
        )
        
        assert final_rec == "BUY"
        assert confidence > 0.6  # Should have good confidence
        
        # Test case 2: Conflicting signals
        agent_recommendations = {
            "strategy": "buy",
            "ml": "sell",
            "prediction": "hold",
            "sentiment": "sell"
        }
        
        agent_confidences = {
            "strategy": 0.6,
            "ml": 0.7,
            "prediction": 0.5,
            "sentiment": 0.8
        }
        
        final_rec, confidence = await intelligence_orchestrator._build_consensus(
            agent_recommendations, agent_confidences
        )
        
        # With conflicting signals, should either be HOLD or have low confidence
        assert confidence < 0.8
    
    @pytest.mark.asyncio
    async def test_unified_risk_calculation(self, intelligence_orchestrator):
        """Test unified risk assessment calculation."""
        
        symbol = "AAPL"
        
        # Mock strategy decision with risk metrics
        strategy_result = TradingDecision(
            symbol=symbol,
            action=SignalType.BUY,
            confidence=0.75,
            position_size=0.1,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            time_horizon=timedelta(hours=6),
            strategy_allocation={},
            risk_metrics={"var_95": 0.05, "volatility": 0.02},
            timestamp=datetime.utcnow()
        )
        
        # Mock ML result with anomaly detection
        ml_result = {
            "anomaly_detection": {
                "anomaly_detected": False,
                "reconstruction_error": 0.03
            }
        }
        
        # Mock prediction result with volatility forecast
        prediction_result = {
            "comprehensive_prediction": {
                "volatility_forecast": 0.025
            }
        }
        
        # Mock sentiment result
        sentiment_result = {
            "sentiment_summary": {
                "current_sentiment": {
                    "confidence": 0.7
                }
            }
        }
        
        risk_assessment = await intelligence_orchestrator._calculate_unified_risk(
            symbol, strategy_result, ml_result, prediction_result, sentiment_result
        )
        
        assert "var_95" in risk_assessment
        assert "volatility" in risk_assessment
        assert "anomaly_risk" in risk_assessment
        assert "predicted_volatility" in risk_assessment
        assert "sentiment_uncertainty" in risk_assessment
        assert "overall_risk" in risk_assessment
        
        # Overall risk should be between 0 and 1
        assert 0 <= risk_assessment["overall_risk"] <= 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_intelligence_report_generation(self, intelligence_orchestrator):
        """Test generation of comprehensive intelligence report."""
        
        symbol = "AAPL"
        timeframe = "1h"
        
        # Mock all agent results
        mock_strategy = TradingDecision(
            symbol=symbol,
            action=SignalType.BUY,
            confidence=0.75,
            position_size=0.1,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            time_horizon=timedelta(hours=6),
            strategy_allocation={},
            risk_metrics={"var_95": 0.05},
            timestamp=datetime.utcnow()
        )
        
        mock_ml = {
            "ensemble_prediction": {
                "predicted_price": 155.0,
                "confidence_score": 0.8
            }
        }
        
        mock_prediction = {
            "comprehensive_prediction": {
                "trend_direction": "up",
                "trend_strength": 0.7
            }
        }
        
        mock_sentiment = {
            "sentiment_summary": {
                "current_sentiment": {
                    "sentiment_score": 0.6,
                    "confidence": 0.8
                }
            }
        }
        
        # Mock agent methods
        with patch.object(intelligence_orchestrator, '_get_strategy_intelligence', 
                         new_callable=AsyncMock) as mock_strategy_intel, \
             patch.object(intelligence_orchestrator, '_get_ml_intelligence', 
                         new_callable=AsyncMock) as mock_ml_intel, \
             patch.object(intelligence_orchestrator, '_get_prediction_intelligence', 
                         new_callable=AsyncMock) as mock_pred_intel, \
             patch.object(intelligence_orchestrator, '_get_sentiment_intelligence', 
                         new_callable=AsyncMock) as mock_sent_intel:
            
            mock_strategy_intel.return_value = mock_strategy
            mock_ml_intel.return_value = mock_ml
            mock_pred_intel.return_value = mock_prediction
            mock_sent_intel.return_value = mock_sentiment
            
            report = await intelligence_orchestrator.generate_intelligence_report(symbol, timeframe)
            
            assert report is not None
            assert isinstance(report, IntelligenceReport)
            assert report.symbol == symbol
            assert report.timeframe == timeframe
            assert report.final_recommendation in ["BUY", "SELL", "HOLD"]
            assert 0 <= report.confidence_score <= 1
            assert isinstance(report.risk_assessment, dict)
            assert isinstance(report.agent_consensus, dict)
    
    @pytest.mark.asyncio
    async def test_uncertainty_factors_identification(self, intelligence_orchestrator):
        """Test identification of uncertainty factors."""
        
        # Test case 1: Conflicting recommendations
        agent_recommendations = {
            "strategy": "buy",
            "ml": "sell",
            "prediction": "hold",
            "sentiment": "buy"
        }
        
        agent_confidences = {
            "strategy": 0.4,  # Low confidence
            "ml": 0.3,        # Low confidence
            "prediction": 0.8,
            "sentiment": 0.7
        }
        
        ml_result = {
            "anomaly_detection": {
                "anomaly_detected": True  # Anomaly detected
            }
        }
        
        sentiment_result = {
            "sentiment_summary": {
                "current_sentiment": {
                    "confidence": 0.3  # Low sentiment confidence
                }
            }
        }
        
        uncertainty_factors = await intelligence_orchestrator._identify_uncertainty_factors(
            agent_recommendations, agent_confidences, ml_result, sentiment_result
        )
        
        assert len(uncertainty_factors) > 0
        assert any("Conflicting" in factor for factor in uncertainty_factors)
        assert any("Low confidence" in factor for factor in uncertainty_factors)
        assert any("anomaly detected" in factor for factor in uncertainty_factors)
        assert any("sentiment uncertainty" in factor for factor in uncertainty_factors)
    
    @pytest.mark.asyncio
    async def test_market_conditions_analysis(self, intelligence_orchestrator, sample_market_data):
        """Test market conditions analysis."""
        
        symbol = "AAPL"
        timeframe = "1h"
        
        # Mock cache to return sample market data
        intelligence_orchestrator.cache.get.return_value = sample_market_data
        
        conditions = await intelligence_orchestrator._analyze_market_conditions(symbol, timeframe)
        
        assert "volatility_regime" in conditions
        assert "trend_strength" in conditions
        assert "trend_direction" in conditions
        assert "momentum" in conditions
        
        # Check that values are reasonable
        assert conditions["volatility_regime"] in ["high", "normal", "low"]
        assert conditions["trend_direction"] in ["up", "down", "sideways"]
        assert isinstance(conditions["trend_strength"], float)
        assert isinstance(conditions["momentum"], float)
    
    @pytest.mark.asyncio
    async def test_orchestrator_status(self, intelligence_orchestrator):
        """Test orchestrator status reporting."""
        
        # Mock agent status methods
        mock_status = {"agent_id": "test_agent", "status": "active"}
        
        with patch.object(intelligence_orchestrator.strategy_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_strategy_status, \
             patch.object(intelligence_orchestrator.ml_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_ml_status, \
             patch.object(intelligence_orchestrator.prediction_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_pred_status, \
             patch.object(intelligence_orchestrator.sentiment_agent, 'get_agent_status', 
                         new_callable=AsyncMock) as mock_sent_status:
            
            mock_strategy_status.return_value = mock_status
            mock_ml_status.return_value = mock_status
            mock_pred_status.return_value = mock_status
            mock_sent_status.return_value = mock_status
            
            status = await intelligence_orchestrator.get_orchestrator_status()
            
            assert "orchestrator_id" in status
            assert "active_symbols" in status
            assert "agent_weights" in status
            assert "agent_statuses" in status
            assert "intelligence_history_counts" in status
            
            # Check that all agent statuses are included
            assert "strategy" in status["agent_statuses"]
            assert "ml" in status["agent_statuses"]
            assert "prediction" in status["agent_statuses"]
            assert "sentiment" in status["agent_statuses"]
    
    @pytest.mark.asyncio
    async def test_intelligence_history_management(self, intelligence_orchestrator):
        """Test intelligence history storage and cleanup."""
        
        symbol = "AAPL"
        
        # Create a mock report
        mock_report = IntelligenceReport(
            symbol=symbol,
            timeframe="1h",
            strategy_decision=None,
            ml_predictions={},
            price_forecasts={},
            sentiment_analysis={},
            final_recommendation="BUY",
            confidence_score=0.7,
            risk_assessment={},
            expected_return=0.02,
            time_horizon=timedelta(hours=6),
            agent_consensus={},
            uncertainty_factors=[],
            market_conditions={},
            timestamp=datetime.utcnow()
        )
        
        # Manually add to history
        intelligence_orchestrator.intelligence_history[symbol] = [mock_report]
        
        assert len(intelligence_orchestrator.intelligence_history[symbol]) == 1
        
        # Add an old report (should be cleaned up)
        old_report = IntelligenceReport(
            symbol=symbol,
            timeframe="1h",
            strategy_decision=None,
            ml_predictions={},
            price_forecasts={},
            sentiment_analysis={},
            final_recommendation="HOLD",
            confidence_score=0.5,
            risk_assessment={},
            expected_return=0.0,
            time_horizon=timedelta(hours=6),
            agent_consensus={},
            uncertainty_factors=[],
            market_conditions={},
            timestamp=datetime.utcnow() - timedelta(days=2)  # 2 days old
        )
        
        intelligence_orchestrator.intelligence_history[symbol].append(old_report)
        
        # Mock the generate_intelligence_report to trigger cleanup
        with patch.object(intelligence_orchestrator, '_get_strategy_intelligence', 
                         new_callable=AsyncMock) as mock_strategy, \
             patch.object(intelligence_orchestrator, '_get_ml_intelligence', 
                         new_callable=AsyncMock) as mock_ml, \
             patch.object(intelligence_orchestrator, '_get_prediction_intelligence', 
                         new_callable=AsyncMock) as mock_prediction, \
             patch.object(intelligence_orchestrator, '_get_sentiment_intelligence', 
                         new_callable=AsyncMock) as mock_sentiment:
            
            mock_strategy.return_value = None
            mock_ml.return_value = {}
            mock_prediction.return_value = {}
            mock_sentiment.return_value = {}
            
            await intelligence_orchestrator.generate_intelligence_report(symbol, "1h")
            
            # Check that old reports were cleaned up (only reports from last 24 hours should remain)
            recent_reports = [
                r for r in intelligence_orchestrator.intelligence_history[symbol]
                if r.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            # Should only have recent reports
            assert len(recent_reports) <= len(intelligence_orchestrator.intelligence_history[symbol])


@pytest.mark.asyncio
async def test_end_to_end_intelligence_workflow():
    """Test the complete intelligence analysis workflow."""
    
    # This test simulates a complete end-to-end workflow
    cache = MagicMock(spec=TradingCache)
    cache.get = AsyncMock()
    cache.set = AsyncMock()
    
    orchestrator = IntelligenceOrchestrator(cache)
    
    # Mock initialization
    with patch.object(orchestrator, 'initialize_agents', new_callable=AsyncMock):
        await orchestrator.initialize_agents()
    
    # Add a symbol for analysis
    await orchestrator.add_symbol_for_analysis("AAPL", ["1h", "4h"])
    
    # Mock all agent responses to simulate real workflow
    with patch.object(orchestrator, '_get_strategy_intelligence', new_callable=AsyncMock) as mock_strategy, \
         patch.object(orchestrator, '_get_ml_intelligence', new_callable=AsyncMock) as mock_ml, \
         patch.object(orchestrator, '_get_prediction_intelligence', new_callable=AsyncMock) as mock_prediction, \
         patch.object(orchestrator, '_get_sentiment_intelligence', new_callable=AsyncMock) as mock_sentiment:
        
        # Set up realistic mock responses
        mock_strategy.return_value = TradingDecision(
            symbol="AAPL",
            action=SignalType.BUY,
            confidence=0.75,
            position_size=0.1,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            time_horizon=timedelta(hours=6),
            strategy_allocation={},
            risk_metrics={"var_95": 0.05},
            timestamp=datetime.utcnow()
        )
        
        mock_ml.return_value = {
            "ensemble_prediction": {
                "predicted_price": 155.0,
                "confidence_score": 0.8
            }
        }
        
        mock_prediction.return_value = {
            "comprehensive_prediction": {
                "trend_direction": "up",
                "trend_strength": 0.7
            }
        }
        
        mock_sentiment.return_value = {
            "sentiment_summary": {
                "current_sentiment": {
                    "sentiment_score": 0.6,
                    "confidence": 0.8
                }
            }
        }
        
        # Generate intelligence report
        report = await orchestrator.generate_intelligence_report("AAPL", "1h")
        
        # Verify the complete workflow
        assert report is not None
        assert report.symbol == "AAPL"
        assert report.final_recommendation in ["BUY", "SELL", "HOLD"]
        assert 0 <= report.confidence_score <= 1
        
        # Verify that all agents were consulted
        mock_strategy.assert_called_once_with("AAPL", "1h")
        mock_ml.assert_called_once_with("AAPL", "1h")
        mock_prediction.assert_called_once_with("AAPL", "1h")
        mock_sentiment.assert_called_once_with("AAPL")
        
        # Verify that the report was cached
        cache.set.assert_called()
        
        # Verify that the symbol tracking was updated
        assert orchestrator.active_symbols["AAPL"]["analysis_count"] == 1 