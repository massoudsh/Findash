"""
Phase 4 Integration Tests
Tests for Real-Time Execution & Portfolio Management

This test suite covers:
- Risk Management Agent (M6)
- Execution Manager
- Portfolio Manager
- Enhanced Backtester (M10)
- Integration between components
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import json

from src.core.cache_manager import CacheManager
from src.realtime.websockets import WebSocketManager
from src.risk.risk_manager import RiskManager, RiskLevel
from src.trading.execution_manager import ExecutionManager, OrderRequest, OrderType, OrderSide
from src.portfolio.portfolio_manager import PortfolioManager, AllocationMethod
from src.backtesting.enhanced_backtester import EnhancedBacktester, BacktestConfig, BacktestMode
from src.strategies.strategy_agent import StrategyAgent

@pytest.fixture
async def cache_manager():
    """Mock cache manager for testing"""
    cache = Mock(spec=CacheManager)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    return cache

@pytest.fixture
async def websocket_manager():
    """Mock WebSocket manager for testing"""
    websocket = Mock(spec=WebSocketManager)
    websocket.broadcast_to_channel = AsyncMock(return_value=True)
    return websocket

@pytest.fixture
async def strategy_agent(cache_manager):
    """Mock strategy agent for testing"""
    strategy = Mock(spec=StrategyAgent)
    strategy.make_trading_decision = AsyncMock()
    return strategy

@pytest.fixture
async def risk_manager(cache_manager):
    """Risk manager instance for testing"""
    return RiskManager(cache_manager)

@pytest.fixture
async def execution_manager(cache_manager, websocket_manager):
    """Execution manager instance for testing"""
    return ExecutionManager(cache_manager, websocket_manager)

@pytest.fixture
async def portfolio_manager(cache_manager, risk_manager, execution_manager, strategy_agent):
    """Portfolio manager instance for testing"""
    return PortfolioManager(
        cache_manager=cache_manager,
        risk_manager=risk_manager,
        execution_manager=execution_manager,
        strategy_agent=strategy_agent
    )

@pytest.fixture
async def enhanced_backtester(cache_manager, strategy_agent):
    """Enhanced backtester instance for testing"""
    return EnhancedBacktester(cache_manager, strategy_agent)

@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    return {
        "AAPL": 10000.0,
        "MSFT": 8000.0,
        "GOOGL": 6000.0,
        "TSLA": 4000.0,
        "BTC-USD": 2000.0
    }

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    market_data = {}
    for symbol in ["AAPL", "MSFT", "GOOGL", "TSLA", "BTC-USD"]:
        # Generate synthetic data
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data[symbol] = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, len(dates)),
            'returns': returns
        })
    
    return market_data

class TestRiskManager:
    """Test suite for Risk Manager (M6)"""
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(self, risk_manager, sample_portfolio, sample_market_data):
        """Test comprehensive portfolio risk assessment"""
        
        risk_assessment = await risk_manager.assess_portfolio_risk(
            sample_portfolio, sample_market_data
        )
        
        # Verify risk assessment structure
        assert risk_assessment.total_value == sum(sample_portfolio.values())
        assert risk_assessment.risk_level in [level for level in RiskLevel]
        assert 0 <= risk_assessment.diversification_ratio <= 10  # Reasonable range
        assert 0 <= risk_assessment.concentration_risk <= 1
        assert isinstance(risk_assessment.sector_exposure, dict)
        
        # Verify VaR is reasonable
        assert 0 <= risk_assessment.total_var <= risk_assessment.total_value * 0.5
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, risk_manager):
        """Test optimal position sizing calculation"""
        
        position_size = await risk_manager.calculate_optimal_position_size(
            symbol="AAPL",
            expected_return=0.12,  # 12% expected return
            confidence=0.8,
            risk_budget=10000.0,
            current_portfolio={"MSFT": 20000.0, "GOOGL": 15000.0}
        )
        
        # Verify position size is reasonable
        assert 0 <= position_size <= 10000.0  # Within risk budget
        assert isinstance(position_size, float)
    
    @pytest.mark.asyncio
    async def test_risk_alerts(self, risk_manager):
        """Test risk alert generation"""
        
        # Portfolio with high concentration
        concentrated_portfolio = {
            "AAPL": 80000.0,  # 80% concentration
            "MSFT": 20000.0
        }
        
        alerts = await risk_manager.get_risk_alerts(concentrated_portfolio)
        
        # Should generate concentration alert
        assert len(alerts) > 0
        concentration_alerts = [a for a in alerts if a['type'] == 'concentration']
        assert len(concentration_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_stress_testing(self, risk_manager, sample_portfolio):
        """Test portfolio stress testing"""
        
        stress_scenarios = {
            'market_crash': {'AAPL': -0.30, 'default': -0.20},
            'tech_bubble': {'AAPL': -0.40, 'MSFT': -0.35, 'GOOGL': -0.35, 'default': -0.10}
        }
        
        stress_results = await risk_manager.stress_test_portfolio(
            sample_portfolio, stress_scenarios
        )
        
        # Verify stress test results
        assert 'market_crash' in stress_results
        assert 'tech_bubble' in stress_results
        
        # Results should be negative (losses)
        assert stress_results['market_crash'] < 0
        assert stress_results['tech_bubble'] < 0

class TestExecutionManager:
    """Test suite for Execution Manager"""
    
    @pytest.mark.asyncio
    async def test_order_submission(self, execution_manager):
        """Test order submission and validation"""
        
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        order_id = await execution_manager.submit_order(order_request)
        
        # Verify order was submitted
        assert isinstance(order_id, str)
        assert order_id in execution_manager.active_orders
        
        # Check order status
        order_status = await execution_manager.get_order_status(order_id)
        assert order_status is not None
        assert order_status['symbol'] == "AAPL"
        assert order_status['quantity'] == 100
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_manager):
        """Test order cancellation"""
        
        order_request = OrderRequest(
            symbol="MSFT",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            price=100.0
        )
        
        order_id = await execution_manager.submit_order(order_request)
        
        # Cancel the order
        cancelled = await execution_manager.cancel_order(order_id)
        
        assert cancelled is True
        assert order_id not in execution_manager.active_orders
        assert order_id in execution_manager.completed_orders
    
    @pytest.mark.asyncio
    async def test_execution_metrics(self, execution_manager):
        """Test execution metrics calculation"""
        
        # Submit a few orders
        for i in range(3):
            order_request = OrderRequest(
                symbol=f"TEST{i}",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET
            )
            await execution_manager.submit_order(order_request)
        
        metrics = await execution_manager.get_execution_metrics()
        
        # Verify metrics structure
        assert 'total_trades' in metrics
        assert 'total_volume' in metrics
        assert 'average_slippage' in metrics
        assert isinstance(metrics['total_trades'], int)

class TestPortfolioManager:
    """Test suite for Portfolio Manager"""
    
    @pytest.mark.asyncio
    async def test_portfolio_creation(self, portfolio_manager):
        """Test portfolio creation"""
        
        portfolio = await portfolio_manager.create_portfolio(
            portfolio_id="test_portfolio",
            name="Test Portfolio",
            initial_cash=100000.0,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Verify portfolio creation
        assert portfolio.portfolio_id == "test_portfolio"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.cash == 100000.0
        assert portfolio.total_value == 100000.0
        assert len(portfolio.positions) == 0
    
    @pytest.mark.asyncio
    async def test_position_management(self, portfolio_manager):
        """Test adding and removing positions"""
        
        # Create portfolio
        await portfolio_manager.create_portfolio(
            portfolio_id="test_positions",
            name="Position Test",
            initial_cash=50000.0
        )
        
        # Add position
        success = await portfolio_manager.add_position(
            portfolio_id="test_positions",
            symbol="AAPL",
            quantity=100,
            price=150.0
        )
        
        assert success is True
        
        portfolio = portfolio_manager.portfolios["test_positions"]
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100
        assert portfolio.cash == 35000.0  # 50000 - (100 * 150)
        
        # Remove position
        success = await portfolio_manager.remove_position(
            portfolio_id="test_positions",
            symbol="AAPL",
            quantity=50,
            price=155.0
        )
        
        assert success is True
        assert portfolio.positions["AAPL"].quantity == 50
        assert portfolio.cash > 35000.0  # Should have more cash
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self, portfolio_manager, cache_manager, strategy_agent):
        """Test portfolio optimization"""
        
        # Mock strategy agent decisions
        strategy_agent.make_trading_decision.return_value = Mock(
            expected_return=0.10,
            confidence=0.8,
            action="buy"
        )
        
        # Mock cache for price data
        cache_manager.get.return_value = {
            'returns': np.random.normal(0.001, 0.02, 252).tolist()
        }
        
        # Create portfolio
        await portfolio_manager.create_portfolio(
            portfolio_id="test_optimization",
            name="Optimization Test",
            initial_cash=100000.0
        )
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        optimal_allocation = await portfolio_manager.optimize_portfolio_allocation(
            portfolio_id="test_optimization",
            symbols=symbols,
            method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Verify optimization results
        assert len(optimal_allocation) == len(symbols)
        assert all(symbol in optimal_allocation for symbol in symbols)
        assert abs(sum(optimal_allocation.values()) - 1.0) < 0.01  # Should sum to ~1
    
    @pytest.mark.asyncio
    async def test_portfolio_performance(self, portfolio_manager):
        """Test portfolio performance calculation"""
        
        # Create portfolio with positions
        await portfolio_manager.create_portfolio(
            portfolio_id="test_performance",
            name="Performance Test",
            initial_cash=50000.0
        )
        
        await portfolio_manager.add_position(
            portfolio_id="test_performance",
            symbol="AAPL",
            quantity=100,
            price=150.0
        )
        
        await portfolio_manager.add_position(
            portfolio_id="test_performance",
            symbol="MSFT",
            quantity=200,
            price=100.0
        )
        
        performance = await portfolio_manager.get_portfolio_performance("test_performance")
        
        # Verify performance structure
        assert performance['portfolio_id'] == "test_performance"
        assert 'total_value' in performance
        assert 'positions' in performance
        assert 'allocation' in performance
        assert len(performance['positions']) == 2

class TestEnhancedBacktester:
    """Test suite for Enhanced Backtester (M10)"""
    
    @pytest.mark.asyncio
    async def test_historical_backtest(self, enhanced_backtester):
        """Test historical backtesting"""
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=100000.0,
            symbols=["AAPL", "MSFT"],
            rebalance_frequency="monthly"
        )
        
        results = await enhanced_backtester.run_backtest(
            config=config,
            mode=BacktestMode.HISTORICAL
        )
        
        # Verify backtest results
        assert results.config == config
        assert isinstance(results.total_return, float)
        assert isinstance(results.sharpe_ratio, float)
        assert isinstance(results.max_drawdown, float)
        assert results.execution_time > 0
        assert len(results.equity_curve) > 0
    
    @pytest.mark.asyncio
    async def test_monte_carlo_backtest(self, enhanced_backtester):
        """Test Monte Carlo simulation"""
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_capital=100000.0,
            symbols=["AAPL"],
            num_simulations=10  # Small number for testing
        )
        
        results = await enhanced_backtester.run_backtest(
            config=config,
            mode=BacktestMode.MONTE_CARLO
        )
        
        # Verify Monte Carlo results
        assert hasattr(results, 'monte_carlo_results')
        assert results.monte_carlo_results.num_simulations == 10
        assert 'total_return' in results.monte_carlo_results.confidence_intervals
        assert 0 <= results.monte_carlo_results.probability_of_loss <= 1
    
    @pytest.mark.asyncio
    async def test_walk_forward_backtest(self, enhanced_backtester):
        """Test walk-forward analysis"""
        
        config = BacktestConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0,
            symbols=["AAPL"],
            training_period_months=6,
            testing_period_months=3,
            step_size_months=3
        )
        
        results = await enhanced_backtester.run_backtest(
            config=config,
            mode=BacktestMode.WALK_FORWARD
        )
        
        # Verify walk-forward results
        assert hasattr(results, 'walk_forward_results')
        assert len(results.walk_forward_results.periods) > 0
        assert 'return' in results.walk_forward_results.average_performance
        assert 'return_std' in results.walk_forward_results.stability_metrics

class TestPhase4Integration:
    """Integration tests for Phase 4 components"""
    
    @pytest.mark.asyncio
    async def test_risk_portfolio_integration(self, risk_manager, portfolio_manager):
        """Test integration between risk manager and portfolio manager"""
        
        # Create portfolio
        await portfolio_manager.create_portfolio(
            portfolio_id="integration_test",
            name="Integration Test",
            initial_cash=100000.0
        )
        
        # Add some positions
        await portfolio_manager.add_position("integration_test", "AAPL", 100, 150.0)
        await portfolio_manager.add_position("integration_test", "MSFT", 200, 100.0)
        
        # Get portfolio
        portfolio = portfolio_manager.portfolios["integration_test"]
        portfolio_dict = {pos.symbol: pos.market_value for pos in portfolio.positions.values()}
        
        # Assess risk
        risk_alerts = await risk_manager.get_risk_alerts(portfolio_dict)
        
        # Should work without errors
        assert isinstance(risk_alerts, list)
    
    @pytest.mark.asyncio
    async def test_execution_portfolio_integration(self, execution_manager, portfolio_manager):
        """Test integration between execution manager and portfolio manager"""
        
        # Create portfolio
        await portfolio_manager.create_portfolio(
            portfolio_id="execution_test",
            name="Execution Test",
            initial_cash=100000.0
        )
        
        # Submit order through execution manager
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        order_id = await execution_manager.submit_order(order_request)
        
        # Verify order was created
        assert order_id in execution_manager.active_orders
        order_status = await execution_manager.get_order_status(order_id)
        assert order_status['symbol'] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_full_trading_workflow(self, portfolio_manager, risk_manager, execution_manager):
        """Test complete trading workflow"""
        
        # 1. Create portfolio
        portfolio = await portfolio_manager.create_portfolio(
            portfolio_id="workflow_test",
            name="Workflow Test",
            initial_cash=100000.0
        )
        
        # 2. Calculate position size using risk manager
        position_size = await risk_manager.calculate_optimal_position_size(
            symbol="AAPL",
            expected_return=0.15,
            confidence=0.8,
            risk_budget=20000.0,
            current_portfolio={}
        )
        
        # 3. Submit order through execution manager
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=position_size / 150.0,  # Assume $150 price
            order_type=OrderType.MARKET
        )
        
        order_id = await execution_manager.submit_order(order_request)
        
        # 4. Simulate order fill by adding position to portfolio
        await portfolio_manager.add_position(
            portfolio_id="workflow_test",
            symbol="AAPL",
            quantity=position_size / 150.0,
            price=150.0
        )
        
        # 5. Check portfolio performance
        performance = await portfolio_manager.get_portfolio_performance("workflow_test")
        
        # Verify workflow completed successfully
        assert performance['portfolio_id'] == "workflow_test"
        assert len(performance['positions']) == 1
        assert "AAPL" in performance['positions']

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 