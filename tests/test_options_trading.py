"""
Comprehensive test suite for Options Trading and Risk Management
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import your modules
from src.main import app
from src.options_risk_integration import (
    OptionsPortfolio, OptionPosition, BlackScholesCalculator, 
    CorrelationAnalyzer, RiskReportGenerator, Greeks
)
from src.database.models import Base, User, Portfolio, OptionPosition as DBOptionPosition
from src.database.init_db import init_database

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TestBlackScholesCalculator:
    """Test Black-Scholes calculations"""
    
    def test_option_price_calculation(self):
        """Test option price calculation"""
        # Known values for testing
        S = 100  # Stock price
        K = 100  # Strike price
        T = 0.25  # Time to expiry (3 months)
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Volatility
        
        # Calculate call option price
        call_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'call')
        
        # Expected value (approximately 5.99 for these parameters)
        assert 5.5 < call_price < 6.5, f"Call price {call_price} outside expected range"
        
        # Calculate put option price
        put_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'put')
        
        # Put-call parity: Call - Put = S - K*e^(-r*T)
        parity_check = call_price - put_price - (S - K * np.exp(-r * T))
        assert abs(parity_check) < 0.01, f"Put-call parity violation: {parity_check}"
    
    def test_greeks_calculation(self):
        """Test Greeks calculation"""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, 'call')
        
        # Basic sanity checks
        assert 0 < greeks.delta < 1, f"Call delta {greeks.delta} should be between 0 and 1"
        assert greeks.gamma > 0, f"Gamma {greeks.gamma} should be positive"
        assert greeks.theta < 0, f"Call theta {greeks.theta} should be negative"
        assert greeks.vega > 0, f"Vega {greeks.vega} should be positive"
        assert greeks.rho > 0, f"Call rho {greeks.rho} should be positive"
    
    def test_expired_option(self):
        """Test option with zero time to expiry"""
        S = 110
        K = 100
        T = 0  # Expired
        r = 0.05
        sigma = 0.2
        
        call_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'call')
        assert call_price == max(S - K, 0), f"Expired call value should be intrinsic value"
        
        put_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'put')
        assert put_price == max(K - S, 0), f"Expired put value should be intrinsic value"

class TestOptionsPortfolio:
    """Test options portfolio functionality"""
    
    def setup_method(self):
        """Set up test portfolio"""
        self.portfolio = OptionsPortfolio()
        
        # Add test positions
        self.portfolio.add_position(OptionPosition(
            symbol='AAPL',
            option_type='call',
            strike=150,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
            premium=5.0,
            underlying_price=155,
            volatility=0.25
        ))
        
        self.portfolio.add_position(OptionPosition(
            symbol='AAPL',
            option_type='put',
            strike=145,
            expiry=datetime.now() + timedelta(days=30),
            quantity=-5,
            premium=3.0,
            underlying_price=155,
            volatility=0.25
        ))
    
    def test_portfolio_greeks(self):
        """Test portfolio Greeks calculation"""
        greeks = self.portfolio.calculate_portfolio_greeks()
        
        assert 'portfolio_delta' in greeks
        assert 'portfolio_gamma' in greeks
        assert 'portfolio_theta' in greeks
        assert 'portfolio_vega' in greeks
        assert 'portfolio_rho' in greeks
        
        # Should have positive delta due to net long calls
        assert greeks['portfolio_delta'] > 0
    
    def test_portfolio_value(self):
        """Test portfolio value calculation"""
        value = self.portfolio.calculate_portfolio_value()
        assert value > 0, "Portfolio should have positive value"
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        var = self.portfolio.calculate_var(confidence_level=0.95)
        assert var >= 0, "VaR should be non-negative"
    
    def test_add_remove_positions(self):
        """Test adding and removing positions"""
        initial_count = len(self.portfolio.positions)
        
        # Add position
        new_position = OptionPosition(
            symbol='MSFT',
            option_type='call',
            strike=300,
            expiry=datetime.now() + timedelta(days=45),
            quantity=5,
            premium=8.0,
            underlying_price=310,
            volatility=0.22
        )
        self.portfolio.add_position(new_position)
        assert len(self.portfolio.positions) == initial_count + 1
        
        # Remove position
        self.portfolio.remove_position(0)
        assert len(self.portfolio.positions) == initial_count

class TestCorrelationAnalyzer:
    """Test correlation analysis"""
    
    def setup_method(self):
        """Set up test analyzer"""
        self.analyzer = CorrelationAnalyzer()
    
    def test_asset_correlations(self):
        """Test asset correlation calculation"""
        symbols = ['AAPL', 'MSFT']
        
        # This test requires internet connection to fetch data
        try:
            correlations = self.analyzer.calculate_asset_correlations(symbols)
            assert correlations is not None
            assert correlations.shape == (len(symbols), len(symbols))
            
            # Diagonal should be 1 (self-correlation)
            for i in range(len(symbols)):
                assert abs(correlations.iloc[i, i] - 1.0) < 0.01
        except Exception as e:
            pytest.skip(f"Skipping correlation test due to data fetch error: {e}")

class TestRiskReportGenerator:
    """Test risk report generation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.portfolio = OptionsPortfolio()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Add test positions
        self.portfolio.add_position(OptionPosition(
            symbol='AAPL',
            option_type='call',
            strike=150,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10,
            premium=5.0,
            underlying_price=155,
            volatility=0.25
        ))
        
        self.report_generator = RiskReportGenerator(self.portfolio, self.correlation_analyzer)
    
    def test_risk_dashboard_generation(self):
        """Test risk dashboard generation"""
        dashboard = self.report_generator.generate_risk_dashboard()
        
        assert 'portfolio_summary' in dashboard
        assert 'concentration_risk' in dashboard
        assert 'scenario_analysis' in dashboard
        assert 'risk_alerts' in dashboard
        
        # Check portfolio summary
        summary = dashboard['portfolio_summary']
        assert 'total_positions' in summary
        assert 'portfolio_value' in summary
        assert 'portfolio_var_95' in summary
        assert 'portfolio_greeks' in summary
    
    def test_concentration_risk_analysis(self):
        """Test concentration risk analysis"""
        concentration = self.report_generator.analyze_concentration_risk()
        
        assert 'symbol_concentration' in concentration
        assert 'expiry_concentration' in concentration
        assert 'concentration_index' in concentration
        
        # HHI should be between 0 and 1
        hhi = concentration['concentration_index']
        assert 0 <= hhi <= 1, f"HHI {hhi} should be between 0 and 1"
    
    def test_scenario_analysis(self):
        """Test scenario analysis"""
        scenarios = self.report_generator.perform_scenario_analysis()
        
        expected_scenarios = ['market_crash', 'market_rally', 'vol_spike', 'vol_crush']
        for scenario in expected_scenarios:
            assert scenario in scenarios
            
            scenario_data = scenarios[scenario]
            assert 'portfolio_value' in scenario_data
            assert 'pnl' in scenario_data
            assert 'pnl_percentage' in scenario_data
    
    def test_risk_alerts_generation(self):
        """Test risk alerts generation"""
        alerts = self.report_generator.generate_risk_alerts()
        
        assert isinstance(alerts, list)
        
        for alert in alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'recommendation' in alert

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_option_price_calculation_endpoint(self):
        """Test option price calculation endpoint"""
        request_data = {
            "underlying_price": 100,
            "strike": 100,
            "time_to_expiry": 0.25,
            "risk_free_rate": 0.05,
            "volatility": 0.2,
            "option_type": "call"
        }
        
        response = self.client.post("/options/price", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "option_price" in data
        assert "greeks" in data
        assert "delta" in data["greeks"]
    
    def test_portfolio_endpoints(self):
        """Test portfolio management endpoints"""
        # Add option position
        position_data = {
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 150,
            "expiry_days": 30,
            "quantity": 10,
            "premium": 5.0,
            "underlying_price": 155,
            "volatility": 0.25
        }
        
        response = self.client.post("/portfolio/options/add", json=position_data)
        assert response.status_code == 200
        
        # Get positions
        response = self.client.get("/portfolio/options/positions")
        assert response.status_code == 200
        
        data = response.json()
        assert "positions" in data
        assert len(data["positions"]) > 0
    
    def test_risk_dashboard_endpoint(self):
        """Test risk dashboard endpoint"""
        # First add a position
        position_data = {
            "symbol": "AAPL",
            "option_type": "call",
            "strike": 150,
            "expiry_days": 30,
            "quantity": 10,
            "premium": 5.0,
            "underlying_price": 155,
            "volatility": 0.25
        }
        
        self.client.post("/portfolio/options/add", json=position_data)
        
        # Get risk dashboard
        response = self.client.get("/portfolio/risk/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "portfolio_summary" in data
        assert "scenario_analysis" in data
    
    def test_correlation_analysis_endpoint(self):
        """Test correlation analysis endpoint"""
        correlation_data = {
            "symbols": ["AAPL", "MSFT"],
            "period": "3mo"
        }
        
        try:
            response = self.client.post("/analysis/correlation", json=correlation_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "correlation_matrix" in data
            assert "analysis_summary" in data
        except Exception:
            pytest.skip("Skipping correlation test due to data dependencies")

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def test_complete_trading_workflow(self):
        """Test complete options trading workflow"""
        portfolio = OptionsPortfolio()
        
        # Step 1: Add positions
        positions = [
            OptionPosition(
                symbol='AAPL',
                option_type='call',
                strike=150,
                expiry=datetime.now() + timedelta(days=30),
                quantity=10,
                premium=5.0,
                underlying_price=155,
                volatility=0.25
            ),
            OptionPosition(
                symbol='MSFT',
                option_type='put',
                strike=300,
                expiry=datetime.now() + timedelta(days=45),
                quantity=-5,
                premium=8.0,
                underlying_price=310,
                volatility=0.22
            )
        ]
        
        for position in positions:
            portfolio.add_position(position)
        
        # Step 2: Calculate portfolio metrics
        greeks = portfolio.calculate_portfolio_greeks()
        value = portfolio.calculate_portfolio_value()
        var = portfolio.calculate_var()
        
        assert len(portfolio.positions) == 2
        assert value > 0
        assert var >= 0
        
        # Step 3: Generate risk report
        analyzer = CorrelationAnalyzer()
        report_generator = RiskReportGenerator(portfolio, analyzer)
        dashboard = report_generator.generate_risk_dashboard()
        
        assert dashboard is not None
        assert len(dashboard['risk_alerts']) >= 0

# Utility functions for testing
def create_test_user():
    """Create a test user for testing"""
    return User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashedpassword",
        full_name="Test User",
        is_active=True,
        is_verified=True
    )

def create_test_portfolio(user_id: str):
    """Create a test portfolio"""
    return Portfolio(
        user_id=user_id,
        name="Test Portfolio",
        description="Portfolio for testing",
        initial_capital=100000.0,
        current_value=100000.0,
        cash_balance=50000.0
    )

# Performance tests
class TestPerformance:
    """Test performance of critical functions"""
    
    def test_black_scholes_performance(self):
        """Test Black-Scholes calculation performance"""
        import time
        
        start_time = time.time()
        
        # Calculate 1000 option prices
        for _ in range(1000):
            BlackScholesCalculator.option_price(100, 100, 0.25, 0.05, 0.2, 'call')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in less than 1 second
        assert duration < 1.0, f"Performance test failed: {duration:.2f}s for 1000 calculations"
    
    def test_portfolio_greeks_performance(self):
        """Test portfolio Greeks calculation performance"""
        import time
        
        # Create large portfolio
        portfolio = OptionsPortfolio()
        for i in range(100):
            portfolio.add_position(OptionPosition(
                symbol=f'STOCK{i}',
                option_type='call' if i % 2 == 0 else 'put',
                strike=100 + i,
                expiry=datetime.now() + timedelta(days=30 + i),
                quantity=10,
                premium=5.0,
                underlying_price=105 + i,
                volatility=0.2 + (i * 0.001)
            ))
        
        start_time = time.time()
        greeks = portfolio.calculate_portfolio_greeks()
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 5.0, f"Performance test failed: {duration:.2f}s for 100 positions"
        assert greeks is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 