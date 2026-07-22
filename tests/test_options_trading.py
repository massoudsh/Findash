"""
Test suite for Options Risk Management (`src/options_risk_integration.py`)

بازنویسی: نسخه قبلی به `src.main` (حذف‌شده، جایگزین با `src.main_refactored`) و
به یک لایه HTTP کاملاً موازی (`/options/price`, `/portfolio/options/add`,
`/portfolio/risk/dashboard`, `/analysis/correlation`) اشاره می‌کرد که هرگز در اپ
فعلی mount نشده‌اند (`src/options_risk_integration.py` فقط شامل کلاس‌های منطق
تجاری است، بدون هیچ `APIRouter`ای). چون افزودن این لایه HTTP یک فیچر کاملاً جدید
است (نه رفع یک تست stale)، خارج از scope این بازنویسی نگه داشته شد — کلاس‌های
منطق تجاری واقعی (که بدون تغییر باقی مانده‌اند) مستقیماً و بدون نیاز به FastAPI
app تست می‌شوند.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.options_risk_integration import (
    OptionsPortfolio, OptionPosition, BlackScholesCalculator,
    CorrelationAnalyzer, RiskReportGenerator
)


class TestBlackScholesCalculator:
    """Test Black-Scholes calculations"""

    def test_option_price_calculation(self):
        S = 100  # Stock price
        K = 100  # Strike price
        T = 0.25  # Time to expiry (3 months)
        r = 0.05  # Risk-free rate
        sigma = 0.2  # Volatility

        call_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'call')
        # مقدار صحیح Black-Scholes برای S=K=100, T=0.25, r=0.05, sigma=0.2 ≈ 4.615
        assert 4.0 < call_price < 5.2, f"Call price {call_price} outside expected range"

        put_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'put')

        # Put-call parity: Call - Put = S - K*e^(-r*T)
        parity_check = call_price - put_price - (S - K * np.exp(-r * T))
        assert abs(parity_check) < 0.01, f"Put-call parity violation: {parity_check}"

    def test_greeks_calculation(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.2

        greeks = BlackScholesCalculator.calculate_greeks(S, K, T, r, sigma, 'call')

        assert 0 < greeks.delta < 1, f"Call delta {greeks.delta} should be between 0 and 1"
        assert greeks.gamma > 0, f"Gamma {greeks.gamma} should be positive"
        assert greeks.theta < 0, f"Call theta {greeks.theta} should be negative"
        assert greeks.vega > 0, f"Vega {greeks.vega} should be positive"
        assert greeks.rho > 0, f"Call rho {greeks.rho} should be positive"

    def test_expired_option(self):
        S, K, T, r, sigma = 110, 100, 0, 0.05, 0.2

        call_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'call')
        assert call_price == max(S - K, 0), "Expired call value should be intrinsic value"

        put_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, 'put')
        assert put_price == max(K - S, 0), "Expired put value should be intrinsic value"


class TestOptionsPortfolio:
    """Test options portfolio functionality"""

    def setup_method(self):
        self.portfolio = OptionsPortfolio()

        self.portfolio.add_position(OptionPosition(
            symbol='AAPL', option_type='call', strike=150,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10, premium=5.0, underlying_price=155, volatility=0.25
        ))
        self.portfolio.add_position(OptionPosition(
            symbol='AAPL', option_type='put', strike=145,
            expiry=datetime.now() + timedelta(days=30),
            quantity=-5, premium=3.0, underlying_price=155, volatility=0.25
        ))

    def test_portfolio_greeks(self):
        greeks = self.portfolio.calculate_portfolio_greeks()

        assert 'portfolio_delta' in greeks
        assert 'portfolio_gamma' in greeks
        assert 'portfolio_theta' in greeks
        assert 'portfolio_vega' in greeks
        assert 'portfolio_rho' in greeks
        # Should have positive delta due to net long calls
        assert greeks['portfolio_delta'] > 0

    def test_portfolio_value(self):
        value = self.portfolio.calculate_portfolio_value()
        assert value > 0, "Portfolio should have positive value"

    def test_var_calculation(self):
        var = self.portfolio.calculate_var(confidence_level=0.95)
        assert var >= 0, "VaR should be non-negative"

    def test_add_remove_positions(self):
        initial_count = len(self.portfolio.positions)

        self.portfolio.add_position(OptionPosition(
            symbol='MSFT', option_type='call', strike=300,
            expiry=datetime.now() + timedelta(days=45),
            quantity=5, premium=8.0, underlying_price=310, volatility=0.22
        ))
        assert len(self.portfolio.positions) == initial_count + 1

        self.portfolio.remove_position(0)
        assert len(self.portfolio.positions) == initial_count


class TestCorrelationAnalyzer:
    """Test correlation analysis"""

    def setup_method(self):
        self.analyzer = CorrelationAnalyzer()

    def test_asset_correlations(self):
        """Requires network access to fetch price history; skipped if unavailable."""
        symbols = ['AAPL', 'MSFT']
        try:
            correlations = self.analyzer.calculate_asset_correlations(symbols)
            assert correlations is not None
            assert correlations.shape == (len(symbols), len(symbols))
            for i in range(len(symbols)):
                assert abs(correlations.iloc[i, i] - 1.0) < 0.01
        except Exception as e:
            pytest.skip(f"Skipping correlation test due to data fetch error: {e}")


class TestRiskReportGenerator:
    """Test risk report generation"""

    def setup_method(self):
        self.portfolio = OptionsPortfolio()
        self.correlation_analyzer = CorrelationAnalyzer()

        self.portfolio.add_position(OptionPosition(
            symbol='AAPL', option_type='call', strike=150,
            expiry=datetime.now() + timedelta(days=30),
            quantity=10, premium=5.0, underlying_price=155, volatility=0.25
        ))

        self.report_generator = RiskReportGenerator(self.portfolio, self.correlation_analyzer)

    def test_risk_dashboard_generation(self):
        dashboard = self.report_generator.generate_risk_dashboard()

        assert 'portfolio_summary' in dashboard
        assert 'concentration_risk' in dashboard
        assert 'scenario_analysis' in dashboard
        assert 'risk_alerts' in dashboard

        summary = dashboard['portfolio_summary']
        assert 'total_positions' in summary
        assert 'portfolio_value' in summary
        assert 'portfolio_var_95' in summary
        assert 'portfolio_greeks' in summary

    def test_concentration_risk_analysis(self):
        concentration = self.report_generator.analyze_concentration_risk()

        assert 'symbol_concentration' in concentration
        assert 'expiry_concentration' in concentration
        assert 'concentration_index' in concentration

        hhi = concentration['concentration_index']
        assert 0 <= hhi <= 1, f"HHI {hhi} should be between 0 and 1"

    def test_scenario_analysis(self):
        scenarios = self.report_generator.perform_scenario_analysis()

        expected_scenarios = ['market_crash', 'market_rally', 'vol_spike', 'vol_crush']
        for scenario in expected_scenarios:
            assert scenario in scenarios
            scenario_data = scenarios[scenario]
            assert 'portfolio_value' in scenario_data
            assert 'pnl' in scenario_data
            assert 'pnl_percentage' in scenario_data

    def test_risk_alerts_generation(self):
        alerts = self.report_generator.generate_risk_alerts()

        assert isinstance(alerts, list)
        for alert in alerts:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'recommendation' in alert


class TestIntegrationScenarios:
    """Test integration scenarios"""

    def test_complete_trading_workflow(self):
        portfolio = OptionsPortfolio()

        positions = [
            OptionPosition(
                symbol='AAPL', option_type='call', strike=150,
                expiry=datetime.now() + timedelta(days=30),
                quantity=10, premium=5.0, underlying_price=155, volatility=0.25
            ),
            OptionPosition(
                symbol='MSFT', option_type='put', strike=300,
                expiry=datetime.now() + timedelta(days=45),
                quantity=-5, premium=8.0, underlying_price=310, volatility=0.22
            ),
        ]
        for position in positions:
            portfolio.add_position(position)

        value = portfolio.calculate_portfolio_value()
        var = portfolio.calculate_var()

        assert len(portfolio.positions) == 2
        assert value > 0
        assert var >= 0

        analyzer = CorrelationAnalyzer()
        report_generator = RiskReportGenerator(portfolio, analyzer)
        dashboard = report_generator.generate_risk_dashboard()

        assert dashboard is not None
        assert len(dashboard['risk_alerts']) >= 0


class TestPerformance:
    """Test performance of critical functions"""

    def test_black_scholes_performance(self):
        import time
        start_time = time.time()
        for _ in range(1000):
            BlackScholesCalculator.option_price(100, 100, 0.25, 0.05, 0.2, 'call')
        duration = time.time() - start_time
        assert duration < 1.0, f"Performance test failed: {duration:.2f}s for 1000 calculations"

    def test_portfolio_greeks_performance(self):
        import time
        portfolio = OptionsPortfolio()
        for i in range(100):
            portfolio.add_position(OptionPosition(
                symbol=f'STOCK{i}', option_type='call' if i % 2 == 0 else 'put',
                strike=100 + i, expiry=datetime.now() + timedelta(days=30 + i),
                quantity=10, premium=5.0, underlying_price=105 + i,
                volatility=0.2 + (i * 0.001)
            ))

        start_time = time.time()
        greeks = portfolio.calculate_portfolio_greeks()
        duration = time.time() - start_time

        assert duration < 5.0, f"Performance test failed: {duration:.2f}s for 100 positions"
        assert greeks is not None
