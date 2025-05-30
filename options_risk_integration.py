"""
Options Trading and Risk Management Integration Module
Handles options pricing, Greeks calculation, portfolio risk, and correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf

@dataclass
class OptionPosition:
    """Represents an individual option position"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    quantity: int  # positive for long, negative for short
    premium: float
    underlying_price: float
    risk_free_rate: float = 0.05
    volatility: float = 0.2

@dataclass
class Greeks:
    """Option Greeks for risk analysis"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculation"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return BlackScholesCalculator.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def option_price(S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes"""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call') -> Greeks:
        """Calculate all Greeks for an option"""
        if T <= 0:
            return Greeks(0, 0, 0, 0, 0)
        
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
        else:
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r*T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r*T) * stats.norm.cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r*T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r*T) * stats.norm.cdf(-d2) / 100
        
        return Greeks(delta, gamma, theta, vega, rho)

class OptionsPortfolio:
    """Manages a portfolio of option positions with risk analysis"""
    
    def __init__(self):
        self.positions: List[OptionPosition] = []
        self.correlation_matrix = None
        self.risk_metrics = {}
    
    def add_position(self, position: OptionPosition):
        """Add an option position to the portfolio"""
        self.positions.append(position)
    
    def remove_position(self, index: int):
        """Remove a position by index"""
        if 0 <= index < len(self.positions):
            self.positions.pop(index)
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate aggregate Greeks for the entire portfolio"""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for position in self.positions:
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            greeks = BlackScholesCalculator.calculate_greeks(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            
            total_delta += greeks.delta * position.quantity
            total_gamma += greeks.gamma * position.quantity
            total_theta += greeks.theta * position.quantity
            total_vega += greeks.vega * position.quantity
            total_rho += greeks.rho * position.quantity
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'portfolio_theta': total_theta,
            'portfolio_vega': total_vega,
            'portfolio_rho': total_rho
        }
    
    def calculate_var(self, confidence_level=0.95, time_horizon=1) -> float:
        """Calculate Value at Risk for the options portfolio"""
        portfolio_value = self.calculate_portfolio_value()
        portfolio_greeks = self.calculate_portfolio_greeks()
        
        # Simplified VaR calculation using delta-normal method
        daily_volatility = 0.02  # Assume 2% daily volatility
        var_multiplier = stats.norm.ppf(1 - confidence_level)
        
        # Portfolio VaR considering delta exposure
        portfolio_var = abs(portfolio_greeks['portfolio_delta'] * 
                          portfolio_value * daily_volatility * var_multiplier * 
                          np.sqrt(time_horizon))
        
        return portfolio_var
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0
        for position in self.positions:
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            option_price = BlackScholesCalculator.option_price(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            total_value += option_price * position.quantity
        
        return total_value

class CorrelationAnalyzer:
    """Analyzes correlations between options, underlying assets, and risk factors"""
    
    def __init__(self):
        self.data_cache = {}
    
    def fetch_market_data(self, symbols: List[str], period='1y') -> pd.DataFrame:
        """Fetch market data for correlation analysis"""
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                data[symbol] = hist['Close'].pct_change().dropna()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        return pd.DataFrame(data).dropna()
    
    def calculate_asset_correlations(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for underlying assets"""
        price_data = self.fetch_market_data(symbols)
        correlation_matrix = price_data.corr()
        return correlation_matrix
    
    def calculate_options_correlation(self, portfolio: OptionsPortfolio) -> Dict:
        """Calculate correlations between option positions"""
        # Extract underlying symbols from portfolio
        symbols = list(set([pos.symbol for pos in portfolio.positions]))
        
        # Get price correlations
        price_correlations = self.calculate_asset_correlations(symbols)
        
        # Calculate Greeks correlations
        greeks_data = []
        for position in portfolio.positions:
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            greeks = BlackScholesCalculator.calculate_greeks(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            greeks_data.append([
                greeks.delta, greeks.gamma, greeks.theta, greeks.vega, greeks.rho
            ])
        
        greeks_df = pd.DataFrame(greeks_data, 
                               columns=['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'])
        greeks_correlations = greeks_df.corr()
        
        return {
            'price_correlations': price_correlations,
            'greeks_correlations': greeks_correlations,
            'cross_correlations': self.calculate_cross_correlations(portfolio)
        }
    
    def calculate_cross_correlations(self, portfolio: OptionsPortfolio) -> Dict:
        """Calculate correlations between different risk factors"""
        # This would analyze correlations between:
        # - Implied volatility and underlying price
        # - Interest rates and option values
        # - Time decay and volatility
        
        cross_corr_data = {}
        
        for position in portfolio.positions:
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            
            # Sensitivity analysis
            price_range = np.linspace(position.underlying_price * 0.8, 
                                    position.underlying_price * 1.2, 20)
            vol_range = np.linspace(0.1, 0.5, 20)
            
            price_sensitivity = []
            vol_sensitivity = []
            
            for price in price_range:
                option_price = BlackScholesCalculator.option_price(
                    price, position.strike, time_to_expiry,
                    position.risk_free_rate, position.volatility, position.option_type
                )
                price_sensitivity.append(option_price)
            
            for vol in vol_range:
                option_price = BlackScholesCalculator.option_price(
                    position.underlying_price, position.strike, time_to_expiry,
                    position.risk_free_rate, vol, position.option_type
                )
                vol_sensitivity.append(option_price)
            
            cross_corr_data[f"{position.symbol}_{position.option_type}"] = {
                'price_sensitivity': price_sensitivity,
                'vol_sensitivity': vol_sensitivity
            }
        
        return cross_corr_data

class RiskReportGenerator:
    """Generates comprehensive risk reports for options portfolios"""
    
    def __init__(self, portfolio: OptionsPortfolio, correlation_analyzer: CorrelationAnalyzer):
        self.portfolio = portfolio
        self.correlation_analyzer = correlation_analyzer
    
    def generate_risk_dashboard(self) -> Dict:
        """Generate comprehensive risk dashboard data"""
        portfolio_greeks = self.portfolio.calculate_portfolio_greeks()
        portfolio_var = self.portfolio.calculate_var()
        correlations = self.correlation_analyzer.calculate_options_correlation(self.portfolio)
        
        # Risk concentration analysis
        concentration_risk = self.analyze_concentration_risk()
        
        # Scenario analysis
        scenario_analysis = self.perform_scenario_analysis()
        
        return {
            'portfolio_summary': {
                'total_positions': len(self.portfolio.positions),
                'portfolio_value': self.portfolio.calculate_portfolio_value(),
                'portfolio_var_95': portfolio_var,
                'portfolio_greeks': portfolio_greeks
            },
            'correlation_analysis': correlations,
            'concentration_risk': concentration_risk,
            'scenario_analysis': scenario_analysis,
            'risk_alerts': self.generate_risk_alerts()
        }
    
    def analyze_concentration_risk(self) -> Dict:
        """Analyze portfolio concentration across different dimensions"""
        # By underlying asset
        symbol_exposure = {}
        for position in self.portfolio.positions:
            if position.symbol not in symbol_exposure:
                symbol_exposure[position.symbol] = 0
            
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            option_value = BlackScholesCalculator.option_price(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            symbol_exposure[position.symbol] += option_value * position.quantity
        
        # By expiry date
        expiry_exposure = {}
        for position in self.portfolio.positions:
            expiry_key = position.expiry.strftime('%Y-%m')
            if expiry_key not in expiry_exposure:
                expiry_exposure[expiry_key] = 0
            
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            option_value = BlackScholesCalculator.option_price(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            expiry_exposure[expiry_key] += option_value * position.quantity
        
        return {
            'symbol_concentration': symbol_exposure,
            'expiry_concentration': expiry_exposure,
            'concentration_index': self.calculate_herfindahl_index(symbol_exposure)
        }
    
    def calculate_herfindahl_index(self, exposures: Dict) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        total_exposure = sum(abs(exp) for exp in exposures.values())
        if total_exposure == 0:
            return 0
        
        hhi = sum((abs(exp) / total_exposure) ** 2 for exp in exposures.values())
        return hhi
    
    def perform_scenario_analysis(self) -> Dict:
        """Perform stress testing scenarios"""
        base_portfolio_value = self.portfolio.calculate_portfolio_value()
        
        scenarios = {
            'market_crash': {'price_shock': -0.3, 'vol_shock': 0.5},
            'market_rally': {'price_shock': 0.2, 'vol_shock': -0.2},
            'vol_spike': {'price_shock': 0, 'vol_shock': 1.0},
            'vol_crush': {'price_shock': 0, 'vol_shock': -0.5}
        }
        
        scenario_results = {}
        
        for scenario_name, shocks in scenarios.items():
            scenario_value = 0
            
            for position in self.portfolio.positions:
                time_to_expiry = (position.expiry - datetime.now()).days / 365.0
                
                # Apply shocks
                shocked_price = position.underlying_price * (1 + shocks['price_shock'])
                shocked_vol = max(0.01, position.volatility * (1 + shocks['vol_shock']))
                
                option_value = BlackScholesCalculator.option_price(
                    shocked_price, position.strike, time_to_expiry,
                    position.risk_free_rate, shocked_vol, position.option_type
                )
                scenario_value += option_value * position.quantity
            
            pnl = scenario_value - base_portfolio_value
            scenario_results[scenario_name] = {
                'portfolio_value': scenario_value,
                'pnl': pnl,
                'pnl_percentage': (pnl / base_portfolio_value) * 100 if base_portfolio_value != 0 else 0
            }
        
        return scenario_results
    
    def generate_risk_alerts(self) -> List[Dict]:
        """Generate risk alerts based on portfolio analysis"""
        alerts = []
        portfolio_greeks = self.portfolio.calculate_portfolio_greeks()
        
        # Delta exposure alert
        if abs(portfolio_greeks['portfolio_delta']) > 1000:
            alerts.append({
                'type': 'HIGH_DELTA_EXPOSURE',
                'severity': 'HIGH',
                'message': f"High delta exposure: {portfolio_greeks['portfolio_delta']:.2f}",
                'recommendation': 'Consider delta hedging'
            })
        
        # Gamma exposure alert
        if abs(portfolio_greeks['portfolio_gamma']) > 100:
            alerts.append({
                'type': 'HIGH_GAMMA_EXPOSURE',
                'severity': 'MEDIUM',
                'message': f"High gamma exposure: {portfolio_greeks['portfolio_gamma']:.2f}",
                'recommendation': 'Monitor for gamma scalping opportunities'
            })
        
        # Theta decay alert
        if portfolio_greeks['portfolio_theta'] < -100:
            alerts.append({
                'type': 'HIGH_THETA_DECAY',
                'severity': 'MEDIUM',
                'message': f"High theta decay: {portfolio_greeks['portfolio_theta']:.2f}",
                'recommendation': 'Time decay working against portfolio'
            })
        
        # Concentration risk alert
        concentration = self.analyze_concentration_risk()
        if concentration['concentration_index'] > 0.5:
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'severity': 'HIGH',
                'message': f"High concentration risk: HHI = {concentration['concentration_index']:.2f}",
                'recommendation': 'Diversify across more underlying assets'
            })
        
        return alerts

# Example usage and integration
def create_sample_portfolio():
    """Create a sample options portfolio for demonstration"""
    portfolio = OptionsPortfolio()
    
    # Add some sample positions
    portfolio.add_position(OptionPosition(
        symbol='AAPL',
        option_type='call',
        strike=150,
        expiry=datetime.now() + timedelta(days=30),
        quantity=10,
        premium=5.0,
        underlying_price=155,
        volatility=0.25
    ))
    
    portfolio.add_position(OptionPosition(
        symbol='AAPL',
        option_type='put',
        strike=145,
        expiry=datetime.now() + timedelta(days=30),
        quantity=-5,
        premium=3.0,
        underlying_price=155,
        volatility=0.25
    ))
    
    portfolio.add_position(OptionPosition(
        symbol='MSFT',
        option_type='call',
        strike=300,
        expiry=datetime.now() + timedelta(days=45),
        quantity=20,
        premium=8.0,
        underlying_price=310,
        volatility=0.22
    ))
    
    return portfolio

if __name__ == "__main__":
    # Demonstrate the integration
    portfolio = create_sample_portfolio()
    correlation_analyzer = CorrelationAnalyzer()
    report_generator = RiskReportGenerator(portfolio, correlation_analyzer)
    
    # Generate comprehensive risk report
    risk_dashboard = report_generator.generate_risk_dashboard()
    
    print("Options Portfolio Risk Dashboard")
    print("=" * 40)
    print(f"Portfolio Value: ${risk_dashboard['portfolio_summary']['portfolio_value']:.2f}")
    print(f"Portfolio VaR (95%): ${risk_dashboard['portfolio_summary']['portfolio_var_95']:.2f}")
    print(f"Portfolio Delta: {risk_dashboard['portfolio_summary']['portfolio_greeks']['portfolio_delta']:.2f}")
    print(f"Portfolio Gamma: {risk_dashboard['portfolio_summary']['portfolio_greeks']['portfolio_gamma']:.2f}")
    print(f"Portfolio Theta: {risk_dashboard['portfolio_summary']['portfolio_greeks']['portfolio_theta']:.2f}")
    
    print("\nRisk Alerts:")
    for alert in risk_dashboard['risk_alerts']:
        print(f"- {alert['type']}: {alert['message']}") 