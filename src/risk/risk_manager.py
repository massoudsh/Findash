"""
M6 | Risk Management Agent
Advanced Portfolio Risk Management and Position Sizing

This agent handles:
- Value at Risk (VaR) calculations
- Portfolio exposure monitoring
- Position sizing algorithms
- Correlation analysis
- Tail risk assessment
- Risk budget allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..core.cache import TradingCache
from ..core.assets_config import AssetsConfig
from ..database.postgres_connection import get_db

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_size: float
    market_value: float
    var_1d: float
    var_5d: float
    var_10d: float
    expected_shortfall: float
    beta: float
    correlation_to_portfolio: float
    max_loss_scenario: float
    confidence_level: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_value: float
    total_var: float
    diversification_ratio: float
    concentration_risk: float
    sector_exposure: Dict[str, float]
    currency_exposure: Dict[str, float]
    risk_level: RiskLevel
    risk_budget_utilization: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    tail_ratio: float

@dataclass
class RiskBudget:
    """Risk budget allocation"""
    total_budget: float
    strategy_allocations: Dict[str, float]
    sector_limits: Dict[str, float]
    currency_limits: Dict[str, float]
    max_position_size: float
    max_concentration: float
    var_limit: float

class RiskManager:
    """M6 - Advanced Risk Management Agent"""
    
    def __init__(self, cache_manager: TradingCache):
        self.cache_manager = cache_manager
        self.confidence_levels = [0.95, 0.99, 0.999]
        self.lookback_periods = [21, 63, 252]  # 1M, 3M, 1Y
        
        # Risk parameters
        self.max_portfolio_var = 0.02  # 2% daily VaR limit
        self.max_position_weight = 0.10  # 10% max position
        self.max_sector_weight = 0.25  # 25% max sector
        self.target_sharpe = 1.5
        
    async def assess_portfolio_risk(
        self, 
        portfolio: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> PortfolioRisk:
        """Comprehensive portfolio risk assessment"""
        try:
            # Calculate individual position risks
            position_risks = []
            total_value = sum(portfolio.values())
            
            for symbol, position_value in portfolio.items():
                if symbol in market_data:
                    pos_risk = await self._calculate_position_risk(
                        symbol, position_value, market_data[symbol], total_value
                    )
                    position_risks.append(pos_risk)
            
            # Portfolio-level calculations
            portfolio_var = await self._calculate_portfolio_var(position_risks, market_data)
            diversification_ratio = self._calculate_diversification_ratio(position_risks)
            concentration_risk = self._calculate_concentration_risk(portfolio)
            
            # Sector and currency exposure
            sector_exposure = await self._calculate_sector_exposure(portfolio)
            currency_exposure = await self._calculate_currency_exposure(portfolio)
            
            # Performance metrics
            performance_metrics = await self._calculate_performance_metrics(
                portfolio, market_data
            )
            
            # Risk level determination
            risk_level = self._determine_risk_level(portfolio_var, concentration_risk)
            
            risk_assessment = PortfolioRisk(
                total_value=total_value,
                total_var=portfolio_var,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                sector_exposure=sector_exposure,
                currency_exposure=currency_exposure,
                risk_level=risk_level,
                risk_budget_utilization=portfolio_var / self.max_portfolio_var,
                max_drawdown=performance_metrics['max_drawdown'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                tail_ratio=performance_metrics['tail_ratio']
            )
            
            # Cache the result
            await self.cache_manager.set(
                f"portfolio_risk:{hash(str(sorted(portfolio.items())))}",
                risk_assessment.__dict__,
                expire=300  # 5 minutes
            )
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            raise
    
    async def _calculate_position_risk(
        self,
        symbol: str,
        position_value: float,
        price_data: pd.DataFrame,
        total_portfolio_value: float
    ) -> PositionRisk:
        """Calculate risk metrics for individual position"""
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # VaR calculations for different horizons
        var_1d = self._calculate_var(returns, 1, 0.95)
        var_5d = self._calculate_var(returns, 5, 0.95)
        var_10d = self._calculate_var(returns, 10, 0.95)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = self._calculate_expected_shortfall(returns, 0.95)
        
        # Beta calculation (vs market proxy)
        beta = await self._calculate_beta(symbol, returns)
        
        # Maximum loss scenario (stress test)
        max_loss_scenario = self._calculate_stress_scenario(returns, position_value)
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_value / total_portfolio_value,
            market_value=position_value,
            var_1d=var_1d * position_value,
            var_5d=var_5d * position_value,
            var_10d=var_10d * position_value,
            expected_shortfall=expected_shortfall * position_value,
            beta=beta,
            correlation_to_portfolio=0.0,  # Will be calculated at portfolio level
            max_loss_scenario=max_loss_scenario,
            confidence_level=0.95
        )
    
    def _calculate_var(self, returns: pd.Series, horizon: int, confidence: float) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(returns) < 30:  # Insufficient data
            return 0.05  # Conservative 5% VaR
        
        # Scale for time horizon
        scaled_returns = returns * np.sqrt(horizon)
        
        # Historical VaR
        var_percentile = (1 - confidence) * 100
        var = np.percentile(scaled_returns, var_percentile)
        
        return abs(var)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) < 30:
            return 0.07  # Conservative ES
        
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return abs(var_threshold)
        
        return abs(tail_losses.mean())
    
    async def _calculate_beta(self, symbol: str, returns: pd.Series) -> float:
        """Calculate beta vs market benchmark"""
        try:
            # Get market proxy data (SPY as default)
            market_data = await self.cache_manager.get("market_data:SPY")
            if not market_data:
                return 1.0  # Default beta
            
            market_returns = pd.Series(market_data.get('returns', [])).pct_change().dropna()
            
            # Align series
            min_length = min(len(returns), len(market_returns))
            returns_aligned = returns.tail(min_length)
            market_aligned = market_returns.tail(min_length)
            
            # Calculate beta
            covariance = np.cov(returns_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            logger.warning(f"Beta calculation failed for {symbol}: {e}")
            return 1.0
    
    def _calculate_stress_scenario(self, returns: pd.Series, position_value: float) -> float:
        """Calculate maximum loss in stress scenario (99.9th percentile)"""
        if len(returns) < 100:
            return position_value * 0.20  # 20% stress loss
        
        # Use 99.9th percentile worst case
        worst_case_return = np.percentile(returns, 0.1)
        max_loss = abs(worst_case_return * position_value)
        
        return max_loss
    
    async def _calculate_portfolio_var(
        self,
        position_risks: List[PositionRisk],
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate portfolio-level VaR considering correlations"""
        
        if not position_risks:
            return 0.0
        
        # Individual VaRs
        individual_vars = np.array([pos.var_1d for pos in position_risks])
        
        # Calculate correlation matrix
        symbols = [pos.symbol for pos in position_risks]
        correlation_matrix = await self._calculate_correlation_matrix(symbols, market_data)
        
        # Portfolio VaR with correlation
        portfolio_var = np.sqrt(
            individual_vars.T @ correlation_matrix @ individual_vars
        )
        
        return portfolio_var
    
    async def _calculate_correlation_matrix(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Calculate correlation matrix for portfolio assets"""
        
        # Get returns for all symbols
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol]['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            # Return identity matrix if insufficient data
            return np.eye(len(symbols))
        
        # Align all series to same length
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {
            symbol: returns.tail(min_length) 
            for symbol, returns in returns_data.items()
        }
        
        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(aligned_returns)
        correlation_matrix = returns_df.corr().values
        
        # Fill NaN with 0 correlation
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        return correlation_matrix
    
    def _calculate_diversification_ratio(self, position_risks: List[PositionRisk]) -> float:
        """Calculate portfolio diversification ratio"""
        if not position_risks:
            return 1.0
        
        # Weighted average of individual volatilities
        weights = np.array([pos.position_size for pos in position_risks])
        individual_vols = np.array([pos.var_1d / pos.market_value for pos in position_risks])
        
        weighted_avg_vol = np.sum(weights * individual_vols)
        
        # Portfolio volatility (simplified)
        portfolio_vol = np.sqrt(np.sum((weights * individual_vols) ** 2))
        
        if portfolio_vol == 0:
            return 1.0
        
        diversification_ratio = weighted_avg_vol / portfolio_vol
        return diversification_ratio
    
    def _calculate_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl index"""
        total_value = sum(portfolio.values())
        
        if total_value == 0:
            return 0.0
        
        weights = [value / total_value for value in portfolio.values()]
        herfindahl_index = sum(w ** 2 for w in weights)
        
        # Normalize (1 = max concentration, 0 = perfect diversification)
        n_assets = len(portfolio)
        if n_assets <= 1:
            return 1.0
        
        normalized_concentration = (herfindahl_index - 1/n_assets) / (1 - 1/n_assets)
        return normalized_concentration
    
    async def _calculate_sector_exposure(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposure breakdown"""
        # Use centralized sector mapping from AssetsConfig
        sector_mapping = AssetsConfig.SECTOR_MAPPING
        
        total_value = sum(portfolio.values())
        sector_exposure = {}
        
        for symbol, value in portfolio.items():
            sector = sector_mapping.get(symbol, 'Other')
            weight = value / total_value if total_value > 0 else 0
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        return sector_exposure
    
    async def _calculate_currency_exposure(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calculate currency exposure breakdown"""
        # Use centralized currency mapping from AssetsConfig
        currency_mapping = AssetsConfig.CURRENCY_MAPPING
        
        total_value = sum(portfolio.values())
        currency_exposure = {'USD': 1.0}  # Simplified - assume all USD
        
        return currency_exposure
    
    async def _calculate_performance_metrics(
        self,
        portfolio: Dict[str, float],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        # Simplified calculation - in practice, use actual portfolio returns
        default_metrics = {
            'max_drawdown': 0.15,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'tail_ratio': 0.8
        }
        
        return default_metrics
    
    def _determine_risk_level(self, portfolio_var: float, concentration_risk: float) -> RiskLevel:
        """Determine overall portfolio risk level"""
        
        # Risk thresholds
        if portfolio_var > 0.03 or concentration_risk > 0.8:
            return RiskLevel.CRITICAL
        elif portfolio_var > 0.02 or concentration_risk > 0.6:
            return RiskLevel.HIGH
        elif portfolio_var > 0.01 or concentration_risk > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def calculate_optimal_position_size(
        self,
        symbol: str,
        expected_return: float,
        confidence: float,
        risk_budget: float,
        current_portfolio: Dict[str, float]
    ) -> float:
        """Calculate optimal position size using Kelly Criterion and risk constraints"""
        
        try:
            # Get historical volatility
            cache_key = f"price_data:{symbol}"
            price_data = await self.cache_manager.get(cache_key)
            
            if not price_data:
                return risk_budget * 0.1  # Conservative 10% of risk budget
            
            returns = pd.Series(price_data.get('returns', [])).pct_change().dropna()
            
            if len(returns) < 30:
                return risk_budget * 0.1
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Kelly Criterion
            if volatility > 0:
                kelly_fraction = expected_return / (volatility ** 2)
            else:
                kelly_fraction = 0.1
            
            # Apply constraints
            kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Max 25% Kelly
            
            # Risk budget constraint
            position_size = min(kelly_fraction * risk_budget, risk_budget * 0.2)
            
            # Portfolio concentration constraint
            total_portfolio_value = sum(current_portfolio.values())
            max_position_value = total_portfolio_value * self.max_position_weight
            
            position_size = min(position_size, max_position_value)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return risk_budget * 0.05  # Very conservative fallback
    
    async def get_risk_alerts(self, portfolio: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate risk alerts for current portfolio"""
        alerts = []
        
        # Check concentration risk
        total_value = sum(portfolio.values())
        for symbol, value in portfolio.items():
            weight = value / total_value if total_value > 0 else 0
            
            if weight > self.max_position_weight:
                alerts.append({
                    'type': 'concentration',
                    'severity': 'high',
                    'message': f'{symbol} position ({weight:.1%}) exceeds maximum weight ({self.max_position_weight:.1%})',
                    'symbol': symbol,
                    'current_weight': weight,
                    'max_weight': self.max_position_weight
                })
        
        # Check sector concentration
        sector_exposure = await self._calculate_sector_exposure(portfolio)
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_weight:
                alerts.append({
                    'type': 'sector_concentration',
                    'severity': 'medium',
                    'message': f'{sector} sector exposure ({exposure:.1%}) exceeds limit ({self.max_sector_weight:.1%})',
                    'sector': sector,
                    'current_exposure': exposure,
                    'max_exposure': self.max_sector_weight
                })
        
        return alerts
    
    async def stress_test_portfolio(
        self,
        portfolio: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Run stress tests on portfolio"""
        
        stress_results = {}
        
        # Default stress scenarios
        default_scenarios = {
            'market_crash': {'SPY': -0.20, 'default': -0.15},
            'interest_rate_shock': {'financials': 0.10, 'reits': -0.15, 'default': -0.05},
            'crypto_collapse': {'BTC-USD': -0.50, 'ETH-USD': -0.45, 'TRX-USD': -0.40, 'LINK-USD': -0.35, 'CAKE-USD': -0.45, 'default': 0.0},
            'stablecoin_depeg': {'USDT-USD': -0.05, 'USDC-USD': -0.02, 'default': 0.0},
            'tech_bubble': {'technology': -0.30, 'default': -0.10},
            'commodity_crash': {'GLD': -0.25, 'SLV': -0.30, 'default': 0.0}
        }
        
        test_scenarios = scenarios or default_scenarios
        
        for scenario_name, shocks in test_scenarios.items():
            portfolio_pnl = 0.0
            
            for symbol, position_value in portfolio.items():
                # Apply shock based on symbol or default
                shock = shocks.get(symbol, shocks.get('default', 0.0))
                position_pnl = position_value * shock
                portfolio_pnl += position_pnl
            
            stress_results[scenario_name] = portfolio_pnl
        
        return stress_results 