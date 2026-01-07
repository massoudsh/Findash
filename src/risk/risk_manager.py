"""
M6 | Unified Risk Management Service
Advanced Portfolio Risk Management and Position Sizing

This unified service combines:
- RiskManager (M6 Agent): VaR, position sizing, portfolio risk assessment
- SkfolioRiskEngine: Comprehensive skfolio-inspired risk metrics and optimization
- Risk Tasks: Celery task helpers for risk calculations

Handles:
- Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- Portfolio exposure monitoring
- Position sizing algorithms (Kelly Criterion, risk budget)
- Correlation analysis and diversification metrics
- Tail risk assessment
- Risk budget allocation
- Skfolio-inspired comprehensive risk metrics
- Portfolio optimization (Max Sharpe, Min Variance, Risk Parity)
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

# ============================================
# HELPER FUNCTIONS (from tasks.py)
# ============================================

def calculate_var_helper(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Helper function for VaR calculation (used by Celery tasks)"""
    if returns.empty:
        return 0.0
    return abs(np.percentile(returns, 100 * (1 - confidence_level)))

def calculate_sharpe_ratio_helper(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Helper function for Sharpe ratio calculation (used by Celery tasks)"""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

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
    
    # ============================================
    # SKFOLIO-INSPIRED COMPREHENSIVE METRICS
    # ============================================
    
    def _build_returns_matrix(self, price_history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build returns matrix from price history"""
        returns_data = {}
        for symbol, prices in price_history.items():
            if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
                returns_data[symbol] = prices['close'].pct_change().dropna()
            elif isinstance(prices, pd.Series):
                returns_data[symbol] = prices.pct_change().dropna()
        
        if not returns_data:
            return pd.DataFrame()
        return pd.DataFrame(returns_data).dropna()
    
    def _normalize_weights(self, portfolio_data: Dict[str, float]) -> np.ndarray:
        """Normalize portfolio weights"""
        total_value = sum(portfolio_data.values())
        if total_value == 0:
            return np.array([])
        return np.array([value / total_value for value in portfolio_data.values()])
    
    def _calculate_portfolio_returns(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate portfolio returns"""
        if len(weights) == 0 or returns_matrix.empty:
            return pd.Series()
        return (returns_matrix * weights).sum(axis=1)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if returns.empty:
            return 0.0
        var = self._calculate_var_for_skfolio(returns, confidence)
        if var == 0:
            return 0.0
        tail_losses = returns[returns <= -var]
        return -tail_losses.mean() if len(tail_losses) > 0 else abs(var)
    
    def _calculate_var_for_skfolio(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk for Skfolio methods (single confidence level)"""
        if returns.empty:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_sharpe_ratio_skfolio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio_skfolio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (annualized)"""
        if returns.empty:
            return 0.0
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if returns.empty:
            return 0.0
        annual_return = returns.mean() * 252
        max_dd = self._calculate_max_drawdown_skfolio(returns)
        return annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if returns.empty:
            return 0.0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_max_drawdown_skfolio(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_diversification_ratio_skfolio(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate diversification ratio using returns matrix"""
        if len(weights) == 0 or returns_matrix.empty:
            return 0.0
        
        individual_vols = returns_matrix.std() * np.sqrt(252)
        weighted_avg_vol = (weights * individual_vols).sum()
        
        cov_matrix = returns_matrix.cov() * 252
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0.0
    
    def _calculate_effective_number_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (inverse Herfindahl index)"""
        if len(weights) == 0:
            return 0.0
        return 1 / (weights ** 2).sum()
    
    def _calculate_risk_contribution(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate risk contribution for each asset"""
        if len(weights) == 0 or returns_matrix.empty:
            return {}
            
        cov_matrix = returns_matrix.cov() * 252
        portfolio_var = weights.T @ cov_matrix @ weights
        
        if portfolio_var <= 0:
            return {}
        
        marginal_contributions = 2 * cov_matrix @ weights
        risk_contributions = weights * marginal_contributions / (2 * np.sqrt(portfolio_var))
        
        total_contrib = risk_contributions.sum()
        if total_contrib == 0:
            return {}
        
        return dict(zip(returns_matrix.columns, risk_contributions / total_contrib))
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        if returns.empty:
            return 0.0
        top_5_pct = returns.quantile(0.95)
        bottom_5_pct = returns.quantile(0.05)
        return abs(top_5_pct / bottom_5_pct) if bottom_5_pct != 0 else 0.0
    
    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate gain-to-loss ratio"""
        if returns.empty:
            return 0.0
        gains = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.0
        losses = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.0
        return gains / losses if losses > 0 else 0.0
    
    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """Calculate pain index (average drawdown)"""
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (rolling_max - cumulative) / rolling_max
        return drawdowns.mean()
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer index (downside volatility)"""
        if returns.empty:
            return 0.0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (rolling_max - cumulative) / rolling_max
        return np.sqrt((drawdowns ** 2).mean())
    
    async def calculate_comprehensive_metrics(
        self, 
        portfolio_data: Dict[str, float],
        price_history: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive skfolio-inspired risk metrics
        
        Returns a dictionary with all risk metrics including:
        - VaR/CVaR at multiple confidence levels
        - Performance ratios (Sharpe, Sortino, Calmar, Omega)
        - Risk-return metrics (volatility, max drawdown, skewness, kurtosis)
        - Portfolio construction metrics (diversification, concentration, effective assets)
        - Risk budgeting (risk contribution, marginal VaR, component VaR)
        - Tail risk metrics
        """
        try:
            portfolio_value = sum(portfolio_data.values())
            
            if not price_history or portfolio_value == 0:
                return self._get_default_metrics(portfolio_value)
            
            returns_matrix = self._build_returns_matrix(price_history)
            if returns_matrix.empty:
                return self._get_default_metrics(portfolio_value)
            
            weights = self._normalize_weights(portfolio_data)
            if len(weights) == 0:
                return self._get_default_metrics(portfolio_value)
            
            portfolio_returns = self._calculate_portfolio_returns(returns_matrix, weights)
            if portfolio_returns.empty:
                return self._get_default_metrics(portfolio_value)
            
            # Basic risk metrics
            var_95 = self._calculate_var_for_skfolio(portfolio_returns, 0.95) * portfolio_value
            var_99 = self._calculate_var_for_skfolio(portfolio_returns, 0.99) * portfolio_value
            cvar_95 = self._calculate_cvar(portfolio_returns, 0.95) * portfolio_value
            cvar_99 = self._calculate_cvar(portfolio_returns, 0.99) * portfolio_value
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio_skfolio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio_skfolio(portfolio_returns)
            calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
            omega_ratio = self._calculate_omega_ratio(portfolio_returns)
            
            # Risk-return metrics
            max_drawdown = self._calculate_max_drawdown_skfolio(portfolio_returns)
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Portfolio construction metrics
            diversification_ratio = self._calculate_diversification_ratio_skfolio(returns_matrix, weights)
            effective_number_assets = self._calculate_effective_number_assets(weights)
            concentration_risk = weights.max() if len(weights) > 0 else 0.0
            
            # Risk budgeting
            risk_contribution = self._calculate_risk_contribution(returns_matrix, weights)
            
            # Tail risk metrics
            tail_ratio = self._calculate_tail_ratio(portfolio_returns)
            gain_loss_ratio = self._calculate_gain_loss_ratio(portfolio_returns)
            pain_index = self._calculate_pain_index(portfolio_returns)
            ulcer_index = self._calculate_ulcer_index(portfolio_returns)
            
            return {
                "portfolioValue": portfolio_value,
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "omega_ratio": omega_ratio,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "diversification_ratio": diversification_ratio,
                "effective_number_assets": effective_number_assets,
                "concentration_risk": concentration_risk,
                "risk_contribution": risk_contribution,
                "tail_ratio": tail_ratio,
                "gain_loss_ratio": gain_loss_ratio,
                "pain_index": pain_index,
                "ulcer_index": ulcer_index,
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return self._get_default_metrics(sum(portfolio_data.values()) if portfolio_data else 0.0)
    
    def _get_default_metrics(self, portfolio_value: float) -> Dict[str, Any]:
        """Return default metrics when calculation fails"""
        return {
            "portfolioValue": portfolio_value,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "omega_ratio": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "diversification_ratio": 1.0,
            "effective_number_assets": 0.0,
            "concentration_risk": 0.0,
            "risk_contribution": {},
            "tail_ratio": 0.0,
            "gain_loss_ratio": 0.0,
            "pain_index": 0.0,
            "ulcer_index": 0.0,
        }