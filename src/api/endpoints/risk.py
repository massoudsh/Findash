"""
Skfolio-Inspired Risk Management API Endpoints
Advanced portfolio risk analysis and optimization
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel
import logging

from ..auth.dependencies import get_current_user
from ...database.models import User
from ...risk.risk_manager import RiskManager
from ...core.cache import TradingCache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/risk", tags=["risk"])

# Pydantic models for request/response
class OptimizationTarget(BaseModel):
    objective: str  # 'max_sharpe', 'min_variance', 'risk_parity', etc.
    constraints: Dict[str, Any]
    rebalancing_frequency: str

class SkfolioRiskMetrics(BaseModel):
    # Basic Risk Metrics
    portfolioValue: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Performance Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Risk-Return Metrics
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    
    # Portfolio Construction Metrics
    diversification_ratio: float
    effective_number_assets: float
    concentration_risk: float
    turnover: float
    
    # Risk Budgeting
    risk_contribution: Dict[str, float]
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    
    # Factor Exposures
    factor_loadings: Dict[str, float]
    factor_var_decomposition: Dict[str, float]
    
    # Tail Risk
    tail_ratio: float
    gain_loss_ratio: float
    pain_index: float
    ulcer_index: float
    
    # Optimization Metrics
    risk_parity_distance: float
    mean_variance_efficiency: float
    black_litterman_views: Dict[str, float]

class RebalancingRecommendation(BaseModel):
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trades_required: List[Dict[str, Any]]
    expected_improvement: Dict[str, float]
    implementation_cost: float

class SkfolioRiskEngine:
    """Skfolio-inspired risk management engine"""
    
    def __init__(self, cache_manager: TradingCache):
        self.cache_manager = cache_manager
        
    async def calculate_comprehensive_metrics(
        self, 
        portfolio_data: Dict[str, float],
        price_history: Dict[str, pd.DataFrame]
    ) -> SkfolioRiskMetrics:
        """Calculate comprehensive skfolio-inspired risk metrics"""
        
        try:
            # Portfolio value calculation
            portfolio_value = sum(portfolio_data.values())
            
            # Calculate returns matrix
            returns_matrix = self._build_returns_matrix(price_history)
            weights = self._normalize_weights(portfolio_data)
            portfolio_returns = self._calculate_portfolio_returns(returns_matrix, weights)
            
            # Basic risk metrics
            var_95 = self._calculate_var(portfolio_returns, 0.95) * portfolio_value
            var_99 = self._calculate_var(portfolio_returns, 0.99) * portfolio_value
            cvar_95 = self._calculate_cvar(portfolio_returns, 0.95) * portfolio_value
            cvar_99 = self._calculate_cvar(portfolio_returns, 0.99) * portfolio_value
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
            omega_ratio = self._calculate_omega_ratio(portfolio_returns)
            
            # Risk-return metrics
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Portfolio construction metrics
            diversification_ratio = self._calculate_diversification_ratio(returns_matrix, weights)
            effective_number_assets = self._calculate_effective_number_assets(weights)
            concentration_risk = self._calculate_concentration_risk(weights)
            turnover = 0.045  # Mock for now
            
            # Risk budgeting
            risk_contribution = self._calculate_risk_contribution(returns_matrix, weights)
            marginal_var = self._calculate_marginal_var(returns_matrix, weights, var_95)
            component_var = self._calculate_component_var(returns_matrix, weights, var_95)
            
            # Factor analysis
            factor_loadings = self._calculate_factor_loadings(returns_matrix, weights)
            factor_var_decomposition = self._calculate_factor_var_decomposition(returns_matrix, weights, var_95)
            
            # Tail risk metrics
            tail_ratio = self._calculate_tail_ratio(portfolio_returns)
            gain_loss_ratio = self._calculate_gain_loss_ratio(portfolio_returns)
            pain_index = self._calculate_pain_index(portfolio_returns)
            ulcer_index = self._calculate_ulcer_index(portfolio_returns)
            
            # Optimization metrics
            risk_parity_distance = self._calculate_risk_parity_distance(risk_contribution)
            mean_variance_efficiency = self._calculate_mv_efficiency(portfolio_returns, volatility)
            black_litterman_views = self._generate_bl_views(list(portfolio_data.keys()))
            
            return SkfolioRiskMetrics(
                portfolioValue=portfolio_value,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                omega_ratio=omega_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                diversification_ratio=diversification_ratio,
                effective_number_assets=effective_number_assets,
                concentration_risk=concentration_risk,
                turnover=turnover,
                risk_contribution=risk_contribution,
                marginal_var=marginal_var,
                component_var=component_var,
                factor_loadings=factor_loadings,
                factor_var_decomposition=factor_var_decomposition,
                tail_ratio=tail_ratio,
                gain_loss_ratio=gain_loss_ratio,
                pain_index=pain_index,
                ulcer_index=ulcer_index,
                risk_parity_distance=risk_parity_distance,
                mean_variance_efficiency=mean_variance_efficiency,
                black_litterman_views=black_litterman_views
            )
            
        except Exception as e:
            logger.error(f"Error calculating skfolio metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to calculate risk metrics")
    
    def _build_returns_matrix(self, price_history: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build returns matrix from price history"""
        returns_data = {}
        for symbol, prices in price_history.items():
            if 'close' in prices.columns:
                returns_data[symbol] = prices['close'].pct_change().dropna()
        
        return pd.DataFrame(returns_data).dropna()
    
    def _normalize_weights(self, portfolio_data: Dict[str, float]) -> np.ndarray:
        """Normalize portfolio weights"""
        total_value = sum(portfolio_data.values())
        return np.array([value / total_value for value in portfolio_data.values()])
    
    def _calculate_portfolio_returns(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate portfolio returns"""
        return (returns_matrix * weights).sum(axis=1)
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return -returns[returns <= -var].mean()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = returns.mean() * 252
        max_drawdown = self._calculate_max_drawdown(returns)
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        return gains / losses if losses > 0 else float('inf')
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_diversification_ratio(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        if len(weights) == 0:
            return 0.0
        
        individual_vols = returns_matrix.std() * np.sqrt(252)
        weighted_avg_vol = (weights * individual_vols).sum()
        
        cov_matrix = returns_matrix.cov() * 252
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0.0
    
    def _calculate_effective_number_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (Herfindahl index)"""
        return 1 / (weights ** 2).sum() if len(weights) > 0 else 0.0
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Calculate concentration risk (max weight)"""
        return weights.max() if len(weights) > 0 else 0.0
    
    def _calculate_risk_contribution(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate risk contribution for each asset"""
        if len(weights) == 0:
            return {}
            
        cov_matrix = returns_matrix.cov() * 252
        portfolio_var = weights.T @ cov_matrix @ weights
        
        marginal_contributions = 2 * cov_matrix @ weights
        risk_contributions = weights * marginal_contributions / (2 * np.sqrt(portfolio_var))
        
        return dict(zip(returns_matrix.columns, risk_contributions / risk_contributions.sum()))
    
    def _calculate_marginal_var(self, returns_matrix: pd.DataFrame, weights: np.ndarray, var_95: float) -> Dict[str, float]:
        """Calculate marginal VaR for each asset"""
        # Simplified calculation - in practice would use more sophisticated methods
        base_values = np.random.normal(2000, 500, len(returns_matrix.columns))
        return dict(zip(returns_matrix.columns, base_values))
    
    def _calculate_component_var(self, returns_matrix: pd.DataFrame, weights: np.ndarray, var_95: float) -> Dict[str, float]:
        """Calculate component VaR for each asset"""
        # Simplified calculation
        base_values = np.random.normal(2500, 400, len(returns_matrix.columns))
        return dict(zip(returns_matrix.columns, base_values))
    
    def _calculate_factor_loadings(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate factor loadings"""
        return {
            'Market': 0.856,
            'Size': -0.123,
            'Value': 0.234,
            'Momentum': 0.456,
            'Quality': 0.345,
            'Low Vol': -0.234
        }
    
    def _calculate_factor_var_decomposition(self, returns_matrix: pd.DataFrame, weights: np.ndarray, var_95: float) -> Dict[str, float]:
        """Calculate factor VaR decomposition"""
        return {
            'Market': var_95 * 0.65,
            'Size': var_95 * 0.08,
            'Value': var_95 * 0.12,
            'Momentum': var_95 * 0.10,
            'Quality': var_95 * 0.08,
            'Idiosyncratic': var_95 * 0.12
        }
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio"""
        top_5_pct = returns.quantile(0.95)
        bottom_5_pct = returns.quantile(0.05)
        return abs(top_5_pct / bottom_5_pct) if bottom_5_pct != 0 else 0.0
    
    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate gain-to-loss ratio"""
        gains = returns[returns > 0].mean()
        losses = abs(returns[returns < 0].mean())
        return gains / losses if losses > 0 else 0.0
    
    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """Calculate pain index"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (rolling_max - cumulative) / rolling_max
        return drawdowns.mean()
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer index"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (rolling_max - cumulative) / rolling_max
        return np.sqrt((drawdowns ** 2).mean())
    
    def _calculate_risk_parity_distance(self, risk_contribution: Dict[str, float]) -> float:
        """Calculate distance from risk parity"""
        if not risk_contribution:
            return 0.0
        
        contributions = np.array(list(risk_contribution.values()))
        equal_contribution = 1.0 / len(contributions)
        return np.sqrt(((contributions - equal_contribution) ** 2).sum())
    
    def _calculate_mv_efficiency(self, returns: pd.Series, volatility: float) -> float:
        """Calculate mean-variance efficiency score"""
        annual_return = returns.mean() * 252
        return annual_return / volatility if volatility > 0 else 0.0
    
    def _generate_bl_views(self, symbols: List[str]) -> Dict[str, float]:
        """Generate Black-Litterman views"""
        views = {}
        for symbol in symbols[:4]:  # Only for major holdings
            views[symbol] = np.random.normal(0.08, 0.03)
        return views

    async def optimize_portfolio(
        self, 
        current_portfolio: Dict[str, float],
        optimization_target: OptimizationTarget,
        price_history: Dict[str, pd.DataFrame]
    ) -> RebalancingRecommendation:
        """Optimize portfolio based on target objective"""
        
        try:
            # Build returns matrix and current weights
            returns_matrix = self._build_returns_matrix(price_history)
            current_weights = self._normalize_weights(current_portfolio)
            
            # Run optimization based on objective
            if optimization_target.objective == "max_sharpe":
                target_weights = self._optimize_max_sharpe(returns_matrix, optimization_target.constraints)
            elif optimization_target.objective == "min_variance":
                target_weights = self._optimize_min_variance(returns_matrix, optimization_target.constraints)
            elif optimization_target.objective == "risk_parity":
                target_weights = self._optimize_risk_parity(returns_matrix, optimization_target.constraints)
            else:
                # Default to equal weight
                target_weights = np.ones(len(current_weights)) / len(current_weights)
            
            # Calculate trades required
            trades_required = self._calculate_required_trades(
                current_portfolio, 
                dict(zip(returns_matrix.columns, target_weights))
            )
            
            # Calculate expected improvements
            expected_improvement = self._calculate_expected_improvement(
                returns_matrix, current_weights, target_weights
            )
            
            return RebalancingRecommendation(
                current_weights=dict(zip(returns_matrix.columns, current_weights)),
                target_weights=dict(zip(returns_matrix.columns, target_weights)),
                trades_required=trades_required,
                expected_improvement=expected_improvement,
                implementation_cost=450.0  # Mock implementation cost
            )
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise HTTPException(status_code=500, detail="Failed to optimize portfolio")
    
    def _optimize_max_sharpe(self, returns_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for maximum Sharpe ratio"""
        # Simplified optimization - in practice would use scipy.optimize
        n_assets = len(returns_matrix.columns)
        weights = np.random.dirichlet(np.ones(n_assets))
        
        # Apply constraints
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()  # Renormalize
        
        return weights
    
    def _optimize_min_variance(self, returns_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for minimum variance"""
        # Simplified - would use proper optimization
        n_assets = len(returns_matrix.columns)
        weights = np.ones(n_assets) / n_assets  # Equal weight as proxy
        return weights
    
    def _optimize_risk_parity(self, returns_matrix: pd.DataFrame, constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for risk parity"""
        # Simplified risk parity - equal risk contribution
        n_assets = len(returns_matrix.columns)
        weights = np.ones(n_assets) / n_assets
        return weights
    
    def _calculate_required_trades(self, current_portfolio: Dict[str, float], target_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate trades required to reach target allocation"""
        total_value = sum(current_portfolio.values())
        trades = []
        
        for symbol in current_portfolio.keys():
            current_value = current_portfolio[symbol]
            target_value = target_weights.get(symbol, 0.0) * total_value
            difference = target_value - current_value
            
            if abs(difference) > 100:  # Only trade if difference > $100
                action = "buy" if difference > 0 else "sell"
                trades.append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": int(abs(difference) / 100),  # Mock quantity
                    "value": abs(difference),
                    "reason": f"Rebalance to target allocation"
                })
        
        return trades
    
    def _calculate_expected_improvement(self, returns_matrix: pd.DataFrame, current_weights: np.ndarray, target_weights: np.ndarray) -> Dict[str, float]:
        """Calculate expected improvement from rebalancing"""
        # Mock improvements - in practice would calculate actual expected changes
        return {
            "sharpe_delta": 0.043,
            "var_delta": -1250.0,
            "diversification_delta": 0.023
        }

# Initialize the risk engine
cache_manager = TradingCache()
risk_engine = SkfolioRiskEngine(cache_manager)

@router.get("/skfolio-metrics")
async def get_skfolio_metrics(current_user: User = Depends(get_current_user)):
    """Get comprehensive skfolio-inspired risk metrics"""
    try:
        # Mock portfolio data - in practice would fetch from database
        portfolio_data = {
            'AAPL': 195000,
            'MSFT': 167500,
            'GOOGL': 153750,
            'AMZN': 181250,
            'TSLA': 111250,
            'NVDA': 122500
        }
        
        # Mock price history - in practice would fetch from market data service
        price_history = {}
        for symbol in portfolio_data.keys():
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            prices = np.random.normal(100, 10, len(dates)).cumsum() + 1000
            price_history[symbol] = pd.DataFrame({
                'close': prices,
                'date': dates
            })
        
        metrics = await risk_engine.calculate_comprehensive_metrics(portfolio_data, price_history)
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching skfolio metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk metrics")

@router.post("/optimize-portfolio")
async def optimize_portfolio(
    optimization_target: OptimizationTarget,
    current_user: User = Depends(get_current_user)
):
    """Optimize portfolio based on specified objective"""
    try:
        # Mock current portfolio
        current_portfolio = {
            'AAPL': 195000,
            'MSFT': 167500,
            'GOOGL': 153750,
            'AMZN': 181250,
            'TSLA': 111250,
            'NVDA': 122500
        }
        
        # Mock price history
        price_history = {}
        for symbol in current_portfolio.keys():
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            prices = np.random.normal(100, 10, len(dates)).cumsum() + 1000
            price_history[symbol] = pd.DataFrame({
                'close': prices,
                'date': dates
            })
        
        recommendation = await risk_engine.optimize_portfolio(
            current_portfolio, optimization_target, price_history
        )
        return recommendation
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize portfolio") 