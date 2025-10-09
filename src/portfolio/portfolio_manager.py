"""
Portfolio Management System
Advanced Portfolio Optimization and Position Management

This module handles:
- Portfolio construction and optimization
- Dynamic rebalancing
- Position sizing and allocation
- Performance attribution
- Risk-adjusted returns
- Real-time portfolio monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy import stats
import json

from ..core.cache import CacheManager
from ..database.postgres_connection import get_db
from ..risk.risk_manager import RiskManager, PortfolioRisk
from ..strategies.strategy_agent import StrategyAgent, TradingDecision
from ..trading.execution_manager import ExecutionManager, OrderRequest, OrderType, OrderSide

logger = logging.getLogger(__name__)

class AllocationMethod(Enum):
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    KELLY_CRITERION = "kelly_criterion"

class RebalanceFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD = "threshold"

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    entry_time: datetime
    last_updated: datetime

@dataclass
class Portfolio:
    """Complete portfolio representation"""
    portfolio_id: str
    name: str
    cash: float
    positions: Dict[str, Position]
    total_value: float
    inception_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_1d: float
    beta: float
    
    # Allocation
    target_allocation: Dict[str, float]
    current_allocation: Dict[str, float]
    
    last_rebalance: datetime
    last_updated: datetime

@dataclass
class RebalanceOrder:
    """Rebalancing trade order"""
    symbol: str
    target_weight: float
    current_weight: float
    target_value: float
    current_value: float
    trade_value: float
    action: str  # "buy", "sell", "hold"

@dataclass
class PerformanceAttribution:
    """Performance attribution analysis"""
    asset_allocation: Dict[str, float]  # Return from asset allocation
    security_selection: Dict[str, float]  # Return from security selection
    interaction: Dict[str, float]  # Interaction effects
    total_excess_return: float
    benchmark_return: float
    portfolio_return: float

class PortfolioOptimizer:
    """Portfolio optimization engine"""
    
    def __init__(self):
        self.optimization_methods = {
            AllocationMethod.EQUAL_WEIGHT: self._equal_weight,
            AllocationMethod.RISK_PARITY: self._risk_parity,
            AllocationMethod.MEAN_VARIANCE: self._mean_variance,
            AllocationMethod.BLACK_LITTERMAN: self._black_litterman,
            AllocationMethod.KELLY_CRITERION: self._kelly_criterion
        }
    
    async def optimize_portfolio(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        method: AllocationMethod,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Optimize portfolio allocation"""
        
        constraints = constraints or {}
        optimization_func = self.optimization_methods.get(method)
        
        if not optimization_func:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        weights = await optimization_func(
            symbols, expected_returns, covariance_matrix, constraints
        )
        
        return dict(zip(symbols, weights))
    
    async def _equal_weight(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Equal weight allocation"""
        n_assets = len(symbols)
        return np.ones(n_assets) / n_assets
    
    async def _risk_parity(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Risk parity allocation"""
        
        n_assets = len(symbols)
        
        def risk_budget_objective(weights):
            """Minimize difference between marginal risk contributions"""
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x if result.success else x0
    
    async def _mean_variance(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Mean-variance optimization (Markowitz)"""
        
        # Convert expected returns to array
        returns_array = np.array([expected_returns.get(s, 0.0) for s in symbols])
        
        # Risk aversion parameter
        risk_aversion = constraints.get('risk_aversion', 1.0)
        
        def objective(weights):
            """Maximize utility: return - 0.5 * risk_aversion * variance"""
            portfolio_return = weights.T @ returns_array
            portfolio_variance = weights.T @ covariance_matrix @ weights
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        n_assets = len(symbols)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Add minimum return constraint if specified
        min_return = constraints.get('min_return')
        if min_return:
            cons.append({
                'type': 'ineq',
                'fun': lambda w: w.T @ returns_array - min_return
            })
        
        # Bounds
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x if result.success else x0
    
    async def _black_litterman(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Black-Litterman optimization"""
        
        # Simplified Black-Litterman (full implementation would require views)
        # Fall back to mean-variance for now
        return await self._mean_variance(symbols, expected_returns, covariance_matrix, constraints)
    
    async def _kelly_criterion(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Kelly Criterion optimization"""
        
        # Kelly optimal weights: w = Σ^-1 * μ
        returns_array = np.array([expected_returns.get(s, 0.0) for s in symbols])
        
        try:
            # Compute Kelly weights
            inv_cov = np.linalg.inv(covariance_matrix)
            kelly_weights = inv_cov @ returns_array
            
            # Normalize to sum to 1
            kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            # Apply leverage constraint (Kelly can suggest >100% allocation)
            max_leverage = constraints.get('max_leverage', 1.0)
            if np.sum(np.abs(kelly_weights)) > max_leverage:
                kelly_weights = kelly_weights * max_leverage / np.sum(np.abs(kelly_weights))
            
            # Ensure non-negative weights if specified
            if constraints.get('long_only', True):
                kelly_weights = np.maximum(kelly_weights, 0)
                kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            return kelly_weights
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            logger.warning("Covariance matrix is singular, using equal weights")
            return np.ones(len(symbols)) / len(symbols)

class PortfolioManager:
    """Advanced Portfolio Management System"""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        risk_manager: RiskManager,
        execution_manager: ExecutionManager,
        strategy_agent: StrategyAgent
    ):
        self.cache_manager = cache_manager
        self.risk_manager = risk_manager
        self.execution_manager = execution_manager
        self.strategy_agent = strategy_agent
        self.optimizer = PortfolioOptimizer()
        
        # Portfolio tracking
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        
        # Rebalancing parameters
        self.rebalance_threshold = 0.05  # 5% drift threshold
        self.min_trade_size = 100  # Minimum trade size
        
    async def create_portfolio(
        self,
        portfolio_id: str,
        name: str,
        initial_cash: float,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    ) -> Portfolio:
        """Create new portfolio"""
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            name=name,
            cash=initial_cash,
            positions={},
            total_value=initial_cash,
            inception_date=datetime.utcnow(),
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            var_1d=0.0,
            beta=1.0,
            target_allocation={},
            current_allocation={},
            last_rebalance=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        self.portfolios[portfolio_id] = portfolio
        
        # Store in database
        await self._store_portfolio_in_db(portfolio)
        
        logger.info(f"Created portfolio {portfolio_id}: {name}")
        return portfolio
    
    async def add_position(
        self,
        portfolio_id: str,
        symbol: str,
        quantity: float,
        price: float
    ) -> bool:
        """Add position to portfolio"""
        
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        # Calculate trade value
        trade_value = quantity * price
        
        # Check if sufficient cash
        if trade_value > portfolio.cash:
            raise ValueError(f"Insufficient cash: {portfolio.cash:.2f} < {trade_value:.2f}")
        
        # Update or create position
        if symbol in portfolio.positions:
            # Update existing position
            pos = portfolio.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost = pos.quantity * pos.average_cost + trade_value
            
            pos.quantity = total_quantity
            pos.average_cost = total_cost / total_quantity if total_quantity > 0 else 0
            pos.last_updated = datetime.utcnow()
        else:
            # Create new position
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_cost=price,
                current_price=price,
                market_value=trade_value,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                weight=0.0,
                entry_time=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
        
        # Update portfolio cash
        portfolio.cash -= trade_value
        
        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio)
        
        logger.info(f"Added position: {symbol} x{quantity} @ {price} to portfolio {portfolio_id}")
        return True
    
    async def remove_position(
        self,
        portfolio_id: str,
        symbol: str,
        quantity: float,
        price: float
    ) -> bool:
        """Remove position from portfolio"""
        
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        
        if symbol not in portfolio.positions:
            raise ValueError(f"Position {symbol} not found in portfolio")
        
        pos = portfolio.positions[symbol]
        
        if quantity > pos.quantity:
            raise ValueError(f"Insufficient quantity: {pos.quantity} < {quantity}")
        
        # Calculate realized PnL
        realized_pnl = (price - pos.average_cost) * quantity
        
        # Update position
        pos.quantity -= quantity
        pos.realized_pnl += realized_pnl
        pos.last_updated = datetime.utcnow()
        
        # Add cash back
        portfolio.cash += quantity * price
        
        # Remove position if quantity is zero
        if pos.quantity <= 0:
            del portfolio.positions[symbol]
        
        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio)
        
        logger.info(f"Removed position: {symbol} x{quantity} @ {price} from portfolio {portfolio_id}")
        return True
    
    async def update_portfolio_prices(self, portfolio_id: str) -> bool:
        """Update current prices for all positions"""
        
        if portfolio_id not in self.portfolios:
            return False
        
        portfolio = self.portfolios[portfolio_id]
        
        # Update prices for all positions
        for symbol, position in portfolio.positions.items():
            # Get current price from cache or market data
            current_price = await self._get_current_price(symbol)
            
            # Update position metrics
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_cost) * position.quantity
            position.last_updated = datetime.utcnow()
        
        # Update portfolio metrics
        await self._update_portfolio_metrics(portfolio)
        
        return True
    
    async def _update_portfolio_metrics(self, portfolio: Portfolio):
        """Update portfolio-level metrics"""
        
        # Calculate total value
        positions_value = sum(pos.market_value for pos in portfolio.positions.values())
        portfolio.total_value = portfolio.cash + positions_value
        
        # Calculate current allocation
        portfolio.current_allocation = {}
        if portfolio.total_value > 0:
            for symbol, position in portfolio.positions.items():
                portfolio.current_allocation[symbol] = position.market_value / portfolio.total_value
                position.weight = position.market_value / portfolio.total_value
        
        # Calculate returns
        initial_value = portfolio.cash + sum(
            pos.quantity * pos.average_cost for pos in portfolio.positions.values()
        )
        if initial_value > 0:
            portfolio.total_return = (portfolio.total_value - initial_value) / initial_value
        
        # Calculate performance metrics (simplified)
        await self._calculate_performance_metrics(portfolio)
        
        portfolio.last_updated = datetime.utcnow()
        
        # Cache updated portfolio
        await self.cache_manager.set(
            f"portfolio:{portfolio.portfolio_id}",
            portfolio.__dict__,
            expire=3600
        )
    
    async def _calculate_performance_metrics(self, portfolio: Portfolio):
        """Calculate portfolio performance metrics"""
        
        # Get historical performance data
        history = self.performance_history.get(portfolio.portfolio_id, [])
        
        if len(history) < 2:
            # Insufficient data for calculations
            portfolio.annualized_return = 0.0
            portfolio.volatility = 0.0
            portfolio.sharpe_ratio = 0.0
            portfolio.max_drawdown = 0.0
            return
        
        # Calculate daily returns
        values = [h['total_value'] for h in history]
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) == 0:
            return
        
        # Annualized return
        portfolio.annualized_return = returns.mean() * 252
        
        # Volatility
        portfolio.volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        portfolio.sharpe_ratio = portfolio.annualized_return / portfolio.volatility if portfolio.volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        portfolio.max_drawdown = abs(drawdowns.min())
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        
        # Try cache first
        cache_key = f"current_price:{symbol}"
        cached_price = await self.cache_manager.get(cache_key)
        
        if cached_price:
            return cached_price
        
        # Get from market data (simplified)
        market_data = await self.cache_manager.get(f"market_data:{symbol}")
        if market_data:
            price = market_data.get('last_price', 100.0)
        else:
            price = 100.0  # Default price
        
        # Cache for short time
        await self.cache_manager.set(cache_key, price, expire=60)
        
        return price
    
    async def optimize_portfolio_allocation(
        self,
        portfolio_id: str,
        symbols: List[str],
        method: AllocationMethod = AllocationMethod.MEAN_VARIANCE,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Optimize portfolio allocation"""
        
        # Get expected returns from strategy agent
        expected_returns = {}
        for symbol in symbols:
            decision = await self.strategy_agent.make_trading_decision(symbol)
            expected_returns[symbol] = decision.expected_return
        
        # Calculate covariance matrix
        covariance_matrix = await self._calculate_covariance_matrix(symbols)
        
        # Optimize allocation
        optimal_allocation = await self.optimizer.optimize_portfolio(
            symbols, expected_returns, covariance_matrix, method, constraints
        )
        
        return optimal_allocation
    
    async def _calculate_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """Calculate covariance matrix for symbols"""
        
        # Get return data for all symbols
        returns_data = {}
        for symbol in symbols:
            cache_key = f"price_data:{symbol}"
            price_data = await self.cache_manager.get(cache_key)
            
            if price_data and 'returns' in price_data:
                returns = pd.Series(price_data['returns']).pct_change().dropna()
                if len(returns) >= 30:  # Minimum data requirement
                    returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            # Return identity matrix if insufficient data
            n = len(symbols)
            return np.eye(n) * 0.01  # 1% variance assumption
        
        # Align series to same length
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {
            symbol: returns.tail(min_length) 
            for symbol, returns in returns_data.items()
        }
        
        # Create DataFrame and calculate covariance
        returns_df = pd.DataFrame(aligned_returns)
        
        # Annualize covariance matrix
        covariance_matrix = returns_df.cov().values * 252
        
        # Fill missing symbols with default variance
        if len(covariance_matrix) < len(symbols):
            full_cov = np.eye(len(symbols)) * 0.01
            for i, symbol in enumerate(symbols):
                if symbol in returns_data:
                    idx = list(returns_data.keys()).index(symbol)
                    full_cov[i, i] = covariance_matrix[idx, idx]
            covariance_matrix = full_cov
        
        return covariance_matrix
    
    async def rebalance_portfolio(
        self,
        portfolio_id: str,
        target_allocation: Dict[str, float],
        rebalance_threshold: float = None
    ) -> List[RebalanceOrder]:
        """Rebalance portfolio to target allocation"""
        
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        threshold = rebalance_threshold or self.rebalance_threshold
        
        # Update current prices
        await self.update_portfolio_prices(portfolio_id)
        
        # Calculate rebalancing orders
        rebalance_orders = []
        
        for symbol, target_weight in target_allocation.items():
            current_weight = portfolio.current_allocation.get(symbol, 0.0)
            weight_diff = abs(target_weight - current_weight)
            
            # Check if rebalancing is needed
            if weight_diff >= threshold:
                target_value = target_weight * portfolio.total_value
                current_value = portfolio.positions.get(symbol, Position(
                    symbol=symbol, quantity=0, average_cost=0, current_price=0,
                    market_value=0, unrealized_pnl=0, realized_pnl=0, weight=0,
                    entry_time=datetime.utcnow(), last_updated=datetime.utcnow()
                )).market_value
                
                trade_value = target_value - current_value
                
                # Skip small trades
                if abs(trade_value) < self.min_trade_size:
                    continue
                
                action = "buy" if trade_value > 0 else "sell" if trade_value < 0 else "hold"
                
                rebalance_order = RebalanceOrder(
                    symbol=symbol,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    target_value=target_value,
                    current_value=current_value,
                    trade_value=trade_value,
                    action=action
                )
                
                rebalance_orders.append(rebalance_order)
        
        # Execute rebalancing orders
        if rebalance_orders:
            await self._execute_rebalance_orders(portfolio_id, rebalance_orders)
            portfolio.last_rebalance = datetime.utcnow()
            portfolio.target_allocation = target_allocation.copy()
        
        return rebalance_orders
    
    async def _execute_rebalance_orders(
        self,
        portfolio_id: str,
        rebalance_orders: List[RebalanceOrder]
    ):
        """Execute rebalancing trades"""
        
        for order in rebalance_orders:
            if order.action == "hold":
                continue
            
            # Get current price
            current_price = await self._get_current_price(order.symbol)
            
            # Calculate quantity to trade
            quantity = abs(order.trade_value) / current_price
            
            # Create order request
            order_request = OrderRequest(
                symbol=order.symbol,
                side=OrderSide.BUY if order.action == "buy" else OrderSide.SELL,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            try:
                # Submit order
                order_id = await self.execution_manager.submit_order(order_request)
                
                logger.info(f"Rebalance order submitted: {order_id} - {order.symbol} {order.action} {quantity}")
                
                # Update portfolio position (simplified - in practice wait for fills)
                if order.action == "buy":
                    await self.add_position(portfolio_id, order.symbol, quantity, current_price)
                else:
                    await self.remove_position(portfolio_id, order.symbol, quantity, current_price)
                
            except Exception as e:
                logger.error(f"Error executing rebalance order for {order.symbol}: {e}")
    
    async def get_portfolio_performance(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio performance metrics"""
        
        if portfolio_id not in self.portfolios:
            return {}
        
        portfolio = self.portfolios[portfolio_id]
        
        # Calculate risk metrics
        portfolio_dict = {pos.symbol: pos.market_value for pos in portfolio.positions.values()}
        market_data = {}
        
        # Get market data for risk calculation
        for symbol in portfolio_dict.keys():
            cache_key = f"price_data:{symbol}"
            price_data = await self.cache_manager.get(cache_key)
            if price_data:
                df = pd.DataFrame(price_data)
                market_data[symbol] = df
        
        # Calculate portfolio risk
        risk_assessment = None
        if market_data:
            risk_assessment = await self.risk_manager.assess_portfolio_risk(portfolio_dict, market_data)
        
        # Performance attribution (simplified)
        attribution = await self._calculate_performance_attribution(portfolio)
        
        performance = {
            'portfolio_id': portfolio.portfolio_id,
            'name': portfolio.name,
            'total_value': portfolio.total_value,
            'cash': portfolio.cash,
            'total_return': portfolio.total_return,
            'annualized_return': portfolio.annualized_return,
            'volatility': portfolio.volatility,
            'sharpe_ratio': portfolio.sharpe_ratio,
            'max_drawdown': portfolio.max_drawdown,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                    'weight': pos.weight,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in portfolio.positions.items()
            },
            'allocation': portfolio.current_allocation,
            'target_allocation': portfolio.target_allocation,
            'risk_metrics': risk_assessment.__dict__ if risk_assessment else {},
            'attribution': attribution.__dict__ if attribution else {},
            'last_updated': portfolio.last_updated.isoformat()
        }
        
        return performance
    
    async def _calculate_performance_attribution(self, portfolio: Portfolio) -> PerformanceAttribution:
        """Calculate performance attribution (simplified)"""
        
        # Simplified attribution - in practice would need benchmark data
        return PerformanceAttribution(
            asset_allocation={},
            security_selection={},
            interaction={},
            total_excess_return=0.0,
            benchmark_return=0.0,
            portfolio_return=portfolio.total_return
        )
    
    async def _store_portfolio_in_db(self, portfolio: Portfolio):
        """Store portfolio in database"""
        
        try:
            async with get_db_connection() as conn:
                await conn.execute("""
                    INSERT INTO portfolios (
                        portfolio_id, name, cash, total_value, inception_date,
                        total_return, annualized_return, volatility, sharpe_ratio,
                        max_drawdown, last_updated
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (portfolio_id) DO UPDATE SET
                        cash = EXCLUDED.cash,
                        total_value = EXCLUDED.total_value,
                        total_return = EXCLUDED.total_return,
                        annualized_return = EXCLUDED.annualized_return,
                        volatility = EXCLUDED.volatility,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        last_updated = EXCLUDED.last_updated
                """, (
                    portfolio.portfolio_id,
                    portfolio.name,
                    portfolio.cash,
                    portfolio.total_value,
                    portfolio.inception_date,
                    portfolio.total_return,
                    portfolio.annualized_return,
                    portfolio.volatility,
                    portfolio.sharpe_ratio,
                    portfolio.max_drawdown,
                    portfolio.last_updated
                ))
                
                # Store positions
                for symbol, position in portfolio.positions.items():
                    await conn.execute("""
                        INSERT INTO positions (
                            portfolio_id, symbol, quantity, average_cost,
                            current_price, market_value, unrealized_pnl,
                            realized_pnl, weight, entry_time, last_updated
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (portfolio_id, symbol) DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            average_cost = EXCLUDED.average_cost,
                            current_price = EXCLUDED.current_price,
                            market_value = EXCLUDED.market_value,
                            unrealized_pnl = EXCLUDED.unrealized_pnl,
                            realized_pnl = EXCLUDED.realized_pnl,
                            weight = EXCLUDED.weight,
                            last_updated = EXCLUDED.last_updated
                    """, (
                        portfolio.portfolio_id,
                        symbol,
                        position.quantity,
                        position.average_cost,
                        position.current_price,
                        position.market_value,
                        position.unrealized_pnl,
                        position.realized_pnl,
                        position.weight,
                        position.entry_time,
                        position.last_updated
                    ))
        
        except Exception as e:
            logger.error(f"Error storing portfolio in database: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of all portfolios"""
        
        summary = {
            'total_portfolios': len(self.portfolios),
            'total_aum': sum(p.total_value for p in self.portfolios.values()),
            'portfolios': []
        }
        
        for portfolio in self.portfolios.values():
            portfolio_summary = {
                'portfolio_id': portfolio.portfolio_id,
                'name': portfolio.name,
                'total_value': portfolio.total_value,
                'total_return': portfolio.total_return,
                'sharpe_ratio': portfolio.sharpe_ratio,
                'positions_count': len(portfolio.positions),
                'last_updated': portfolio.last_updated.isoformat()
            }
            summary['portfolios'].append(portfolio_summary)
        
        return summary 