"""
Advanced Options Trading Engine
Sophisticated Options Strategies and Risk Management

This module handles:
- Black-Scholes option pricing with Greeks
- Implied volatility calculation and modeling
- Advanced options strategies (spreads, straddles, iron condors)
- Options portfolio risk management
- Real-time volatility surface construction
- Options flow analysis and sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

from ..core.cache import TradingCache
# Database connection will be handled by main application

logger = logging.getLogger(__name__)

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class OptionStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

class StrategyType(Enum):
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COLLAR = "collar"

@dataclass
class OptionContract:
    """Individual option contract specification"""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: datetime
    style: OptionStyle = OptionStyle.AMERICAN
    
    # Market data
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    # Calculated values
    theoretical_price: Optional[float] = None
    implied_volatility: Optional[float] = None

@dataclass
class Greeks:
    """Option Greeks for risk management"""
    delta: float        # Price sensitivity to underlying
    gamma: float        # Delta sensitivity to underlying
    theta: float        # Time decay
    vega: float         # Volatility sensitivity
    rho: float          # Interest rate sensitivity
    
    # Second-order Greeks
    charm: float = 0.0  # Delta decay
    vanna: float = 0.0  # Vega/delta cross-sensitivity
    volga: float = 0.0  # Vega convexity

@dataclass
class OptionPosition:
    """Position in option contracts"""
    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    
    # Current market values
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    # Greeks
    greeks: Optional[Greeks] = None

@dataclass
class OptionsStrategy:
    """Multi-leg options strategy"""
    strategy_id: str
    strategy_type: StrategyType
    underlying: str
    positions: List[OptionPosition]
    
    # Strategy-level metrics
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakeven_points: List[float] = field(default_factory=list)
    probability_of_profit: Optional[float] = None
    
    # Portfolio Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

class BlackScholesEngine:
    """Advanced Black-Scholes pricing engine with Greeks"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def calculate_option_price(
        self,
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiration (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: OptionType,
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate Black-Scholes option price"""
        
        if T <= 0:
            # Option has expired
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Adjust for dividend yield
        S_adj = S * np.exp(-dividend_yield * T)
        
        d1 = (np.log(S_adj / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == OptionType.CALL:
            price = (S_adj * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S_adj * norm.cdf(-d1))
        
        return max(price, 0.0)
    
    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        dividend_yield: float = 0.0
    ) -> Greeks:
        """Calculate all option Greeks"""
        
        if T <= 0:
            # Option has expired - all Greeks are zero except delta
            delta = 1.0 if (option_type == OptionType.CALL and S > K) else 0.0
            if option_type == OptionType.PUT and S < K:
                delta = -1.0
            return Greeks(delta=delta, gamma=0, theta=0, vega=0, rho=0)
        
        # Adjust for dividend yield
        S_adj = S * np.exp(-dividend_yield * T)
        
        d1 = (np.log(S_adj / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal PDF and CDF
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = np.exp(-dividend_yield * T) * cdf_d1
        else:
            delta = -np.exp(-dividend_yield * T) * norm.cdf(-d1)
        
        # Gamma (same for calls and puts)
        gamma = (np.exp(-dividend_yield * T) * pdf_d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * pdf_d1 * sigma * np.exp(-dividend_yield * T)) / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            theta = (theta_common - r * K * np.exp(-r * T) * cdf_d2 + 
                    dividend_yield * S * np.exp(-dividend_yield * T) * cdf_d1) / 365
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2) - 
                    dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)) / 365
        
        # Vega (same for calls and puts)
        vega = S * np.exp(-dividend_yield * T) * pdf_d1 * np.sqrt(T) / 100
        
        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * cdf_d2 / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Second-order Greeks
        charm = -np.exp(-dividend_yield * T) * pdf_d1 * (
            (2 * (r - dividend_yield) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        ) / 365
        
        vanna = vega * (-d2 / sigma) / 100
        
        volga = vega * d1 * d2 / sigma / 100
        
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            charm=charm,
            vanna=vanna,
            volga=volga
        )
    
    def calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """Calculate implied volatility using Brent's method"""
        
        if T <= 0:
            return None
        
        def objective(sigma):
            theoretical_price = self.calculate_option_price(
                S, K, T, r, sigma, option_type, dividend_yield
            )
            return theoretical_price - market_price
        
        try:
            # Search for implied volatility between 0.01% and 500%
            implied_vol = brentq(objective, 0.0001, 5.0, xtol=1e-6, maxiter=100)
            return implied_vol
        except (ValueError, RuntimeError):
            # Could not find implied volatility
            return None

class VolatilitySurface:
    """3D volatility surface modeling"""
    
    def __init__(self):
        self.surface_data: Dict[Tuple[float, float], float] = {}  # (strike, expiry) -> iv
        self.underlying_price: Optional[float] = None
        self.last_updated: Optional[datetime] = None
    
    def add_implied_volatility(
        self,
        strike: float,
        time_to_expiry: float,
        implied_vol: float
    ):
        """Add implied volatility point to surface"""
        self.surface_data[(strike, time_to_expiry)] = implied_vol
        self.last_updated = datetime.utcnow()
    
    def interpolate_volatility(
        self,
        strike: float,
        time_to_expiry: float
    ) -> Optional[float]:
        """Interpolate implied volatility for given strike and time"""
        
        if not self.surface_data:
            return None
        
        # Simple interpolation - in practice would use more sophisticated methods
        strikes = [k for k, t in self.surface_data.keys()]
        times = [t for k, t in self.surface_data.keys()]
        
        # Find closest points for interpolation
        closest_points = sorted(
            self.surface_data.items(),
            key=lambda x: abs(x[0][0] - strike) + abs(x[0][1] - time_to_expiry)
        )
        
        if len(closest_points) >= 4:
            # Bilinear interpolation
            return self._bilinear_interpolation(strike, time_to_expiry, closest_points[:4])
        else:
            # Return closest point
            return closest_points[0][1]
    
    def _bilinear_interpolation(
        self,
        target_strike: float,
        target_time: float,
        points: List[Tuple[Tuple[float, float], float]]
    ) -> float:
        """Perform bilinear interpolation"""
        
        # Simplified bilinear interpolation
        weights = []
        values = []
        
        for (strike, time), vol in points:
            weight = 1.0 / (1.0 + abs(strike - target_strike) + abs(time - target_time))
            weights.append(weight)
            values.append(vol)
        
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(w * v for w, v in zip(weights, values)) / total_weight
        else:
            return points[0][1]  # Fallback to first point

class OptionsStrategyBuilder:
    """Builder for complex options strategies"""
    
    def __init__(self, pricing_engine: BlackScholesEngine):
        self.pricing_engine = pricing_engine
    
    def create_long_call(
        self,
        underlying: str,
        strike: float,
        expiration: datetime,
        premium: float,
        quantity: int = 1
    ) -> OptionsStrategy:
        """Create long call strategy"""
        
        contract = OptionContract(
            symbol=f"{underlying}_{strike}C_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=strike,
            expiration=expiration
        )
        
        position = OptionPosition(
            contract=contract,
            quantity=quantity,
            entry_price=premium,
            entry_time=datetime.utcnow()
        )
        
        strategy = OptionsStrategy(
            strategy_id=f"LC_{underlying}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_type=StrategyType.LONG_CALL,
            underlying=underlying,
            positions=[position],
            max_profit=float('inf'),  # Unlimited upside
            max_loss=premium * quantity,
            breakeven_points=[strike + premium]
        )
        
        return strategy
    
    def create_bull_call_spread(
        self,
        underlying: str,
        long_strike: float,
        short_strike: float,
        expiration: datetime,
        long_premium: float,
        short_premium: float,
        quantity: int = 1
    ) -> OptionsStrategy:
        """Create bull call spread strategy"""
        
        # Long call (lower strike)
        long_contract = OptionContract(
            symbol=f"{underlying}_{long_strike}C_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=long_strike,
            expiration=expiration
        )
        
        # Short call (higher strike)
        short_contract = OptionContract(
            symbol=f"{underlying}_{short_strike}C_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=short_strike,
            expiration=expiration
        )
        
        long_position = OptionPosition(
            contract=long_contract,
            quantity=quantity,
            entry_price=long_premium,
            entry_time=datetime.utcnow()
        )
        
        short_position = OptionPosition(
            contract=short_contract,
            quantity=-quantity,  # Short position
            entry_price=short_premium,
            entry_time=datetime.utcnow()
        )
        
        net_premium = long_premium - short_premium
        max_profit = (short_strike - long_strike) - net_premium
        max_loss = net_premium
        
        strategy = OptionsStrategy(
            strategy_id=f"BCS_{underlying}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            underlying=underlying,
            positions=[long_position, short_position],
            max_profit=max_profit * quantity,
            max_loss=max_loss * quantity,
            breakeven_points=[long_strike + net_premium]
        )
        
        return strategy
    
    def create_iron_condor(
        self,
        underlying: str,
        put_strikes: Tuple[float, float],  # (short_put_strike, long_put_strike)
        call_strikes: Tuple[float, float],  # (short_call_strike, long_call_strike)
        expiration: datetime,
        premiums: Tuple[float, float, float, float],  # (short_put, long_put, short_call, long_call)
        quantity: int = 1
    ) -> OptionsStrategy:
        """Create iron condor strategy"""
        
        short_put_strike, long_put_strike = put_strikes
        short_call_strike, long_call_strike = call_strikes
        short_put_premium, long_put_premium, short_call_premium, long_call_premium = premiums
        
        # Create all four positions
        positions = []
        
        # Short put
        short_put_contract = OptionContract(
            symbol=f"{underlying}_{short_put_strike}P_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.PUT,
            strike=short_put_strike,
            expiration=expiration
        )
        positions.append(OptionPosition(
            contract=short_put_contract,
            quantity=-quantity,
            entry_price=short_put_premium,
            entry_time=datetime.utcnow()
        ))
        
        # Long put
        long_put_contract = OptionContract(
            symbol=f"{underlying}_{long_put_strike}P_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.PUT,
            strike=long_put_strike,
            expiration=expiration
        )
        positions.append(OptionPosition(
            contract=long_put_contract,
            quantity=quantity,
            entry_price=long_put_premium,
            entry_time=datetime.utcnow()
        ))
        
        # Short call
        short_call_contract = OptionContract(
            symbol=f"{underlying}_{short_call_strike}C_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=short_call_strike,
            expiration=expiration
        )
        positions.append(OptionPosition(
            contract=short_call_contract,
            quantity=-quantity,
            entry_price=short_call_premium,
            entry_time=datetime.utcnow()
        ))
        
        # Long call
        long_call_contract = OptionContract(
            symbol=f"{underlying}_{long_call_strike}C_{expiration.strftime('%y%m%d')}",
            underlying=underlying,
            option_type=OptionType.CALL,
            strike=long_call_strike,
            expiration=expiration
        )
        positions.append(OptionPosition(
            contract=long_call_contract,
            quantity=quantity,
            entry_price=long_call_premium,
            entry_time=datetime.utcnow()
        ))
        
        # Calculate strategy metrics
        net_credit = (short_put_premium + short_call_premium - 
                     long_put_premium - long_call_premium)
        
        put_spread_width = short_put_strike - long_put_strike
        call_spread_width = long_call_strike - short_call_strike
        max_loss = max(put_spread_width, call_spread_width) - net_credit
        
        strategy = OptionsStrategy(
            strategy_id=f"IC_{underlying}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            strategy_type=StrategyType.IRON_CONDOR,
            underlying=underlying,
            positions=positions,
            max_profit=net_credit * quantity,
            max_loss=max_loss * quantity,
            breakeven_points=[
                short_put_strike - net_credit,
                short_call_strike + net_credit
            ]
        )
        
        return strategy

class OptionsRiskManager:
    """Advanced options portfolio risk management"""
    
    def __init__(self, pricing_engine: BlackScholesEngine):
        self.pricing_engine = pricing_engine
    
    def calculate_portfolio_greeks(
        self,
        positions: List[OptionPosition],
        underlying_price: float,
        risk_free_rate: float = 0.05
    ) -> Greeks:
        """Calculate portfolio-level Greeks"""
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        
        for position in positions:
            contract = position.contract
            
            # Calculate time to expiration
            time_to_expiry = (contract.expiration - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
            
            if time_to_expiry > 0:
                # Use implied volatility if available, otherwise estimate
                volatility = contract.implied_volatility or 0.25  # Default 25% vol
                
                greeks = self.pricing_engine.calculate_greeks(
                    S=underlying_price,
                    K=contract.strike,
                    T=time_to_expiry,
                    r=risk_free_rate,
                    sigma=volatility,
                    option_type=contract.option_type
                )
                
                # Scale by position size
                total_delta += greeks.delta * position.quantity
                total_gamma += greeks.gamma * position.quantity
                total_theta += greeks.theta * position.quantity
                total_vega += greeks.vega * position.quantity
                total_rho += greeks.rho * position.quantity
        
        return Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho
        )
    
    def calculate_var_options(
        self,
        positions: List[OptionPosition],
        underlying_price: float,
        confidence_level: float = 0.95,
        time_horizon: int = 1  # days
    ) -> float:
        """Calculate VaR for options portfolio"""
        
        # Monte Carlo simulation for options VaR
        num_simulations = 10000
        portfolio_values = []
        
        # Estimate underlying volatility
        avg_volatility = 0.25  # Default assumption
        
        for sim in range(num_simulations):
            # Simulate underlying price movement
            random_return = np.random.normal(0, avg_volatility / np.sqrt(252)) * np.sqrt(time_horizon)
            simulated_price = underlying_price * np.exp(random_return)
            
            # Calculate portfolio value at simulated price
            portfolio_value = 0.0
            
            for position in positions:
                contract = position.contract
                time_to_expiry = (contract.expiration - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
                time_to_expiry = max(0, time_to_expiry - time_horizon / 365)
                
                if time_to_expiry > 0:
                    volatility = contract.implied_volatility or avg_volatility
                    option_value = self.pricing_engine.calculate_option_price(
                        S=simulated_price,
                        K=contract.strike,
                        T=time_to_expiry,
                        r=0.05,
                        sigma=volatility,
                        option_type=contract.option_type
                    )
                else:
                    # Option expired
                    if contract.option_type == OptionType.CALL:
                        option_value = max(simulated_price - contract.strike, 0)
                    else:
                        option_value = max(contract.strike - simulated_price, 0)
                
                portfolio_value += option_value * position.quantity
            
            portfolio_values.append(portfolio_value)
        
        # Calculate current portfolio value
        current_value = 0.0
        for position in positions:
            if position.current_price:
                current_value += position.current_price * position.quantity
        
        # Calculate P&L distribution
        pnl_distribution = [value - current_value for value in portfolio_values]
        
        # VaR at specified confidence level
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(pnl_distribution, var_percentile)
        
        return abs(var)

class OptionsFlowAnalyzer:
    """Options flow and sentiment analysis"""
    
    def __init__(self, cache_manager: TradingCache):
        self.cache_manager = cache_manager
    
    async def analyze_options_flow(
        self,
        underlying: str,
        timeframe: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Analyze recent options flow for sentiment"""
        
        # Get recent options transactions (simplified)
        flow_data = await self._get_options_flow_data(underlying, timeframe)
        
        if not flow_data:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Analyze flow patterns
        call_volume = sum(trade['volume'] for trade in flow_data if trade['option_type'] == 'call')
        put_volume = sum(trade['volume'] for trade in flow_data if trade['option_type'] == 'put')
        
        # Calculate put/call ratio
        put_call_ratio = put_volume / call_volume if call_volume > 0 else float('inf')
        
        # Analyze unusual activity
        large_trades = [trade for trade in flow_data if trade['volume'] > 1000]
        sweep_activity = len([trade for trade in large_trades if trade.get('is_sweep', False)])
        
        # Sentiment scoring
        if put_call_ratio < 0.7:
            sentiment = "bullish"
            confidence = min(1.0, (0.7 - put_call_ratio) * 2)
        elif put_call_ratio > 1.3:
            sentiment = "bearish"
            confidence = min(1.0, (put_call_ratio - 1.3) * 0.5)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        # Boost confidence for unusual activity
        if sweep_activity > 5:
            confidence = min(1.0, confidence * 1.2)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "put_call_ratio": put_call_ratio,
            "total_volume": call_volume + put_volume,
            "unusual_activity": sweep_activity,
            "large_trades_count": len(large_trades)
        }
    
    async def _get_options_flow_data(
        self,
        underlying: str,
        timeframe: timedelta
    ) -> List[Dict[str, Any]]:
        """Retrieve options flow data (mock implementation)"""
        
        # In practice, this would connect to options data providers
        # For now, return mock data
        
        cache_key = f"options_flow:{underlying}:{timeframe.total_seconds()}"
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Generate mock flow data
        np.random.seed(hash(underlying) % 1000)
        
        mock_flow = []
        for _ in range(np.random.randint(50, 200)):
            trade = {
                "timestamp": datetime.utcnow() - timedelta(seconds=np.random.randint(0, int(timeframe.total_seconds()))),
                "option_type": np.random.choice(["call", "put"], p=[0.6, 0.4]),
                "strike": 100 + np.random.randint(-20, 21),
                "volume": np.random.randint(1, 5000),
                "premium": np.random.uniform(0.1, 10.0),
                "is_sweep": np.random.random() < 0.1
            }
            mock_flow.append(trade)
        
        # Cache for short time
        await self.cache_manager.set(cache_key, mock_flow, expire=300)
        
        return mock_flow

class OptionsEngine:
    """Main options trading engine"""
    
    def __init__(self, cache_manager: TradingCache):
        self.cache_manager = cache_manager
        self.pricing_engine = BlackScholesEngine()
        self.strategy_builder = OptionsStrategyBuilder(self.pricing_engine)
        self.risk_manager = OptionsRiskManager(self.pricing_engine)
        self.flow_analyzer = OptionsFlowAnalyzer(cache_manager)
        self.volatility_surfaces: Dict[str, VolatilitySurface] = {}
        
        # Active positions and strategies
        self.positions: Dict[str, OptionPosition] = {}
        self.strategies: Dict[str, OptionsStrategy] = {}
    
    async def price_option(
        self,
        contract: OptionContract,
        underlying_price: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, Greeks]:
        """Price option and calculate Greeks"""
        
        time_to_expiry = (contract.expiration - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
        
        if time_to_expiry <= 0:
            # Option expired
            if contract.option_type == OptionType.CALL:
                price = max(underlying_price - contract.strike, 0)
            else:
                price = max(contract.strike - underlying_price, 0)
            
            greeks = Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
            return price, greeks
        
        # Use provided volatility or estimate from surface
        if volatility is None:
            if contract.underlying in self.volatility_surfaces:
                volatility = self.volatility_surfaces[contract.underlying].interpolate_volatility(
                    contract.strike, time_to_expiry
                )
            else:
                volatility = 0.25  # Default assumption
        
        price = self.pricing_engine.calculate_option_price(
            S=underlying_price,
            K=contract.strike,
            T=time_to_expiry,
            r=0.05,  # Risk-free rate
            sigma=volatility,
            option_type=contract.option_type
        )
        
        greeks = self.pricing_engine.calculate_greeks(
            S=underlying_price,
            K=contract.strike,
            T=time_to_expiry,
            r=0.05,
            sigma=volatility,
            option_type=contract.option_type
        )
        
        return price, greeks
    
    async def create_strategy(
        self,
        strategy_type: StrategyType,
        underlying: str,
        parameters: Dict[str, Any]
    ) -> OptionsStrategy:
        """Create options strategy"""
        
        if strategy_type == StrategyType.LONG_CALL:
            strategy = self.strategy_builder.create_long_call(
                underlying=underlying,
                strike=parameters['strike'],
                expiration=parameters['expiration'],
                premium=parameters['premium'],
                quantity=parameters.get('quantity', 1)
            )
        
        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            strategy = self.strategy_builder.create_bull_call_spread(
                underlying=underlying,
                long_strike=parameters['long_strike'],
                short_strike=parameters['short_strike'],
                expiration=parameters['expiration'],
                long_premium=parameters['long_premium'],
                short_premium=parameters['short_premium'],
                quantity=parameters.get('quantity', 1)
            )
        
        elif strategy_type == StrategyType.IRON_CONDOR:
            strategy = self.strategy_builder.create_iron_condor(
                underlying=underlying,
                put_strikes=parameters['put_strikes'],
                call_strikes=parameters['call_strikes'],
                expiration=parameters['expiration'],
                premiums=parameters['premiums'],
                quantity=parameters.get('quantity', 1)
            )
        
        else:
            raise ValueError(f"Strategy type {strategy_type} not implemented")
        
        # Store strategy
        self.strategies[strategy.strategy_id] = strategy
        
        # Add positions to tracking
        for position in strategy.positions:
            position_id = f"{position.contract.symbol}_{position.quantity}_{position.entry_time.isoformat()}"
            self.positions[position_id] = position
        
        return strategy
    
    async def update_volatility_surface(
        self,
        underlying: str,
        options_data: List[Dict[str, Any]]
    ):
        """Update volatility surface with market data"""
        
        if underlying not in self.volatility_surfaces:
            self.volatility_surfaces[underlying] = VolatilitySurface()
        
        surface = self.volatility_surfaces[underlying]
        
        # Get current underlying price
        underlying_price = await self._get_underlying_price(underlying)
        surface.underlying_price = underlying_price
        
        for option_data in options_data:
            contract = OptionContract(
                symbol=option_data['symbol'],
                underlying=underlying,
                option_type=OptionType(option_data['option_type']),
                strike=option_data['strike'],
                expiration=datetime.fromisoformat(option_data['expiration']),
                bid=option_data.get('bid'),
                ask=option_data.get('ask'),
                last=option_data.get('last')
            )
            
            # Calculate mid price
            if contract.bid and contract.ask:
                mid_price = (contract.bid + contract.ask) / 2
            elif contract.last:
                mid_price = contract.last
            else:
                continue
            
            # Calculate implied volatility
            time_to_expiry = (contract.expiration - datetime.utcnow()).total_seconds() / (365 * 24 * 3600)
            
            if time_to_expiry > 0:
                implied_vol = self.pricing_engine.calculate_implied_volatility(
                    market_price=mid_price,
                    S=underlying_price,
                    K=contract.strike,
                    T=time_to_expiry,
                    r=0.05,
                    option_type=contract.option_type
                )
                
                if implied_vol:
                    surface.add_implied_volatility(contract.strike, time_to_expiry, implied_vol)
                    contract.implied_volatility = implied_vol
    
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        """Get comprehensive options portfolio risk metrics"""
        
        if not self.positions:
            return {"message": "No positions"}
        
        # Calculate portfolio Greeks
        underlying_prices = {}
        positions_by_underlying = {}
        
        for position in self.positions.values():
            underlying = position.contract.underlying
            
            if underlying not in underlying_prices:
                underlying_prices[underlying] = await self._get_underlying_price(underlying)
            
            if underlying not in positions_by_underlying:
                positions_by_underlying[underlying] = []
            
            positions_by_underlying[underlying].append(position)
        
        risk_metrics = {}
        
        for underlying, positions in positions_by_underlying.items():
            portfolio_greeks = self.risk_manager.calculate_portfolio_greeks(
                positions, underlying_prices[underlying]
            )
            
            var_1d = self.risk_manager.calculate_var_options(
                positions, underlying_prices[underlying], confidence_level=0.95, time_horizon=1
            )
            
            risk_metrics[underlying] = {
                "portfolio_greeks": {
                    "delta": portfolio_greeks.delta,
                    "gamma": portfolio_greeks.gamma,
                    "theta": portfolio_greeks.theta,
                    "vega": portfolio_greeks.vega,
                    "rho": portfolio_greeks.rho
                },
                "var_1d": var_1d,
                "position_count": len(positions),
                "underlying_price": underlying_prices[underlying]
            }
        
        return risk_metrics
    
    async def _get_underlying_price(self, underlying: str) -> float:
        """Get current underlying asset price"""
        
        cache_key = f"price:{underlying}"
        cached_price = await self.cache_manager.get(cache_key)
        
        if cached_price:
            return cached_price
        
        # Mock price for demo
        mock_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "TSLA": 200.0,
            "SPY": 400.0
        }
        
        price = mock_prices.get(underlying, 100.0)
        await self.cache_manager.set(cache_key, price, expire=60)
        
        return price
    
    async def get_strategy_analysis(self, strategy_id: str) -> Dict[str, Any]:
        """Get comprehensive strategy analysis"""
        
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        # Get current underlying price
        underlying_price = await self._get_underlying_price(strategy.underlying)
        
        # Calculate current strategy value and Greeks
        total_value = 0.0
        strategy_greeks = Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
        
        for position in strategy.positions:
            current_price, greeks = await self.price_option(
                position.contract, underlying_price
            )
            
            position.current_price = current_price
            position.greeks = greeks
            
            # Update P&L
            if position.quantity > 0:  # Long position
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # Short position
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
            
            total_value += current_price * position.quantity
            
            # Aggregate Greeks
            strategy_greeks.delta += greeks.delta * position.quantity
            strategy_greeks.gamma += greeks.gamma * position.quantity
            strategy_greeks.theta += greeks.theta * position.quantity
            strategy_greeks.vega += greeks.vega * position.quantity
            strategy_greeks.rho += greeks.rho * position.quantity
        
        # Update strategy Greeks
        strategy.net_delta = strategy_greeks.delta
        strategy.net_gamma = strategy_greeks.gamma
        strategy.net_theta = strategy_greeks.theta
        strategy.net_vega = strategy_greeks.vega
        
        # Calculate profit probability (simplified)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in strategy.positions)
        
        analysis = {
            "strategy_id": strategy_id,
            "strategy_type": strategy.strategy_type.value,
            "underlying": strategy.underlying,
            "underlying_price": underlying_price,
            "current_value": total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "max_profit": strategy.max_profit,
            "max_loss": strategy.max_loss,
            "breakeven_points": strategy.breakeven_points,
            "greeks": {
                "delta": strategy_greeks.delta,
                "gamma": strategy_greeks.gamma,
                "theta": strategy_greeks.theta,
                "vega": strategy_greeks.vega,
                "rho": strategy_greeks.rho
            },
            "positions": [
                {
                    "contract_symbol": pos.contract.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "greeks": {
                        "delta": pos.greeks.delta if pos.greeks else 0,
                        "gamma": pos.greeks.gamma if pos.greeks else 0,
                        "theta": pos.greeks.theta if pos.greeks else 0,
                        "vega": pos.greeks.vega if pos.greeks else 0,
                        "rho": pos.greeks.rho if pos.greeks else 0
                    }
                }
                for pos in strategy.positions
            ]
        }
        
        return analysis 