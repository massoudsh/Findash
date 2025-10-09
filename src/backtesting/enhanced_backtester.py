"""
M10 | Enhanced Backtesting Agent
Advanced Strategy Validation and Performance Analysis

This agent handles:
- Historical strategy backtesting
- Monte Carlo simulation
- Walk-forward analysis
- Statistical significance testing
- Regime-based performance analysis
- Strategy robustness assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from ..core.cache import CacheManager
from ..database.postgres_connection import get_db
from ..strategies.strategy_agent import StrategyAgent, TradingDecision

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    WALK_FORWARD = "walk_forward"
    CROSS_VALIDATION = "cross_validation"

class MarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    rebalance_frequency: str = "monthly"
    transaction_costs: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps
    max_leverage: float = 1.0
    benchmark: str = "SPY"
    
    # Risk management
    max_drawdown_limit: float = 0.20
    position_size_limit: float = 0.20
    var_limit: float = 0.05
    
    # Monte Carlo parameters
    num_simulations: int = 1000
    
    # Walk-forward parameters
    training_period_months: int = 24
    testing_period_months: int = 6
    step_size_months: int = 3

@dataclass
class Trade:
    """Individual trade record"""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    return_pct: float
    holding_period: int  # days
    transaction_cost: float

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    config: BacktestConfig
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float
    beta: float
    alpha: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    returns_series: pd.Series
    
    # Trades
    trades: List[Trade]
    
    # Regime analysis
    regime_performance: Dict[MarketRegime, Dict[str, float]]
    
    # Statistical tests
    statistical_significance: Dict[str, float]
    
    # Execution time
    execution_time: float

@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results"""
    num_simulations: int
    confidence_intervals: Dict[str, Dict[str, float]]  # metric -> {percentile: value}
    probability_of_loss: float
    expected_return: float
    worst_case_scenario: Dict[str, float]
    best_case_scenario: Dict[str, float]
    distribution_stats: Dict[str, float]

@dataclass
class WalkForwardResults:
    """Walk-forward analysis results"""
    periods: List[Dict[str, Any]]
    average_performance: Dict[str, float]
    stability_metrics: Dict[str, float]
    degradation_analysis: Dict[str, float]

class EnhancedBacktester:
    """M10 - Advanced Backtesting and Strategy Validation Agent"""
    
    def __init__(self, cache_manager: CacheManager, strategy_agent: StrategyAgent):
        self.cache_manager = cache_manager
        self.strategy_agent = strategy_agent
        
        # Market data cache
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Benchmark data
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
        
        # Risk-free rate (simplified)
        self.risk_free_rate = 0.02  # 2% annual
        
    async def run_backtest(
        self,
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None,
        mode: BacktestMode = BacktestMode.HISTORICAL
    ) -> BacktestResults:
        """Run comprehensive backtest"""
        
        start_time = datetime.now()
        
        try:
            # Load market data
            await self._load_market_data(config)
            
            # Run backtest based on mode
            if mode == BacktestMode.HISTORICAL:
                results = await self._run_historical_backtest(config, strategy_func)
            elif mode == BacktestMode.MONTE_CARLO:
                results = await self._run_monte_carlo_backtest(config, strategy_func)
            elif mode == BacktestMode.WALK_FORWARD:
                results = await self._run_walk_forward_backtest(config, strategy_func)
            else:
                raise ValueError(f"Unsupported backtest mode: {mode}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            results.execution_time = execution_time
            
            # Store results
            await self._store_backtest_results(results)
            
            logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise
    
    async def _load_market_data(self, config: BacktestConfig):
        """Load historical market data"""
        
        symbols_to_load = config.symbols + [config.benchmark]
        
        for symbol in symbols_to_load:
            cache_key = f"historical_data:{symbol}:{config.start_date.date()}:{config.end_date.date()}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                self.market_data[symbol] = pd.DataFrame(cached_data)
            else:
                # Generate synthetic market data (in practice, load from data provider)
                data = self._generate_synthetic_data(
                    symbol, config.start_date, config.end_date
                )
                self.market_data[symbol] = data
                
                # Cache the data
                await self.cache_manager.set(
                    cache_key, 
                    data.to_dict('records'), 
                    expire=86400
                )
        
        # Store benchmark data separately
        if config.benchmark in self.market_data:
            self.benchmark_data[config.benchmark] = self.market_data[config.benchmark]
    
    def _generate_synthetic_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic market data for backtesting"""
        
        # Create date range
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        n_days = len(dates)
        
        # Market parameters (vary by symbol)
        if symbol in ['SPY', 'QQQ']:
            mu = 0.10 / 252  # 10% annual return
            sigma = 0.16 / np.sqrt(252)  # 16% annual volatility
        elif symbol in ['BTC-USD']:
            mu = 0.50 / 252  # 50% annual return
            sigma = 0.80 / np.sqrt(252)  # 80% annual volatility
        else:
            mu = 0.08 / 252  # 8% annual return
            sigma = 0.20 / np.sqrt(252)  # 20% annual volatility
        
        # Generate returns using geometric Brownian motion
        np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
        returns = np.random.normal(mu, sigma, n_days)
        
        # Calculate prices
        initial_price = 100.0
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Add some intraday noise
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        
        # Generate volume
        base_volume = 1000000
        volumes = base_volume * (1 + np.random.normal(0, 0.3, n_days))
        volumes = np.maximum(volumes, base_volume * 0.1)
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'returns': returns
        })
        
        return data
    
    async def _run_historical_backtest(
        self, 
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None
    ) -> BacktestResults:
        """Run historical backtest"""
        
        # Initialize portfolio
        portfolio = {
            'cash': config.initial_capital,
            'positions': {},  # symbol -> quantity
            'values': [],
            'dates': []
        }
        
        trades = []
        
        # Get trading dates
        trading_dates = self.market_data[config.symbols[0]]['date'].tolist()
        
        # Rebalance frequency mapping
        rebalance_freq_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63
        }
        
        rebalance_freq = rebalance_freq_map.get(config.rebalance_frequency, 21)
        
        # Run backtest day by day
        for i, date in enumerate(trading_dates):
            
            # Update portfolio value
            portfolio_value = await self._calculate_portfolio_value(
                portfolio, date, config.symbols
            )
            portfolio['values'].append(portfolio_value)
            portfolio['dates'].append(date)
            
            # Check for rebalancing
            if i % rebalance_freq == 0 or i == 0:
                
                # Get trading signals
                signals = await self._get_trading_signals(date, config.symbols, strategy_func)
                
                # Execute trades
                new_trades = await self._execute_rebalance(
                    portfolio, signals, date, config
                )
                trades.extend(new_trades)
        
        # Calculate final metrics
        results = await self._calculate_backtest_metrics(
            portfolio, trades, config
        )
        
        return results
    
    async def _get_trading_signals(
        self,
        date: datetime,
        symbols: List[str],
        strategy_func: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Get trading signals for given date"""
        
        signals = {}
        
        for symbol in symbols:
            if strategy_func:
                # Use custom strategy function
                signal = await strategy_func(symbol, date, self.market_data[symbol])
                signals[symbol] = signal
            else:
                # Use default strategy agent
                decision = await self.strategy_agent.make_trading_decision(symbol)
                
                # Convert decision to signal weight
                if decision.action == "buy":
                    signals[symbol] = decision.confidence
                elif decision.action == "sell":
                    signals[symbol] = -decision.confidence
                else:
                    signals[symbol] = 0.0
        
        # Normalize signals to sum to 1 (for long-only portfolio)
        total_long_signal = sum(max(0, s) for s in signals.values())
        if total_long_signal > 0:
            for symbol in signals:
                if signals[symbol] > 0:
                    signals[symbol] = signals[symbol] / total_long_signal
        
        return signals
    
    async def _execute_rebalance(
        self,
        portfolio: Dict,
        signals: Dict[str, float],
        date: datetime,
        config: BacktestConfig
    ) -> List[Trade]:
        """Execute portfolio rebalancing"""
        
        trades = []
        
        # Calculate current portfolio value
        portfolio_value = await self._calculate_portfolio_value(
            portfolio, date, list(signals.keys())
        )
        
        # Calculate target positions
        target_positions = {}
        for symbol, signal in signals.items():
            if signal > 0:  # Long position
                target_value = signal * portfolio_value
                current_price = self._get_price_at_date(symbol, date)
                target_quantity = target_value / current_price
                target_positions[symbol] = target_quantity
        
        # Execute trades
        for symbol in set(list(portfolio['positions'].keys()) + list(target_positions.keys())):
            
            current_quantity = portfolio['positions'].get(symbol, 0)
            target_quantity = target_positions.get(symbol, 0)
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > 1e-6:  # Minimum trade size
                
                current_price = self._get_price_at_date(symbol, date)
                
                # Account for transaction costs and slippage
                if trade_quantity > 0:  # Buy
                    execution_price = current_price * (1 + config.slippage)
                else:  # Sell
                    execution_price = current_price * (1 - config.slippage)
                
                trade_value = abs(trade_quantity) * execution_price
                transaction_cost = trade_value * config.transaction_costs
                
                # Update portfolio
                portfolio['positions'][symbol] = target_quantity
                portfolio['cash'] -= trade_quantity * execution_price + transaction_cost
                
                # Record trade
                trade = Trade(
                    entry_date=date,
                    exit_date=None,  # Will be set on exit
                    symbol=symbol,
                    side="long" if trade_quantity > 0 else "short",
                    entry_price=execution_price,
                    exit_price=None,
                    quantity=abs(trade_quantity),
                    pnl=0.0,  # Will be calculated on exit
                    return_pct=0.0,
                    holding_period=0,
                    transaction_cost=transaction_cost
                )
                
                trades.append(trade)
        
        return trades
    
    def _get_price_at_date(self, symbol: str, date: datetime) -> float:
        """Get price for symbol at specific date"""
        
        data = self.market_data[symbol]
        price_row = data[data['date'] == date]
        
        if len(price_row) > 0:
            return price_row.iloc[0]['close']
        else:
            # Use closest available date
            closest_row = data.iloc[(data['date'] - date).abs().argsort()[:1]]
            return closest_row.iloc[0]['close']
    
    async def _calculate_portfolio_value(
        self,
        portfolio: Dict,
        date: datetime,
        symbols: List[str]
    ) -> float:
        """Calculate total portfolio value at given date"""
        
        total_value = portfolio['cash']
        
        for symbol, quantity in portfolio['positions'].items():
            if symbol in symbols and quantity != 0:
                price = self._get_price_at_date(symbol, date)
                total_value += quantity * price
        
        return total_value
    
    async def _calculate_backtest_metrics(
        self,
        portfolio: Dict,
        trades: List[Trade],
        config: BacktestConfig
    ) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        
        # Create time series
        equity_curve = pd.Series(portfolio['values'], index=portfolio['dates'])
        returns_series = equity_curve.pct_change().dropna()
        
        # Calculate drawdown series
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        
        # Performance metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized metrics
        years = len(returns_series) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        volatility = returns_series.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns_series - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns_series.std() * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Calmar ratio
        max_drawdown = abs(drawdown_series.min())
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR calculations
        var_95 = np.percentile(returns_series, 5)
        var_99 = np.percentile(returns_series, 1)
        
        # Expected Shortfall
        tail_losses = returns_series[returns_series <= var_95]
        expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else var_95
        
        # Beta and Alpha (vs benchmark)
        benchmark_returns = None
        beta = 1.0
        alpha = 0.0
        
        if config.benchmark in self.benchmark_data:
            benchmark_equity = pd.Series(
                self.benchmark_data[config.benchmark]['close'].values,
                index=self.benchmark_data[config.benchmark]['date']
            )
            benchmark_returns = benchmark_equity.pct_change().dropna()
            
            # Align series
            min_length = min(len(returns_series), len(benchmark_returns))
            returns_aligned = returns_series.tail(min_length)
            benchmark_aligned = benchmark_returns.tail(min_length)
            
            # Calculate beta
            covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Calculate alpha
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_aligned.mean() * 252 - self.risk_free_rate))
        
        # Trade statistics
        completed_trades = [t for t in trades if t.exit_date is not None]
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        losing_trades = len([t for t in completed_trades if t.pnl < 0])
        
        win_rate = winning_trades / len(completed_trades) if completed_trades else 0
        
        wins = [t.pnl for t in completed_trades if t.pnl > 0]
        losses = [t.pnl for t in completed_trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        
        # Regime analysis
        regime_performance = await self._analyze_regime_performance(returns_series, config)
        
        # Statistical significance tests
        statistical_significance = await self._perform_statistical_tests(returns_series, benchmark_returns)
        
        results = BacktestResults(
            config=config,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            beta=beta,
            alpha=alpha,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            returns_series=returns_series,
            trades=trades,
            regime_performance=regime_performance,
            statistical_significance=statistical_significance,
            execution_time=0.0  # Will be set later
        )
        
        return results
    
    async def _analyze_regime_performance(
        self,
        returns_series: pd.Series,
        config: BacktestConfig
    ) -> Dict[MarketRegime, Dict[str, float]]:
        """Analyze performance across different market regimes"""
        
        regime_performance = {}
        
        # Simple regime classification based on rolling metrics
        window = 63  # 3-month window
        
        if len(returns_series) < window:
            return {}
        
        rolling_returns = returns_series.rolling(window).mean() * 252
        rolling_volatility = returns_series.rolling(window).std() * np.sqrt(252)
        
        # Define regime thresholds
        high_return_threshold = 0.10  # 10% annual return
        high_vol_threshold = 0.20  # 20% annual volatility
        
        regimes = {}
        
        for i in range(window, len(returns_series)):
            ret = rolling_returns.iloc[i]
            vol = rolling_volatility.iloc[i]
            
            if ret > high_return_threshold:
                regime = MarketRegime.BULL_MARKET
            elif ret < -high_return_threshold:
                regime = MarketRegime.BEAR_MARKET
            else:
                regime = MarketRegime.SIDEWAYS
            
            if vol > high_vol_threshold:
                if regime != MarketRegime.BEAR_MARKET:
                    regime = MarketRegime.HIGH_VOLATILITY
            elif vol < high_vol_threshold * 0.5:
                if regime != MarketRegime.BULL_MARKET:
                    regime = MarketRegime.LOW_VOLATILITY
            
            regimes[returns_series.index[i]] = regime
        
        # Calculate performance by regime
        for regime in MarketRegime:
            regime_dates = [date for date, r in regimes.items() if r == regime]
            
            if regime_dates:
                regime_returns = returns_series.loc[regime_dates]
                
                if len(regime_returns) > 0:
                    regime_performance[regime] = {
                        'return': regime_returns.mean() * 252,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'max_drawdown': 0.0,  # Simplified
                        'periods': len(regime_returns)
                    }
        
        return regime_performance
    
    async def _perform_statistical_tests(
        self,
        returns_series: pd.Series,
        benchmark_returns: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Perform statistical significance tests"""
        
        tests = {}
        
        # T-test for returns significantly different from zero
        if len(returns_series) > 30:
            t_stat, p_value = stats.ttest_1samp(returns_series, 0)
            tests['t_test_vs_zero'] = p_value
        
        # Shapiro-Wilk test for normality
        if len(returns_series) <= 5000:  # Shapiro-Wilk has limitations
            shapiro_stat, shapiro_p = stats.shapiro(returns_series)
            tests['normality_test'] = shapiro_p
        
        # Jarque-Bera test for normality
        if len(returns_series) > 7:
            jb_stat, jb_p = stats.jarque_bera(returns_series)
            tests['jarque_bera_test'] = jb_p
        
        # Ljung-Box test for autocorrelation
        if len(returns_series) > 10:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(returns_series, lags=10, return_df=True)
            tests['ljung_box_test'] = lb_test['lb_pvalue'].iloc[-1]
        
        # If benchmark available, test for alpha significance
        if benchmark_returns is not None and len(benchmark_returns) == len(returns_series):
            excess_returns = returns_series - benchmark_returns
            if len(excess_returns) > 30:
                t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
                tests['alpha_significance'] = p_value
        
        return tests
    
    async def _run_monte_carlo_backtest(
        self, 
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None
    ) -> BacktestResults:
        """Run Monte Carlo simulation backtest"""
        
        # Run multiple simulations with bootstrapped returns
        simulation_results = []
        
        # Get base returns for bootstrapping
        base_returns = {}
        for symbol in config.symbols:
            data = self.market_data[symbol]
            returns = data['returns'].dropna()
            base_returns[symbol] = returns.values
        
        for sim in range(config.num_simulations):
            # Bootstrap returns
            np.random.seed(sim)
            
            # Generate synthetic return series
            synthetic_data = {}
            for symbol in config.symbols:
                original_returns = base_returns[symbol]
                synthetic_returns = np.random.choice(
                    original_returns, 
                    size=len(original_returns), 
                    replace=True
                )
                
                # Create synthetic price series
                initial_price = 100.0
                prices = initial_price * np.exp(np.cumsum(synthetic_returns))
                
                synthetic_data[symbol] = pd.DataFrame({
                    'date': self.market_data[symbol]['date'],
                    'close': prices,
                    'returns': synthetic_returns
                })
            
            # Replace market data temporarily
            original_data = self.market_data.copy()
            self.market_data.update(synthetic_data)
            
            # Run backtest
            try:
                sim_config = BacktestConfig(
                    start_date=config.start_date,
                    end_date=config.end_date,
                    initial_capital=config.initial_capital,
                    symbols=config.symbols,
                    rebalance_frequency=config.rebalance_frequency,
                    transaction_costs=config.transaction_costs,
                    slippage=config.slippage,
                    benchmark=config.benchmark
                )
                
                result = await self._run_historical_backtest(sim_config, strategy_func)
                simulation_results.append(result)
                
            except Exception as e:
                logger.warning(f"Monte Carlo simulation {sim} failed: {e}")
                continue
            finally:
                # Restore original data
                self.market_data = original_data
        
        # Aggregate results
        if simulation_results:
            # Calculate confidence intervals
            returns = [r.total_return for r in simulation_results]
            sharpe_ratios = [r.sharpe_ratio for r in simulation_results]
            max_drawdowns = [r.max_drawdown for r in simulation_results]
            
            monte_carlo_results = MonteCarloResults(
                num_simulations=len(simulation_results),
                confidence_intervals={
                    'total_return': {
                        '5%': np.percentile(returns, 5),
                        '50%': np.percentile(returns, 50),
                        '95%': np.percentile(returns, 95)
                    },
                    'sharpe_ratio': {
                        '5%': np.percentile(sharpe_ratios, 5),
                        '50%': np.percentile(sharpe_ratios, 50),
                        '95%': np.percentile(sharpe_ratios, 95)
                    },
                    'max_drawdown': {
                        '5%': np.percentile(max_drawdowns, 5),
                        '50%': np.percentile(max_drawdowns, 50),
                        '95%': np.percentile(max_drawdowns, 95)
                    }
                },
                probability_of_loss=len([r for r in returns if r < 0]) / len(returns),
                expected_return=np.mean(returns),
                worst_case_scenario={'total_return': min(returns)},
                best_case_scenario={'total_return': max(returns)},
                distribution_stats={
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns)
                }
            )
            
            # Return the median result with Monte Carlo analysis
            median_idx = len(simulation_results) // 2
            best_result = simulation_results[median_idx]
            best_result.monte_carlo_results = monte_carlo_results
            
            return best_result
        else:
            raise ValueError("No successful Monte Carlo simulations")
    
    async def _run_walk_forward_backtest(
        self, 
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None
    ) -> BacktestResults:
        """Run walk-forward analysis"""
        
        training_period = timedelta(days=config.training_period_months * 30)
        testing_period = timedelta(days=config.testing_period_months * 30)
        step_size = timedelta(days=config.step_size_months * 30)
        
        walk_forward_results = []
        current_start = config.start_date
        
        while current_start + training_period + testing_period <= config.end_date:
            
            training_end = current_start + training_period
            testing_start = training_end
            testing_end = testing_start + testing_period
            
            # Create configurations for training and testing
            training_config = BacktestConfig(
                start_date=current_start,
                end_date=training_end,
                initial_capital=config.initial_capital,
                symbols=config.symbols,
                rebalance_frequency=config.rebalance_frequency,
                transaction_costs=config.transaction_costs,
                slippage=config.slippage,
                benchmark=config.benchmark
            )
            
            testing_config = BacktestConfig(
                start_date=testing_start,
                end_date=testing_end,
                initial_capital=config.initial_capital,
                symbols=config.symbols,
                rebalance_frequency=config.rebalance_frequency,
                transaction_costs=config.transaction_costs,
                slippage=config.slippage,
                benchmark=config.benchmark
            )
            
            try:
                # Run training backtest (for strategy optimization)
                # In practice, this would optimize strategy parameters
                
                # Run testing backtest
                test_result = await self._run_historical_backtest(testing_config, strategy_func)
                
                walk_forward_results.append({
                    'training_period': (current_start, training_end),
                    'testing_period': (testing_start, testing_end),
                    'result': test_result
                })
                
            except Exception as e:
                logger.warning(f"Walk-forward period failed: {e}")
            
            # Move to next period
            current_start += step_size
        
        # Aggregate walk-forward results
        if walk_forward_results:
            
            # Calculate stability metrics
            test_returns = [wf['result'].total_return for wf in walk_forward_results]
            test_sharpes = [wf['result'].sharpe_ratio for wf in walk_forward_results]
            
            walk_forward_analysis = WalkForwardResults(
                periods=walk_forward_results,
                average_performance={
                    'return': np.mean(test_returns),
                    'sharpe': np.mean(test_sharpes),
                    'win_rate': len([r for r in test_returns if r > 0]) / len(test_returns)
                },
                stability_metrics={
                    'return_std': np.std(test_returns),
                    'sharpe_std': np.std(test_sharpes),
                    'worst_period': min(test_returns),
                    'best_period': max(test_returns)
                },
                degradation_analysis={
                    'correlation_with_time': stats.pearsonr(range(len(test_returns)), test_returns)[0],
                    'trend_slope': np.polyfit(range(len(test_returns)), test_returns, 1)[0]
                }
            )
            
            # Return aggregated result
            final_result = walk_forward_results[-1]['result']  # Use last period as representative
            final_result.walk_forward_results = walk_forward_analysis
            
            return final_result
        else:
            raise ValueError("No successful walk-forward periods")
    
    async def _store_backtest_results(self, results: BacktestResults):
        """Store backtest results in database"""
        
        try:
            async with get_db_connection() as conn:
                
                # Store main results
                result_id = str(hash(f"{results.config.start_date}{results.config.end_date}{results.config.symbols}"))
                
                await conn.execute("""
                    INSERT INTO backtest_results (
                        result_id, start_date, end_date, symbols, initial_capital,
                        total_return, annualized_return, volatility, sharpe_ratio,
                        max_drawdown, total_trades, win_rate, execution_time
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (result_id) DO UPDATE SET
                        total_return = EXCLUDED.total_return,
                        annualized_return = EXCLUDED.annualized_return,
                        volatility = EXCLUDED.volatility,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        total_trades = EXCLUDED.total_trades,
                        win_rate = EXCLUDED.win_rate,
                        execution_time = EXCLUDED.execution_time
                """, (
                    result_id,
                    results.config.start_date,
                    results.config.end_date,
                    json.dumps(results.config.symbols),
                    results.config.initial_capital,
                    results.total_return,
                    results.annualized_return,
                    results.volatility,
                    results.sharpe_ratio,
                    results.max_drawdown,
                    results.total_trades,
                    results.win_rate,
                    results.execution_time
                ))
                
                # Store trades
                for trade in results.trades:
                    await conn.execute("""
                        INSERT INTO backtest_trades (
                            result_id, entry_date, symbol, side, entry_price,
                            quantity, pnl, return_pct, transaction_cost
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, (
                        result_id,
                        trade.entry_date,
                        trade.symbol,
                        trade.side,
                        trade.entry_price,
                        trade.quantity,
                        trade.pnl,
                        trade.return_pct,
                        trade.transaction_cost
                    ))
        
        except Exception as e:
            logger.error(f"Error storing backtest results: {e}")
    
    async def compare_strategies(
        self,
        strategies: Dict[str, Callable],
        config: BacktestConfig
    ) -> Dict[str, BacktestResults]:
        """Compare multiple strategies"""
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")
                result = await self.run_backtest(config, strategy_func)
                results[strategy_name] = result
                
            except Exception as e:
                logger.error(f"Error backtesting strategy {strategy_name}: {e}")
        
        return results
    
    async def get_strategy_ranking(
        self,
        results: Dict[str, BacktestResults],
        ranking_metric: str = "sharpe_ratio"
    ) -> List[Tuple[str, float]]:
        """Rank strategies by specified metric"""
        
        rankings = []
        
        for strategy_name, result in results.items():
            metric_value = getattr(result, ranking_metric, 0.0)
            rankings.append((strategy_name, metric_value))
        
        # Sort by metric (descending for most metrics)
        reverse_sort = ranking_metric not in ['max_drawdown', 'volatility']
        rankings.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        return rankings 