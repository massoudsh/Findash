import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from market_data.models import StockData
from Modules.M6.Risk.Management.portfolio_metrics import PortfolioMetrics
from Modules.M6.Risk.Management.portfolio_optimization import PortfolioOptimizer
from Modules.M1.Collecting.Data.Scraping import fetch_real_time_data
from Modules.M2.Data.Warehouse import load_historical_data
from Modules.M8.Paper.Trading.execution import execute_order
from Modules.M4.Strategies.risk_aware_strategy import MomentumRiskAwareStrategy
from Modules.M3.Real_Time.Processing import StreamProcessor
from Modules.M9.Market.Sentiment import SentimentAnalyzer
from Modules.M10.Backtesting import Backtest
from .VIS import FinancialVisualizer

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1('Risk-Aware Trading Strategy Dashboard'),
    
    # Strategy Parameters
    html.Div([
        html.H3('Strategy Parameters'),
        html.Div([
            html.Label('Risk Budget (%):'),
            dcc.Slider(
                id='risk-budget-slider',
                min=1,
                max=10,
                step=0.5,
                value=5,
                marks={i: f'{i}%' for i in range(1, 11)}
            ),
        ]),
        html.Div([
            html.Label('Max Drawdown Limit (%):'),
            dcc.Slider(
                id='max-drawdown-slider',
                min=5,
                max=30,
                step=1,
                value=15,
                marks={i: f'{i}%' for i in range(5, 31, 5)}
            ),
        ]),
        html.Div([
            html.Label('Target Sharpe Ratio:'),
            dcc.Slider(
                id='target-sharpe-slider',
                min=0.5,
                max=2.0,
                step=0.1,
                value=1.0,
                marks={i/2: str(i/2) for i in range(1, 5)}
            ),
        ]),
    ], style={'padding': '20px', 'margin': '10px', 'border': '1px solid #ddd'}),
    
    # Technical Analysis
    html.Div([
        html.H3('Technical Analysis'),
        dcc.Graph(id='technical-analysis-chart'),
    ]),
    
    # Portfolio Performance
    html.Div([
        html.H3('Portfolio Performance'),
        dcc.Graph(id='portfolio-value-chart'),
        dcc.Graph(id='drawdown-chart'),
    ]),
    
    # Risk Metrics
    html.Div([
        html.H3('Risk Metrics'),
        html.Div([
            html.Div([
                html.H4('Current Risk Metrics'),
                html.Div(id='risk-metrics-display'),
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H4('Historical Risk Metrics'),
                dcc.Graph(id='risk-metrics-chart'),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ]),
    ]),
    
    # Portfolio Allocation
    html.Div([
        html.H3('Portfolio Allocation'),
        dcc.Graph(id='allocation-pie-chart'),
    ]),
    
    # Statistical Analysis
    html.Div([
        html.H3('Statistical Analysis'),
        dcc.Graph(id='statistical-analysis-chart'),
    ]),
    
    # Update Interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds (1 minute)
        n_intervals=0
    ),
    
    # Store for strategy state
    dcc.Store(id='strategy-state'),
])

@app.callback(
    [Output('technical-analysis-chart', 'figure'),
     Output('portfolio-value-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('risk-metrics-chart', 'figure'),
     Output('allocation-pie-chart', 'figure'),
     Output('statistical-analysis-chart', 'figure'),
     Output('risk-metrics-display', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('risk-budget-slider', 'value'),
     Input('max-drawdown-slider', 'value'),
     Input('target-sharpe-slider', 'value')]
)
def update_dashboard(n_intervals, risk_budget, max_drawdown, target_sharpe):
    # Initialize or update strategy
    strategy = MomentumRiskAwareStrategy(
        risk_budget=risk_budget/100,  # Convert to decimal
        max_drawdown_limit=max_drawdown/100,
        target_sharpe=target_sharpe,
        rebalance_frequency='D'
    )
    
    # Get latest data and update strategy
    prices = load_historical_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    strategy.load_data(prices)
    
    # Initialize FinancialVisualizer
    viz = FinancialVisualizer(prices)
    
    # Create technical analysis chart
    technical_fig = viz.plot_technical_analysis()
    
    # Calculate portfolio metrics
    portfolio_metrics = PortfolioMetrics()
    metrics = portfolio_metrics.calculate_metrics(strategy.returns)
    
    # Create portfolio value chart
    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(go.Scatter(
        x=strategy.returns.index,
        y=strategy.portfolio_value,
        mode='lines',
        name='Portfolio Value'
    ))
    portfolio_fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)'
    )
    
    # Create drawdown chart
    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(
        x=strategy.returns.index,
        y=strategy.drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    drawdown_fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)'
    )
    
    # Create risk metrics chart
    risk_fig = go.Figure()
    risk_fig.add_trace(go.Scatter(
        x=strategy.returns.index,
        y=metrics['rolling_sharpe'],
        mode='lines',
        name='Rolling Sharpe Ratio'
    ))
    risk_fig.add_trace(go.Scatter(
        x=strategy.returns.index,
        y=metrics['rolling_volatility'],
        mode='lines',
        name='Rolling Volatility'
    ))
    risk_fig.update_layout(
        title='Risk Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Value'
    )
    
    # Create allocation pie chart
    allocation_fig = go.Figure(data=[go.Pie(
        labels=strategy.weights.index,
        values=strategy.weights.values,
        hole=.3
    )])
    allocation_fig.update_layout(
        title='Current Portfolio Allocation'
    )
    
    # Create statistical analysis chart
    statistical_fig = viz.plot_statistical_analysis()
    
    # Create risk metrics display
    risk_metrics_display = html.Div([
        html.P(f'Current Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}'),
        html.P(f'Current Volatility: {metrics["volatility"]:.2%}'),
        html.P(f'Maximum Drawdown: {metrics["max_drawdown"]:.2%}'),
        html.P(f'Value at Risk (95%): {metrics["var_95"]:.2%}'),
        html.P(f'Expected Shortfall: {metrics["expected_shortfall"]:.2%}')
    ])
    
    return technical_fig, portfolio_fig, drawdown_fig, risk_fig, allocation_fig, statistical_fig, risk_metrics_display

# 1. Data Collection (M1)
def setup_strategy():
    # Load historical data from your data warehouse (M2)
    prices = load_historical_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Create strategy instance
    strategy = MomentumRiskAwareStrategy(
        risk_budget=0.05,          # 5% CVaR limit
        max_drawdown_limit=0.15,   # 15% maximum drawdown
        target_sharpe=1.0,         # Target Sharpe ratio
        rebalance_frequency='M',   # Monthly rebalancing
        lookback_period=60,        # 60-day momentum
        momentum_percentile=0.7     # Top 70% momentum stocks
    )
    
    # Load data into strategy
    strategy.load_data(prices)
    
    return strategy

# 2. Strategy Backtesting
def run_backtest(strategy):
    results = strategy.backtest(
        initial_capital=1_000_000,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Generate report
    report = strategy.generate_report(output_format='html')
    
    return results, report

# 3. Live Trading Integration
def run_live_trading(strategy):
    while True:
        # Get real-time data from M1
        current_prices = fetch_real_time_data(symbols=strategy.returns.columns)
        
        # Update strategy data
        strategy.load_data(current_prices)
        
        # Check if rebalancing is needed
        if strategy.should_rebalance(pd.Timestamp.now()):
            # Optimize portfolio
            weights = strategy.optimize_portfolio(optimization_method='risk_budget')
            
            # Calculate positions
            positions = strategy.calculate_position_sizes(
                capital=1_000_000,
                weights=weights,
                prices=current_prices
            )
            
            # Execute trades through M8 Paper Trading
            trades = strategy.execute_trades(positions, strategy.current_positions)
            
            # Log trades and update metrics
            strategy.update_portfolio_metrics(positions, pd.Timestamp.now())

class SentimentAwareRiskStrategy(MomentumRiskAwareStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def optimize_portfolio(self, *args, **kwargs):
        # Include sentiment in optimization
        sentiment_scores = self.sentiment_analyzer.get_scores(self.returns.columns)
        # Modify optimization based on sentiment
        return super().optimize_portfolio(*args, **kwargs)

def comprehensive_backtest(strategy):
    backtest = Backtest(
        strategy=strategy,
        data_provider=load_historical_data,
        risk_metrics=PortfolioMetrics()
    )
    
    results = backtest.run(
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=1_000_000
    )
    
    return results

def main():
    # Initialize strategy
    strategy = setup_strategy()
    
    # Run backtest first
    backtest_results, report = run_backtest(strategy)
    
    # If backtest is successful, run live trading
    if backtest_results['sharpe'].iloc[-1] > 1.0:
        # Initialize real-time components
        real_time_strategy = RealTimeRiskStrategy(
            risk_budget=0.05,
            max_drawdown_limit=0.15,
            rebalance_frequency='D'  # Daily for live trading
        )
        
        # Start live trading
        run_live_trading(real_time_strategy)
    
    # Generate visualizations (M11)
    from Modules.M11.Visualization import create_dashboard
    dashboard = create_dashboard(backtest_results, report)
    dashboard.display()

if __name__ == "__main__":
    main()
