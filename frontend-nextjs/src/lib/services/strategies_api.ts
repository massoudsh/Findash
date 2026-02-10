export interface Strategy {
  id: number;
  name: string;
  description: string;
  strategy_type: string;
  is_active: boolean;
  created_at: string;
  symbols?: string[];
  initial_capital?: number;
}

export interface StrategyPerformance {
  strategy_id: number;
  date: string;
  returns: number;
  cumulative_returns: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

const mockStrategies: Strategy[] = [
  {
    id: 1,
    name: 'Moving Average Crossover',
    description: 'Buy when short MA crosses above long MA, sell when opposite',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2024-01-15T10:00:00Z'
  },
  {
    id: 2,
    name: 'RSI Overbought/Oversold',
    description: 'Trade based on RSI levels above 70 and below 30',
    strategy_type: 'technical',
    is_active: false,
    created_at: '2024-01-10T15:30:00Z'
  },
  {
    id: 3,
    name: 'Bollinger Bands Breakout',
    description: 'Buy when price breaks above upper band, sell when below lower band',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2024-01-12T09:00:00Z'
  },
  {
    id: 4,
    name: 'MACD Signal Line Cross',
    description: 'Buy when MACD crosses above signal line, sell when below',
    strategy_type: 'technical',
    is_active: false,
    created_at: '2024-01-08T13:45:00Z'
  },
  {
    id: 5,
    name: 'Mean Reversion',
    description: 'Buy when price deviates significantly from mean, expecting reversion',
    strategy_type: 'quantitative',
    is_active: true,
    created_at: '2024-01-05T11:20:00Z'
  },
  {
    id: 6,
    name: 'Momentum Trading',
    description: 'Buy assets with upward momentum, sell with downward momentum',
    strategy_type: 'quantitative',
    is_active: false,
    created_at: '2024-01-03T16:10:00Z'
  },
  {
    id: 7,
    name: 'Breakout Strategy',
    description: 'Enter trades when price breaks key support/resistance levels',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2024-01-02T14:00:00Z'
  },
  {
    id: 8,
    name: 'Pairs Trading',
    description: 'Trade correlated asset pairs to exploit price divergences',
    strategy_type: 'statistical',
    is_active: false,
    created_at: '2023-12-30T10:00:00Z'
  },
  {
    id: 9,
    name: 'Arbitrage',
    description: 'Exploit price differences across markets for risk-free profit',
    strategy_type: 'arbitrage',
    is_active: true,
    created_at: '2023-12-28T09:30:00Z'
  },
  {
    id: 10,
    name: 'News Sentiment Trading',
    description: 'Trade based on real-time news sentiment analysis',
    strategy_type: 'sentiment',
    is_active: false,
    created_at: '2023-12-25T12:00:00Z'
  },
  {
    id: 11,
    name: 'Deep Learning Price Prediction',
    description: 'Use LSTM/transformer models to predict price movements',
    strategy_type: 'machine learning',
    is_active: true,
    created_at: '2023-12-20T08:00:00Z'
  },
  {
    id: 12,
    name: 'Grid Trading',
    description: 'Place buy/sell orders at regular price intervals to profit from volatility',
    strategy_type: 'quantitative',
    is_active: false,
    created_at: '2023-12-18T15:00:00Z'
  },
  {
    id: 13,
    name: 'Stochastic Oscillator',
    description: 'Trade based on stochastic momentum indicator signals',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2023-12-15T12:00:00Z'
  },
  {
    id: 14,
    name: 'Support & Resistance',
    description: 'Buy at support levels, sell at resistance levels',
    strategy_type: 'technical',
    is_active: false,
    created_at: '2023-12-12T14:30:00Z'
  },
  {
    id: 15,
    name: 'Volume Weighted Average Price',
    description: 'Execute trades around VWAP for better entry/exit points',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2023-12-10T11:00:00Z'
  },
  {
    id: 16,
    name: 'Ichimoku Cloud',
    description: 'Trade using Ichimoku cloud signals and trend identification',
    strategy_type: 'technical',
    is_active: false,
    created_at: '2023-12-08T16:45:00Z'
  },
  {
    id: 17,
    name: 'Williams %R',
    description: 'Momentum oscillator for overbought/oversold conditions',
    strategy_type: 'technical',
    is_active: true,
    created_at: '2023-12-05T09:15:00Z'
  },
  {
    id: 18,
    name: 'Commodity Channel Index',
    description: 'CCI-based strategy for identifying cyclical trends',
    strategy_type: 'technical',
    is_active: false,
    created_at: '2023-12-03T13:20:00Z'
  },
  {
    id: 19,
    name: 'Kelly Criterion',
    description: 'Optimal position sizing based on win rate and reward/risk',
    strategy_type: 'quantitative',
    is_active: true,
    created_at: '2023-12-01T10:30:00Z'
  },
  {
    id: 20,
    name: 'Black-Litterman',
    description: 'Portfolio optimization using Bayesian approach',
    strategy_type: 'quantitative',
    is_active: false,
    created_at: '2023-11-28T15:00:00Z'
  },
  {
    id: 21,
    name: 'Kalman Filter',
    description: 'Dynamic hedge ratio estimation for pairs trading',
    strategy_type: 'statistical',
    is_active: true,
    created_at: '2023-11-25T12:45:00Z'
  },
  {
    id: 22,
    name: 'Cointegration Strategy',
    description: 'Statistical arbitrage using cointegrated asset pairs',
    strategy_type: 'statistical',
    is_active: false,
    created_at: '2023-11-22T14:15:00Z'
  },
  {
    id: 23,
    name: 'Random Forest Predictor',
    description: 'Ensemble learning for price direction prediction',
    strategy_type: 'machine learning',
    is_active: true,
    created_at: '2023-11-20T11:30:00Z'
  },
  {
    id: 24,
    name: 'XGBoost Classifier',
    description: 'Gradient boosting for market regime classification',
    strategy_type: 'machine learning',
    is_active: false,
    created_at: '2023-11-18T16:00:00Z'
  },
  {
    id: 25,
    name: 'Transformer Attention',
    description: 'Attention mechanism for sequential price prediction',
    strategy_type: 'machine learning',
    is_active: true,
    created_at: '2023-11-15T09:45:00Z'
  },
  {
    id: 26,
    name: 'Reinforcement Learning',
    description: 'Q-learning agent for adaptive trading decisions',
    strategy_type: 'machine learning',
    is_active: false,
    created_at: '2023-11-12T13:30:00Z'
  },
  {
    id: 27,
    name: 'Social Media Sentiment',
    description: 'Twitter/Reddit sentiment analysis for trade signals',
    strategy_type: 'sentiment',
    is_active: true,
    created_at: '2023-11-10T10:20:00Z'
  },
  {
    id: 28,
    name: 'Options Flow Scanner',
    description: 'Unusual options activity detection and following',
    strategy_type: 'options',
    is_active: false,
    created_at: '2023-11-08T15:30:00Z'
  },
  {
    id: 29,
    name: 'Volatility Surface Trading',
    description: 'Trade volatility skew and term structure inefficiencies',
    strategy_type: 'options',
    is_active: true,
    created_at: '2023-11-05T12:15:00Z'
  },
  {
    id: 30,
    name: 'Cross-Exchange Arbitrage',
    description: 'Exploit price differences across multiple exchanges',
    strategy_type: 'arbitrage',
    is_active: false,
    created_at: '2023-11-03T14:45:00Z'
  }
];

const mockStrategyPerformance: { [key: number]: StrategyPerformance[] } = {
  1: [
    { strategy_id: 1, date: '2023-10-01', returns: 0.02, cumulative_returns: 0.02, sharpe_ratio: 1.2, max_drawdown: -0.05 },
    { strategy_id: 1, date: '2023-10-02', returns: 0.015, cumulative_returns: 0.035, sharpe_ratio: 1.25, max_drawdown: -0.03 },
    { strategy_id: 1, date: '2023-10-03', returns: -0.01, cumulative_returns: 0.025, sharpe_ratio: 1.1, max_drawdown: -0.05 },
  ],
  3: [
    { strategy_id: 3, date: '2023-10-01', returns: 0.025, cumulative_returns: 0.025, sharpe_ratio: 1.5, max_drawdown: -0.02 },
    { strategy_id: 3, date: '2023-10-02', returns: 0.02, cumulative_returns: 0.045, sharpe_ratio: 1.6, max_drawdown: -0.02 },
    { strategy_id: 3, date: '2023-10-03', returns: 0.015, cumulative_returns: 0.061, sharpe_ratio: 1.7, max_drawdown: -0.01 },
  ],
};

export async function getStrategies(): Promise<{ data: Strategy[] }> {
  console.log("Mock API: Fetching strategies data.");
  return new Promise(resolve => {
    setTimeout(() => resolve({ data: mockStrategies }), 300);
  });
}

export async function getStrategy(id: number): Promise<{ data: Strategy }> {
  console.log(`Mock API: Fetching strategy ${id}.`);
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const strategy = mockStrategies.find(s => s.id === id);
      if (strategy) {
        resolve({ data: strategy });
      } else {
        reject(new Error(`Strategy ${id} not found`));
      }
    }, 200);
  });
}

export async function getStrategyPerformance(strategyId: number): Promise<{ data: StrategyPerformance[] }> {
  console.log(`Mock API: Fetching performance for strategy ${strategyId}.`);
  return new Promise(resolve => {
    setTimeout(() => {
      const performance = mockStrategyPerformance[strategyId] || [];
      resolve({ data: performance });
    }, 400);
  });
} 