# 🎬 Octopus Trading Platform - Interactive Demo

## 🚀 Live Demo Showcase

Welcome to the Octopus Trading Platform demo! This interactive showcase demonstrates the key features and capabilities of our AI-powered trading platform.

---

## 📊 Dashboard Demo

### Real-Time Portfolio Overview

```typescript
// Example: Fetching portfolio data
const portfolio = await fetch('/api/portfolio');
const data = await portfolio.json();

// Display real-time metrics
{
  totalValue: "$1,250,000",
  dailyChange: "+2.5%",
  activePositions: 12,
  unrealizedPnL: "+$15,250"
}
```

**Features Demonstrated:**
- ✅ Real-time portfolio valuation
- ✅ Live P&L tracking
- ✅ Position monitoring
- ✅ Performance metrics

---

## 💹 Trading Interface Demo

### Order Entry Example

```typescript
// Example: Placing a trade order
const order = {
  symbol: "AAPL",
  side: "buy",
  quantity: 100,
  orderType: "limit",
  price: 175.50,
  strategy: "momentum"
};

const response = await fetch('/api/trades', {
  method: 'POST',
  body: JSON.stringify(order)
});
```

**Interactive Demo:**
- 📈 Live orderbook visualization
- 🎯 One-click order execution
- ⚡ Real-time price updates
- 🛡️ Risk validation before execution

---

## 🤖 AI-Powered Predictions Demo

### Market Prediction Example

```python
# Example: AI model prediction
from octopus.ml import PricePredictor

predictor = PricePredictor()
prediction = predictor.predict(
    symbol="BTC-USD",
    timeframe="1h",
    features=["price", "volume", "sentiment"]
)

# Output:
{
    "predicted_price": 45250.00,
    "confidence": 0.87,
    "direction": "bullish",
    "time_horizon": "1h"
}
```

**AI Capabilities:**
- 🧠 Price prediction models
- 📊 Sentiment analysis
- 🎯 Strategy recommendations
- 📈 Trend identification

---

## 📈 Real-Time Market Data Demo

### WebSocket Data Stream

```javascript
// Example: Real-time market data subscription
const ws = new WebSocket('ws://localhost:8000/ws/market-data');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Live price update:', {
    symbol: data.symbol,
    price: data.price,
    volume: data.volume,
    timestamp: data.timestamp
  });
};

// Subscribe to BTC-USD
ws.send(JSON.stringify({
  action: 'subscribe',
  symbol: 'BTC-USD'
}));
```

**Real-Time Features:**
- ⚡ Live price feeds
- 📊 Orderbook updates
- 💹 Trade execution alerts
- 📡 WebSocket streaming

---

## 🎯 Strategy Backtesting Demo

### Backtest Example

```python
# Example: Running a backtest
from octopus.strategies import BacktestRunner

strategy = {
    "name": "Moving Average Crossover",
    "indicators": ["SMA_50", "SMA_200"],
    "entry": "SMA_50 > SMA_200",
    "exit": "SMA_50 < SMA_200"
}

backtest = BacktestRunner(
    strategy=strategy,
    symbol="SPY",
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=100000
)

results = backtest.run()

# Results:
{
    "total_return": 15.5,
    "sharpe_ratio": 1.8,
    "max_drawdown": -8.2,
    "win_rate": 0.62,
    "total_trades": 45
}
```

**Backtesting Features:**
- 📊 Historical performance analysis
- 🎯 Strategy optimization
- 📈 Risk metrics calculation
- 🔍 Parameter tuning

---

## 🛡️ Risk Management Demo

### Risk Assessment Example

```python
# Example: Portfolio risk analysis
from octopus.risk import RiskAnalyzer

analyzer = RiskAnalyzer(portfolio)
risk_metrics = analyzer.analyze()

# Risk Metrics:
{
    "var_95": -12500,  # Value at Risk (95% confidence)
    "expected_shortfall": -18500,
    "beta": 1.2,
    "sharpe_ratio": 1.5,
    "max_drawdown": -12.5,
    "correlation_matrix": {...}
}
```

**Risk Features:**
- 🛡️ VaR calculations
- 📊 Stress testing
- 🎯 Portfolio optimization
- ⚠️ Risk alerts

---

## 🔌 API Demo

### RESTful API Examples

#### Get Market Data
```bash
curl -X GET "http://localhost:8000/api/market-data/BTC-USD" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Place Order
```bash
curl -X POST "http://localhost:8000/api/trades" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 10,
    "order_type": "limit",
    "price": 175.50
  }'
```

#### Get Portfolio
```bash
curl -X GET "http://localhost:8000/api/portfolio" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## 🎨 UI Component Showcase

### Dashboard Widgets

```tsx
// Example: Dashboard component
import { Dashboard } from '@/components/dashboard';

<Dashboard
  portfolio={portfolioData}
  marketData={marketData}
  positions={positions}
  onTrade={handleTrade}
/>
```

### Trading Interface

```tsx
// Example: Trading component
import { TradingInterface } from '@/components/trading';

<TradingInterface
  symbol="AAPL"
  orderbook={orderbookData}
  onOrderSubmit={handleOrder}
  riskCheck={true}
/>
```

### Real-Time Charts

```tsx
// Example: Chart component
import { PriceChart } from '@/components/charts';

<PriceChart
  symbol="BTC-USD"
  timeframe="1h"
  indicators={['SMA_50', 'RSI']}
  realTime={true}
/>
```

---

## 🧪 Try It Yourself

### Quick Start Demo

1. **Start the Platform**
   ```bash
   # Backend
   cd Modules
   uvicorn src.main_refactored:app --reload
   
   # Frontend
   cd frontend-nextjs
   npm run dev
   ```

2. **Access Demo**
   - Frontend: http://localhost:3002
   - API Docs: http://localhost:8000/docs
   - WebSocket: ws://localhost:8000/ws

3. **Try Features**
   - View dashboard with mock data
   - Explore trading interface
   - Test AI predictions
   - Run backtests
   - Analyze risk metrics

---

## 📸 Demo Screenshots

### Dashboard View
![Dashboard](https://via.placeholder.com/1200x600/1a1a1a/ffffff?text=Dashboard+with+Real-time+Portfolio+Data)

### Trading Interface
![Trading](https://via.placeholder.com/1200x600/1a1a1a/ffffff?text=Trading+Center+with+Orderbook+and+Charts)

### AI Predictions
![AI](https://via.placeholder.com/1200x600/1a1a1a/ffffff?text=AI+Model+Predictions+and+Insights)

### Risk Analysis
![Risk](https://via.placeholder.com/1200x600/1a1a1a/ffffff?text=Risk+Management+Dashboard)

---

## 🎯 Demo Scenarios

### Scenario 1: Day Trading
1. Monitor real-time market data
2. Identify trading opportunities using AI
3. Execute trades with risk validation
4. Track performance in real-time

### Scenario 2: Portfolio Management
1. Analyze current portfolio
2. Run risk assessment
3. Optimize allocation
4. Generate performance reports

### Scenario 3: Strategy Development
1. Create custom trading strategy
2. Backtest on historical data
3. Optimize parameters
4. Deploy as automated bot

---

## 🔗 Demo Links

- **Live Demo**: [Coming Soon]
- **API Documentation**: http://localhost:8000/docs
- **WebSocket Test**: ws://localhost:8000/ws
- **GitHub Repository**: https://github.com/massoudsh/Findash

---

## 💡 Demo Tips

1. **Use Mock Data**: The demo includes realistic mock data for testing
2. **Paper Trading**: All trades are simulated - no real money at risk
3. **API Testing**: Use the interactive API docs at `/docs`
4. **WebSocket**: Connect to see real-time data streaming
5. **Customization**: Modify demo data in `src/lib/services/`

---

*This demo showcases the core capabilities of the Octopus Trading Platform. For production deployment, please refer to the [Installation Guide](../README.md#installation).*
