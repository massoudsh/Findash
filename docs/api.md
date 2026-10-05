# 🔗 API Documentation - Octopus Trading Platform™

## Overview

The Octopus Trading Platform provides a comprehensive RESTful API built with FastAPI, offering real-time market data, portfolio management, risk analysis, and advanced trading capabilities.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

All API endpoints require authentication via JWT tokens.

### Getting Started

1. **Register/Login** to get access token:
```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@octopus.trading", "password": "demo123"}'
```

2. **Use the token** in subsequent requests:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/portfolios/"
```

## Core Endpoints

### 🔐 Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/me` - Get current user profile

### 📊 Market Data
- `GET /api/market/quote/{symbol}` - Real-time quote
- `GET /api/market/historical/{symbol}` - Historical data
- `GET /api/market/search` - Symbol search
- `GET /api/market/trending` - Trending stocks

### 💼 Portfolio Management
- `GET /api/portfolios/` - List user portfolios
- `POST /api/portfolios/` - Create new portfolio
- `GET /api/portfolios/{id}` - Get portfolio details
- `PUT /api/portfolios/{id}` - Update portfolio
- `DELETE /api/portfolios/{id}` - Delete portfolio

### 📈 Positions
- `GET /api/portfolios/{id}/positions` - List positions
- `POST /api/portfolios/{id}/positions` - Add position
- `PUT /api/positions/{id}` - Update position
- `DELETE /api/positions/{id}` - Close position

### 🎯 Orders
- `POST /api/orders/` - Place order
- `GET /api/orders/` - List orders
- `GET /api/orders/{id}` - Get order details
- `PUT /api/orders/{id}/cancel` - Cancel order

### ⚠️ Risk Management
- `GET /api/risk/portfolio/{id}` - Portfolio risk metrics
- `GET /api/risk/var` - Value at Risk calculation
- `GET /api/risk/stress-test` - Stress testing
- `GET /api/risk/correlation` - Asset correlation

### 🤖 AI/ML Services
- `POST /api/ml/predict` - Price predictions
- `GET /api/ml/sentiment/{symbol}` - Sentiment analysis
- `POST /api/ml/backtest` - Strategy backtesting
- `GET /api/ml/recommendations` - AI recommendations

### 📰 Alternative Data
- `GET /api/alt-data/news/{symbol}` - News sentiment
- `GET /api/alt-data/social/{symbol}` - Social media data
- `GET /api/alt-data/macro` - Macroeconomic indicators
- `GET /api/alt-data/crypto` - Cryptocurrency data

### 🔔 Notifications
> The historical `/api/notifications/rules` catalog below is **not** the current price-alert API.
> Live alert rules live at `/api/alerts/*`. In-app rows are written to `notifications` by `src/services/notifications.py`; there is no dedicated list router in `src/main_refactored.py`.

### Account, payments, and ops (verified 2026-09)

Registered in `src/main_refactored.py`. Full workflows, examples, and constraints: [ACCOUNT_PLATFORM.md](ACCOUNT_PLATFORM.md).

| Prefix | Router | Purpose |
|--------|--------|---------|
| `/api/payment/zarinpal` | `payment_zarinpal.py` | Create / callback / status / history |
| `/api/wallet` | `wallet.py` | IRT balances, Sheba, deposit, withdraw |
| `/api/subscriptions` | `subscriptions.py` | Plans, subscribe, `GET /me` |
| `/api/kyc` | `kyc.py` | Submit / status / admin review |
| `/api/alerts` | `price_alerts.py` | Price rules, Web Push, admin evaluate |
| `/api/risk-policy` | `risk_policy.py` | Policy CRUD, breaches, evaluate |
| `/api/admin` | `admin_panel.py` | Users, roles, audit log |
| `/api/reports` | `pdf_reports.py` | Persian portfolio PDF |
| `/api/auth/send-otp`, `/verify-otp` | `otp_auth.py` | SMS OTP login |
| `/api/trading-bots/{id}/start` | `trading_bots.py` | **402** unless active subscription (admins exempt) |

```bash
# Wallet deposit (credits only after ZarinPal verify)
curl -X POST "http://localhost:8011/api/wallet/deposit" \
  -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" \
  -d '{"amount_toman": 100000}'

# Subscription status
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8011/api/subscriptions/me"
```

## WebSocket Endpoints

### Real-time Data Streams

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/market-data');

// Subscribe to real-time quotes
ws.send(JSON.stringify({
  type: 'subscribe',
  symbols: ['AAPL', 'GOOGL', 'MSFT']
}));

// Receive real-time updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Real-time quote:', data);
};
```

### Available WebSocket Channels
- `/ws/market-data` - Real-time market data
- `/ws/portfolio/{id}` - Portfolio updates
- `/ws/notifications` - Alert notifications
- `/ws/trading` - Trade execution updates

## Request/Response Examples

### Market Data Request
```bash
curl "http://localhost:8000/api/market/quote/AAPL" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "symbol": "AAPL",
  "price": 175.84,
  "change": 2.15,
  "change_percent": 1.24,
  "volume": 89234567,
  "market_cap": 2800000000000,
  "pe_ratio": 28.5,
  "timestamp": "2024-01-15T16:30:00Z"
}
```

### Create Portfolio Request
```bash
curl -X POST "http://localhost:8000/api/portfolios/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Growth Portfolio",
    "description": "Long-term growth focused portfolio",
    "initial_cash": 100000
  }'
```

**Response:**
```json
{
  "id": "uuid-here",
  "name": "Growth Portfolio",
  "description": "Long-term growth focused portfolio",
  "initial_cash": 100000.00,
  "current_cash": 100000.00,
  "total_value": 100000.00,
  "created_at": "2024-01-15T10:00:00Z"
}
```

### Place Order Request
```bash
curl -X POST "http://localhost:8000/api/orders/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "uuid-here",
    "symbol": "AAPL",
    "side": "buy",
    "order_type": "limit",
    "quantity": 100,
    "price": 175.00
  }'
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```json
{
  "error": "ValidationError",
  "message": "Invalid symbol format",
  "details": {
    "field": "symbol",
    "code": "INVALID_FORMAT"
  },
  "timestamp": "2024-01-15T10:00:00Z"
}
```

### Common Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `402` - Payment Required (active subscription missing — trading-bot start)
- `403` - Forbidden
- `404` - Not Found (also used when a PDF/portfolio is not owned by the caller)
- `422` - Validation Error
- `429` - OTP send/verify rate limit
- `500` - Internal Server Error
- `502` - ZarinPal create/verify upstream error
- `503` - Missing merchant, missing Persian PDF font, or backend unreachable

## Rate Limiting

- **Default**: 100 requests per minute
- **Burst**: 20 requests in 10 seconds
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## SDK Examples

### Python SDK
```python
from octopus_trading import OctopusClient

client = OctopusClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Get real-time quote
quote = client.market.get_quote("AAPL")
print(f"AAPL: ${quote.price}")

# Create portfolio
portfolio = client.portfolios.create(
    name="My Portfolio",
    initial_cash=100000
)

# Place order
order = client.orders.place(
    portfolio_id=portfolio.id,
    symbol="AAPL",
    side="buy",
    quantity=100,
    order_type="market"
)
```

### JavaScript SDK
```javascript
import { OctopusTrading } from '@octopus/trading-sdk';

const client = new OctopusTrading({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Get portfolio
const portfolio = await client.portfolios.get('portfolio-id');

// Subscribe to real-time data
client.subscribe('AAPL', (quote) => {
  console.log(`AAPL: $${quote.price}`);
});
```

## Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

## Support

- **Documentation**: https://docs.octopus.trading
- **API Status**: https://status.octopus.trading
- **Support**: api-support@octopus.trading

---

*Last updated: September 2026 — account-platform routes verified against `src/main_refactored.py`.* 