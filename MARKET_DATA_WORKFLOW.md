# Market Data Workflow Documentation

## Overview

This document describes the complete workflow for fetching real-time market data from free APIs in the Octopus Trading Platform.

## Architecture

### Components

1. **Market Data Service** (`src/services/market_data_service.py`)
   - Unified service layer for fetching and storing market data
   - Handles caching, database persistence, and error handling
   - Integrates with multiple free API providers

2. **Celery Tasks** (`src/data_processing/market_data_tasks.py`)
   - Background tasks for scheduled data fetching
   - Supports single symbol, batch, and watchlist fetching
   - Automatic cleanup of old data

3. **API Endpoints** (`src/api/endpoints/market_data_workflow.py`)
   - RESTful API for fetching market data
   - Supports synchronous and asynchronous operations
   - Includes authentication and rate limiting

4. **Frontend Component** (`frontend-nextjs/src/components/market-data/market-data-manager.tsx`)
   - User interface for managing market data
   - View data sources status
   - Fetch and display market data
   - Historical data viewer

## Free API Providers

The system uses multiple free API providers with intelligent fallback:

### Stock Data Sources
- **Yahoo Finance** (Primary) - No API key, high reliability
- **Finnhub** - 60 calls/minute (free tier)
- **Alpha Vantage** - 25 calls/day (free tier)
- **IEX Cloud** - Free tier available
- **Financial Modeling Prep** - 250 calls/day (free tier)
- **Twelve Data** - 800 calls/day (free tier)

### Crypto Data Sources
- **CoinGecko** - No API key needed, 100 calls/minute
- **Binance Public API** - No API key, 1200 requests/minute
- **CryptoCompare** - Free tier available

## Workflow

### 1. Fetching Market Data

#### Synchronous Fetch
```bash
POST /api/market-data/fetch/batch
{
  "symbols": ["AAPL", "MSFT", "BTC-USD"],
  "force_refresh": false
}
```

#### Asynchronous Fetch (Celery Task)
```bash
POST /api/market-data/fetch/async
{
  "symbols": ["AAPL", "MSFT"],
  "force_refresh": false
}
# Returns task_id for tracking
```

#### Single Symbol
```bash
GET /api/market-data/fetch/{symbol}?force_refresh=false
```

### 2. Scheduled Tasks (Celery Beat)

The system automatically fetches data on a schedule:

- **Watchlist Refresh**: Every 5 minutes
  - Fetches data for default watchlist symbols
  - Task: `market_data.fetch_watchlist`

- **Portfolio Symbols**: Every 10 minutes
  - Updates all symbols in active portfolios
  - Task: `market_data.update_portfolio_symbols`

- **Data Cleanup**: Daily at 2 AM
  - Removes data older than 30 days
  - Task: `market_data.cleanup_old_data`

### 3. Data Storage

Market data is stored in the `financial_time_series` table:
- `time`: Timestamp
- `symbol`: Symbol (e.g., 'AAPL', 'BTC-USD')
- `price`: Current price
- `open`: Open price
- `high`: High price
- `low`: Low price
- `volume`: Trading volume
- `exchange`: Data source name

### 4. Caching Strategy

- **In-Memory Cache**: 1 minute TTL per symbol
- **Database Cache**: Recent data checked before API calls
- **Redis Cache**: Optional, configured via CacheManager

## API Endpoints

### Fetch Market Data
- `GET /api/market-data/fetch/{symbol}` - Fetch single symbol
- `POST /api/market-data/fetch/batch` - Fetch multiple symbols
- `POST /api/market-data/fetch/async` - Queue async fetch task

### Retrieve Data
- `GET /api/market-data/latest/{symbol}` - Get latest data from DB
- `GET /api/market-data/historical/{symbol}` - Get historical data

### Management
- `GET /api/market-data/sources/status` - Get data source status
- `POST /api/market-data/watchlist/refresh` - Refresh watchlist
- `POST /api/market-data/portfolio/refresh` - Refresh portfolio symbols

## Usage Examples

### Python (Backend)
```python
from src.services.market_data_service import market_data_service
from src.database.postgres_connection import SessionLocal

db = SessionLocal()
# Fetch and store data
market_data = await market_data_service.fetch_and_store('AAPL', db)
print(f"AAPL Price: ${market_data.price}")
```

### JavaScript (Frontend)
```javascript
const response = await axios.post('/api/market-data/fetch/batch', {
  symbols: ['AAPL', 'MSFT', 'GOOGL'],
  force_refresh: false
}, {
  headers: { Authorization: `Bearer ${token}` }
});
```

### Celery Task
```python
from src.data_processing.market_data_tasks import fetch_single_market_data

# Queue task
task = fetch_single_market_data.delay('AAPL', force_refresh=True)
result = task.get(timeout=60)
```

## Configuration

### Environment Variables
- `ALPHA_VANTAGE_API_KEY` - Optional, for Alpha Vantage
- `FINNHUB_API_KEY` - Optional, for Finnhub
- `FMP_API_KEY` - Optional, for Financial Modeling Prep
- `TWELVE_DATA_API_KEY` - Optional, for Twelve Data

### Celery Beat Schedule
Configured in `src/core/celery_app.py`:
```python
beat_schedule={
    'fetch-watchlist-market-data': {
        'task': 'market_data.fetch_watchlist',
        'schedule': 300.0,  # 5 minutes
    },
    'update-portfolio-symbols': {
        'task': 'market_data.update_portfolio_symbols',
        'schedule': 600.0,  # 10 minutes
    },
    'cleanup-old-market-data': {
        'task': 'market_data.cleanup_old_data',
        'schedule': 86400.0,  # 24 hours
    },
}
```

## Error Handling

The system includes comprehensive error handling:
- **API Failures**: Automatic fallback to next available source
- **Rate Limiting**: Automatic rate limit management per source
- **Data Validation**: Ensures price > 0 before storing
- **Database Errors**: Rollback on failure, logging for debugging

## Monitoring

### Data Source Status
```bash
GET /api/market-data/sources/status
```

Returns:
- Reliability score for each source
- Rate limit status
- Requests per minute
- Overall health

### Logging
All operations are logged with appropriate levels:
- `INFO`: Successful fetches
- `WARNING`: Source failures (with fallback)
- `ERROR`: Critical failures

## Best Practices

1. **Use Async Tasks for Large Batches**: For >10 symbols, use async endpoint
2. **Respect Rate Limits**: System handles this automatically, but avoid manual rapid calls
3. **Monitor Source Status**: Check source status regularly
4. **Use Caching**: Don't force refresh unless necessary
5. **Clean Up Old Data**: Automatic cleanup runs daily, but can be triggered manually

## Troubleshooting

### No Data Returned
1. Check if symbol is valid
2. Verify data source status
3. Check database for recent data
4. Review logs for API errors

### Rate Limiting
1. Check source status endpoint
2. Wait for rate limit to reset
3. Use async tasks to spread load
4. Consider adding more API keys

### Database Issues
1. Verify database connection
2. Check table exists: `financial_time_series`
3. Review migration status
4. Check disk space

## Future Enhancements

- [ ] WebSocket streaming for real-time updates
- [ ] More data sources (forex, commodities)
- [ ] Advanced caching strategies
- [ ] Data quality scoring
- [ ] Automatic source rotation
- [ ] Historical data backfilling

