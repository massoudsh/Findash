# Database Query Guide - Octopus Trading Platform

## Database Connection Details

- **Host**: localhost
- **Port**: 5433 (mapped from container port 5432)
- **Database**: trading_db
- **User**: postgres
- **Password**: postgres
- **Connection String**: `postgresql://postgres:postgres@localhost:5433/trading_db`

---

## Method 1: Direct PostgreSQL Connection (psql)

### Connect via Docker
```bash
docker exec -it octopus-db psql -U postgres -d trading_db
```

### Connect via Local psql (if installed)
```bash
psql -h localhost -p 5433 -U postgres -d trading_db
```

### Useful SQL Queries

```sql
-- List all tables
\dt

-- List all users
SELECT id, username, email, is_active, created_at FROM users;

-- List all portfolios
SELECT p.id, u.username, p.name, p.total_value, p.current_cash 
FROM portfolios p 
JOIN users u ON p.user_id = u.id;

-- List all trades
SELECT t.id, p.name as portfolio, t.symbol, t.trade_type, 
       t.quantity, t.price, t.total_amount, t.trade_date, t.status
FROM trades t
JOIN portfolios p ON t.portfolio_id = p.id
ORDER BY t.trade_date DESC
LIMIT 20;

-- List all positions
SELECT pos.id, p.name as portfolio, pos.symbol, pos.quantity, 
       pos.average_price, pos.current_price, pos.market_value, pos.unrealized_pnl
FROM positions pos
JOIN portfolios p ON pos.portfolio_id = p.id
WHERE pos.is_active = true;

-- Get portfolio performance
SELECT 
    p.name,
    p.total_value,
    p.current_cash,
    (p.total_value - p.initial_cash) as total_pnl,
    ((p.total_value - p.initial_cash) / p.initial_cash * 100) as pnl_percent
FROM portfolios p;

-- Get recent market data
SELECT symbol, price, volume, time 
FROM market_data 
ORDER BY time DESC 
LIMIT 50;

-- Count records per table
SELECT 
    'users' as table_name, COUNT(*) as count FROM users
UNION ALL
SELECT 'portfolios', COUNT(*) FROM portfolios
UNION ALL
SELECT 'trades', COUNT(*) FROM trades
UNION ALL
SELECT 'positions', COUNT(*) FROM positions
UNION ALL
SELECT 'market_data', COUNT(*) FROM market_data;
```

---

## Method 2: Python Script (SQLAlchemy)

### Create a query script

```python
# query_db.py
from src.database.postgres_connection import get_db_session
from src.database.models import User, Portfolio, Trade, Position, MarketData

# Example 1: Get all users
with get_db_session() as db:
    users = db.query(User).all()
    for user in users:
        print(f"User: {user.username} ({user.email})")

# Example 2: Get portfolios with user info
with get_db_session() as db:
    portfolios = db.query(Portfolio).join(User).all()
    for portfolio in portfolios:
        print(f"Portfolio: {portfolio.name} - Owner: {portfolio.user.username} - Value: ${portfolio.total_value}")

# Example 3: Get recent trades
from datetime import datetime, timedelta
with get_db_session() as db:
    recent_trades = db.query(Trade).filter(
        Trade.trade_date >= datetime.now() - timedelta(days=7)
    ).order_by(Trade.trade_date.desc()).all()
    
    for trade in recent_trades:
        print(f"{trade.trade_type} {trade.quantity} {trade.symbol} @ ${trade.price}")

# Example 4: Custom SQL query
with get_db_session() as db:
    result = db.execute("""
        SELECT 
            p.name as portfolio_name,
            COUNT(t.id) as trade_count,
            SUM(t.total_amount) as total_traded
        FROM portfolios p
        LEFT JOIN trades t ON p.id = t.portfolio_id
        GROUP BY p.id, p.name
        ORDER BY total_traded DESC
    """)
    
    for row in result:
        print(f"{row.portfolio_name}: {row.trade_count} trades, ${row.total_traded} total")

# Example 5: Get active positions
with get_db_session() as db:
    positions = db.query(Position).filter(
        Position.is_active == True
    ).all()
    
    for pos in positions:
        print(f"{pos.symbol}: {pos.quantity} shares @ ${pos.average_price} avg")
```

### Run the script
```bash
cd /Users/massoudshemirani/MyProjects/Octopus/Findash
python query_db.py
```

---

## Method 3: Using Python Interactive Shell

```bash
cd /Users/massoudshemirani/MyProjects/Octopus/Findash
python3
```

```python
from src.database.postgres_connection import get_db_session
from src.database.models import *

# Query users
with get_db_session() as db:
    users = db.query(User).limit(5).all()
    print([u.username for u in users])

# Query with filters
with get_db_session() as db:
    active_portfolios = db.query(Portfolio).filter(
        Portfolio.is_active == True
    ).all()
    print([p.name for p in active_portfolios])

# Join queries
with get_db_session() as db:
    result = db.query(Trade, Portfolio, User).join(
        Portfolio, Trade.portfolio_id == Portfolio.id
    ).join(
        User, Portfolio.user_id == User.id
    ).limit(10).all()
    
    for trade, portfolio, user in result:
        print(f"{user.username} - {portfolio.name} - {trade.symbol}")
```

---

## Method 4: Through API Endpoints

### Get portfolios via API
```bash
curl http://localhost:8000/api/portfolios
```

### Get trades via API
```bash
curl http://localhost:8000/api/trades?portfolio_id=1
```

### Get market data via API
```bash
curl http://localhost:8000/api/market-data/BTC-USD
```

---

## Method 5: Using Database GUI Tools

### Option A: pgAdmin
1. Download pgAdmin: https://www.pgadmin.org/
2. Add server:
   - Host: localhost
   - Port: 5433
   - Database: trading_db
   - Username: postgres
   - Password: postgres

### Option B: DBeaver
1. Download DBeaver: https://dbeaver.io/
2. Create new PostgreSQL connection:
   - Host: localhost
   - Port: 5433
   - Database: trading_db
   - Username: postgres
   - Password: postgres

### Option C: TablePlus
1. Download TablePlus: https://tableplus.com/
2. Create PostgreSQL connection with same credentials

---

## Method 6: Using Python with Raw SQL

```python
from src.database.postgres_connection import get_db_session

# Execute raw SQL
with get_db_session() as db:
    result = db.execute("""
        SELECT 
            symbol,
            AVG(price) as avg_price,
            MAX(price) as max_price,
            MIN(price) as min_price,
            COUNT(*) as data_points
        FROM market_data
        WHERE time >= NOW() - INTERVAL '24 hours'
        GROUP BY symbol
        ORDER BY avg_price DESC
    """)
    
    for row in result:
        print(f"{row.symbol}: Avg=${row.avg_price}, Max=${row.max_price}, Min=${row.min_price}")
```

---

## Common Database Tables

### Main Tables:
- **users** - User accounts
- **portfolios** - Trading portfolios
- **positions** - Current holdings
- **trades** - Trade history
- **portfolio_snapshots** - Historical portfolio values
- **risk_metrics** - Risk calculations
- **market_data** - Market price data
- **news_articles** - News data
- **reddit_sentiment** - Social sentiment data

### View Table Schemas:
```sql
-- In psql, use:
\d users
\d portfolios
\d trades
\d positions
```

---

## Quick Reference Commands

```bash
# Connect to database
docker exec -it octopus-db psql -U postgres -d trading_db

# List all tables
\dt

# Describe a table
\d table_name

# Exit psql
\q

# Show current database
SELECT current_database();

# Show all databases
\l

# Show all users
\du
```

---

## Example: Complete Query Script

Save this as `query_examples.py`:

```python
#!/usr/bin/env python3
"""Example database queries for Octopus Trading Platform"""

from src.database.postgres_connection import get_db_session
from src.database.models import User, Portfolio, Trade, Position, MarketData
from datetime import datetime, timedelta

def query_users():
    """Query all users"""
    with get_db_session() as db:
        users = db.query(User).all()
        print(f"\n=== Users ({len(users)} total) ===")
        for user in users:
            print(f"  {user.id}: {user.username} ({user.email}) - Active: {user.is_active}")

def query_portfolios():
    """Query all portfolios"""
    with get_db_session() as db:
        portfolios = db.query(Portfolio).join(User).all()
        print(f"\n=== Portfolios ({len(portfolios)} total) ===")
        for p in portfolios:
            print(f"  {p.id}: {p.name} - Owner: {p.user.username} - Value: ${p.total_value}")

def query_recent_trades(days=7):
    """Query recent trades"""
    with get_db_session() as db:
        cutoff = datetime.now() - timedelta(days=days)
        trades = db.query(Trade).filter(
            Trade.trade_date >= cutoff
        ).order_by(Trade.trade_date.desc()).limit(20).all()
        
        print(f"\n=== Recent Trades (last {days} days) ===")
        for trade in trades:
            print(f"  {trade.trade_date}: {trade.trade_type} {trade.quantity} {trade.symbol} @ ${trade.price}")

def query_active_positions():
    """Query active positions"""
    with get_db_session() as db:
        positions = db.query(Position).filter(
            Position.is_active == True
        ).all()
        
        print(f"\n=== Active Positions ({len(positions)} total) ===")
        for pos in positions:
            print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.average_price} avg (P&L: ${pos.unrealized_pnl})")

if __name__ == "__main__":
    query_users()
    query_portfolios()
    query_recent_trades()
    query_active_positions()
```

Run it:
```bash
python3 query_examples.py
```

