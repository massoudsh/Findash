# 🗄️ Database Schema Documentation - Octopus Trading Platform™

## Overview

The Octopus Trading Platform uses **PostgreSQL 15** with **TimescaleDB** extension for time-series data optimization. The database is designed for high-performance financial data processing with ACID compliance and advanced security features.

## Database Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   TimescaleDB   │    │   Extensions    │
│   Tables        │ -> │   Hypertables   │ -> │   Security      │
│   (OLTP)        │    │   (Time-series) │    │   (Audit/Logs)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Tables

### 👥 Users & Authentication

#### `users`
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    roles TEXT[] DEFAULT ARRAY['user'],
    permissions TEXT[] DEFAULT ARRAY[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ
);
```

#### `api_keys`
```sql
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL DEFAULT 'API Key',
    is_active BOOLEAN DEFAULT true,
    permissions TEXT[] DEFAULT ARRAY[],
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 💼 Portfolio Management

#### `portfolios`
```sql
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    total_value DECIMAL(20, 2) DEFAULT 0.00,
    cash_balance DECIMAL(20, 2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `positions`
```sql
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    avg_cost DECIMAL(20, 2) NOT NULL,
    current_price DECIMAL(20, 2),
    market_value DECIMAL(20, 2),
    unrealized_pnl DECIMAL(20, 2),
    position_type VARCHAR(20) DEFAULT 'long',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(portfolio_id, symbol)
);
```

#### `orders`
```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20, 2),
    stop_price DECIMAL(20, 2),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    avg_fill_price DECIMAL(20, 2),
    status VARCHAR(20) DEFAULT 'pending',
    external_order_id VARCHAR(100),
    broker VARCHAR(50),
    commission DECIMAL(10, 2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ
);
```

## Time-Series Tables (TimescaleDB Hypertables)

### 📊 Market Data

#### `market_data` (Hypertable)
```sql
CREATE TABLE market_data (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(20, 2),
    high_price DECIMAL(20, 2),
    low_price DECIMAL(20, 2),
    close_price DECIMAL(20, 2),
    volume BIGINT,
    vwap DECIMAL(20, 2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp');
```

#### `portfolio_snapshots` (Hypertable)
```sql
CREATE TABLE portfolio_snapshots (
    portfolio_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20, 2) NOT NULL,
    cash_balance DECIMAL(20, 2) NOT NULL,
    day_change DECIMAL(20, 2),
    day_change_percent DECIMAL(8, 4),
    positions_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('portfolio_snapshots', 'timestamp');
```

#### `trade_executions` (Hypertable)
```sql
CREATE TABLE trade_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES orders(id),
    portfolio_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 2) NOT NULL,
    commission DECIMAL(10, 2) DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    broker VARCHAR(50),
    execution_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('trade_executions', 'timestamp');
```

## Security & Audit Tables

#### `audit_log` (Hypertable)
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id VARCHAR(100)
);

-- Convert to hypertable
SELECT create_hypertable('audit_log', 'timestamp');
```

#### `security_events` (Hypertable)
```sql
CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id UUID REFERENCES users(id),
    ip_address INET,
    details JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by UUID REFERENCES users(id)
);

-- Convert to hypertable
SELECT create_hypertable('security_events', 'timestamp');
```

## Analytics & ML Tables

#### `predictions`
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    predicted_value DECIMAL(20, 6),
    confidence_score DECIMAL(5, 4),
    prediction_horizon VARCHAR(20),
    features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    target_date TIMESTAMPTZ
);
```

#### `model_performance`
```sql
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20),
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10, 6),
    evaluation_date DATE NOT NULL,
    data_period VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Indexes for Performance

### Primary Indexes
```sql
-- Users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;

-- Portfolios
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_portfolios_active ON portfolios(is_active) WHERE is_active = true;

-- Positions
CREATE INDEX idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Orders
CREATE INDEX idx_orders_portfolio_id ON orders(portfolio_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
```

### TimescaleDB Specific Indexes
```sql
-- Market data optimizations
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);

-- Portfolio snapshots
CREATE INDEX idx_portfolio_snapshots_portfolio_time ON portfolio_snapshots(portfolio_id, timestamp DESC);

-- Trade executions
CREATE INDEX idx_trade_executions_portfolio_time ON trade_executions(portfolio_id, timestamp DESC);
CREATE INDEX idx_trade_executions_symbol_time ON trade_executions(symbol, timestamp DESC);
```

## Data Retention Policies

```sql
-- Keep market data for 7 years
SELECT add_retention_policy('market_data', INTERVAL '7 years');

-- Keep portfolio snapshots for 5 years
SELECT add_retention_policy('portfolio_snapshots', INTERVAL '5 years');

-- Keep trade executions permanently (regulatory requirement)
-- No retention policy for trade_executions

-- Keep audit logs for 3 years
SELECT add_retention_policy('audit_log', INTERVAL '3 years');

-- Keep security events for 2 years
SELECT add_retention_policy('security_events', INTERVAL '2 years');
```

## Database Roles & Security

### Application Roles
```sql
-- Main application user
CREATE ROLE octopus_app WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE trading_db TO octopus_app;
GRANT USAGE ON SCHEMA public TO octopus_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO octopus_app;

-- Read-only analytics user
CREATE ROLE octopus_readonly WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE trading_db TO octopus_readonly;
GRANT USAGE ON SCHEMA public TO octopus_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO octopus_readonly;

-- Backup user
CREATE ROLE octopus_backup WITH LOGIN PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE trading_db TO octopus_backup;
ALTER ROLE octopus_backup WITH REPLICATION;
```

### Row Level Security (RLS)
```sql
-- Enable RLS on sensitive tables
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Policies to ensure users only see their own data
CREATE POLICY portfolio_isolation ON portfolios
    FOR ALL TO octopus_app
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY position_isolation ON positions
    FOR ALL TO octopus_app
    USING (portfolio_id IN (
        SELECT id FROM portfolios 
        WHERE user_id = current_setting('app.current_user_id')::UUID
    ));
```

## Backup Strategy

### Daily Backups
```bash
# Automated daily backup script
pg_dump -h localhost -U octopus_backup trading_db > backup_$(date +%Y%m%d).sql

# Compress and encrypt
gzip backup_$(date +%Y%m%d).sql
gpg --encrypt --recipient backup@octopus.trading backup_$(date +%Y%m%d).sql.gz
```

### Point-in-Time Recovery
```sql
-- Enable WAL archiving for PITR
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
max_wal_senders = 3
wal_level = replica
```

## Migration Management

### Alembic Configuration
```python
# alembic/env.py
from src.database.models import Base
target_metadata = Base.metadata

# Migration commands
alembic revision --autogenerate -m "Description"
alembic upgrade head
alembic downgrade -1
```

### Migration Best Practices
1. **Always backup** before running migrations
2. **Test migrations** in staging environment first
3. **Use transactions** for schema changes
4. **Document breaking changes** in migration messages
5. **Monitor performance** after schema changes

## Performance Monitoring

### Key Metrics to Monitor
```sql
-- Query performance
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;

-- Table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
```

## Troubleshooting

### Common Issues

1. **Slow Queries**
   - Check `pg_stat_statements` for long-running queries
   - Analyze query plans with `EXPLAIN ANALYZE`
   - Consider adding indexes

2. **Lock Contention**
   - Monitor `pg_locks` table
   - Check for long-running transactions
   - Consider connection pooling

3. **Storage Issues**
   - Monitor disk space
   - Check TimescaleDB compression
   - Review retention policies

### Performance Tuning
```sql
-- Recommended PostgreSQL settings for trading workload
shared_buffers = '4GB'
effective_cache_size = '12GB'
maintenance_work_mem = '1GB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
random_page_cost = 1.1
```

---

## Account platform tables (SQLAlchemy, 2026-09)

The UUID DDL earlier in this file is a historical sketch. The live account-platform models use **integer user FKs** and are defined in `src/database/models.py` plus `PaymentOrder` in `src/api/endpoints/payment_zarinpal.py`. Migration: `src/alembic/versions/20260919_admin_risk_subscription_schema.py` (`admin_risk_sub_001`).

| Table | Purpose | Key columns / notes |
|-------|---------|---------------------|
| `payment_orders` | ZarinPal orders | `authority` unique; `purpose` `general` \| `wallet_topup` \| `subscription`; `purpose_ref` = plan code; `status` pending/paid/failed/expired |
| `wallet_balances` | IRT ledger | Unique `(user_id, currency)`; `balance`, `available`, `locked`, `pending` |
| `wallet_transactions` | Wallet history | `type` deposit/withdrawal/…; `status` pending/completed/… |
| `bank_accounts` | Sheba payout targets | `sheba_number` `IR` + 24 digits; `verified` defaults false |
| `subscription_plans` | Priced plans | `code` unique; `price_toman`; `duration_days` |
| `user_subscriptions` | Entitlement | `status` active/expired/cancelled; `end_at` required |
| `kyc_profiles` | One profile per user | `status` pending_review/verified/rejected; national code stored, masked on read |
| `price_alert_rules` | One-shot alerts | `direction` above/below; `channels` JSON; deactivated on fire |
| `push_subscriptions` | Web Push | Unique `endpoint` |
| `risk_policies` | One policy per user | Defaults 5% daily DD, 30% concentration, `action_on_breach=alert` |
| `risk_policy_breaches` | Breach history | `rule` + value/threshold/`action_taken` |
| `audit_logs` | Admin / KYC actions | `actor_user_id`, `action`, JSON `detail` |
| `notifications` | In-app inbox | `category` subscription/price_alert/kyc/admin/risk/system |

Workflows that write these tables: [ACCOUNT_PLATFORM.md](ACCOUNT_PLATFORM.md).

---

*Database schema version: 2.2*  
*Last updated: September 2026* 