-- =====================================================================
-- Octopus Trading Platform™ - Database Initialization Script
-- Secure PostgreSQL + TimescaleDB setup for production
-- =====================================================================

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create UUID extension for secure ID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create crypto extension for security functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =====================================================================
-- CORE TABLES
-- =====================================================================

-- Users table with secure authentication
CREATE TABLE IF NOT EXISTS users (
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
    locked_until TIMESTAMPTZ,
    
    -- Security constraints
    CONSTRAINT email_format CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT password_not_empty CHECK (length(password_hash) > 0)
);

-- API Keys table for service authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL DEFAULT 'API Key',
    is_active BOOLEAN DEFAULT true,
    permissions TEXT[] DEFAULT ARRAY[],
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Security constraints
    CONSTRAINT key_hash_format CHECK (length(key_hash) = 64)
);

-- Portfolios table
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    total_value DECIMAL(20, 2) DEFAULT 0.00,
    cash_balance DECIMAL(20, 2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT positive_values CHECK (total_value >= 0 AND cash_balance >= 0)
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    avg_cost DECIMAL(20, 2) NOT NULL,
    current_price DECIMAL(20, 2),
    market_value DECIMAL(20, 2),
    unrealized_pnl DECIMAL(20, 2),
    position_type VARCHAR(20) DEFAULT 'long' CHECK (position_type IN ('long', 'short')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Unique constraint on portfolio + symbol
    UNIQUE(portfolio_id, symbol)
);

-- Orders table for trade execution
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20, 2),
    stop_price DECIMAL(20, 2),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    avg_fill_price DECIMAL(20, 2),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected')),
    external_order_id VARCHAR(100),
    broker VARCHAR(50),
    commission DECIMAL(10, 2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT price_constraints CHECK (
        (order_type = 'market') OR 
        (order_type IN ('limit', 'stop_limit') AND price > 0) OR
        (order_type = 'stop' AND stop_price > 0)
    )
);

-- =====================================================================
-- TIME-SERIES TABLES (TimescaleDB)
-- =====================================================================

-- Market data with TimescaleDB hypertable
CREATE TABLE IF NOT EXISTS market_data (
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

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Portfolio snapshots for historical tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20, 2) NOT NULL,
    cash_balance DECIMAL(20, 2) NOT NULL,
    positions_value DECIMAL(20, 2) NOT NULL,
    daily_pnl DECIMAL(20, 2),
    total_return DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('buy', 'sell', 'hold')),
    strength DECIMAL(3, 2) CHECK (strength BETWEEN 0 AND 1),
    source VARCHAR(100) NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('trading_signals', 'timestamp', if_not_exists => TRUE);

-- =====================================================================
-- AUDIT AND SECURITY TABLES
-- =====================================================================

-- Audit log for security and compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for efficient querying
SELECT create_hypertable('audit_log', 'timestamp', if_not_exists => TRUE);

-- Security events table
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ip_address INET,
    details JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('security_events', 'timestamp', if_not_exists => TRUE);

-- =====================================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================================

-- Users indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at);

-- API Keys indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);

-- Portfolios indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_active ON portfolios(is_active);

-- Positions indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Orders indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_portfolio_id ON orders(portfolio_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_created_at ON orders(created_at);

-- Market data indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);

-- Audit log indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);

-- Security events indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp DESC);

-- =====================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================
-- DATA RETENTION POLICIES
-- =====================================================================

-- Drop old market data (keep 2 years)
SELECT add_retention_policy('market_data', INTERVAL '2 years', if_not_exists => true);

-- Drop old audit logs (keep 1 year)
SELECT add_retention_policy('audit_log', INTERVAL '1 year', if_not_exists => true);

-- Drop old security events (keep 1 year)
SELECT add_retention_policy('security_events', INTERVAL '1 year', if_not_exists => true);

-- Keep portfolio snapshots for 5 years
SELECT add_retention_policy('portfolio_snapshots', INTERVAL '5 years', if_not_exists => true);

-- =====================================================================
-- SECURITY SETTINGS
-- =====================================================================

-- Create restricted application user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'octopus_app') THEN
        CREATE ROLE octopus_app WITH LOGIN PASSWORD 'CHANGE_ME_IN_PRODUCTION';
    END IF;
END
$$;

-- Grant appropriate permissions
GRANT CONNECT ON DATABASE trading_db TO octopus_app;
GRANT USAGE ON SCHEMA public TO octopus_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO octopus_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO octopus_app;

-- Create read-only user for analytics
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'octopus_readonly') THEN
        CREATE ROLE octopus_readonly WITH LOGIN PASSWORD 'CHANGE_ME_IN_PRODUCTION';
    END IF;
END
$$;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE trading_db TO octopus_readonly;
GRANT USAGE ON SCHEMA public TO octopus_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO octopus_readonly;

-- =====================================================================
-- INITIAL DATA
-- =====================================================================

-- Insert demo user (remove in production)
INSERT INTO users (email, password_hash, first_name, last_name, roles, permissions)
VALUES (
    'demo@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3JrF6o0Hze', -- password: 'demo123'
    'Demo',
    'User',
    ARRAY['user', 'trader'],
    ARRAY['read', 'write', 'trade']
) ON CONFLICT (email) DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ Octopus Trading Platform database initialized successfully!';
END
$$; 