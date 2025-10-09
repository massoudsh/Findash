-- 🗄️ Initial Database Schema - Octopus Trading Platform™
-- This script creates the complete database schema with all required tables and indexes

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- ===========================================
-- 👥 USER MANAGEMENT TABLES
-- ===========================================

-- Users table with authentication and profile data
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    date_of_birth DATE,
    
    -- Profile settings
    avatar_url TEXT,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Account status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    is_admin BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP WITH TIME ZONE,
    phone_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- Security and compliance
    kyc_status VARCHAR(20) DEFAULT 'pending' CHECK (kyc_status IN ('pending', 'approved', 'rejected', 'expired')),
    risk_level VARCHAR(20) DEFAULT 'low' CHECK (risk_level IN ('low', 'medium', 'high')),
    two_factor_enabled BOOLEAN DEFAULT false,
    two_factor_secret VARCHAR(255),
    
    -- Trading permissions
    trading_enabled BOOLEAN DEFAULT false,
    margin_enabled BOOLEAN DEFAULT false,
    options_enabled BOOLEAN DEFAULT false,
    crypto_enabled BOOLEAN DEFAULT true,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- User sessions for security tracking
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET,
    user_agent TEXT,
    device_type VARCHAR(50),
    location JSONB,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User preferences and settings
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'dark',
    notifications_email BOOLEAN DEFAULT true,
    notifications_push BOOLEAN DEFAULT true,
    notifications_sms BOOLEAN DEFAULT false,
    auto_save_portfolios BOOLEAN DEFAULT true,
    default_chart_timeframe VARCHAR(10) DEFAULT '1D',
    default_watchlist_id UUID,
    trading_confirmations BOOLEAN DEFAULT true,
    risk_warnings BOOLEAN DEFAULT true,
    market_data_level VARCHAR(20) DEFAULT 'level1',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 📊 PORTFOLIO MANAGEMENT TABLES
-- ===========================================

-- Portfolios - main portfolio containers
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Portfolio type and strategy
    portfolio_type VARCHAR(20) DEFAULT 'standard' CHECK (portfolio_type IN ('standard', 'paper', 'retirement', 'margin')),
    investment_strategy VARCHAR(50),
    risk_tolerance VARCHAR(20) DEFAULT 'moderate' CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
    
    -- Financial data
    initial_value DECIMAL(15,2) DEFAULT 0.00,
    current_value DECIMAL(15,2) DEFAULT 0.00,
    cash_balance DECIMAL(15,2) DEFAULT 0.00,
    buying_power DECIMAL(15,2) DEFAULT 0.00,
    margin_used DECIMAL(15,2) DEFAULT 0.00,
    
    -- Performance metrics
    total_return DECIMAL(15,2) DEFAULT 0.00,
    total_return_percent DECIMAL(8,4) DEFAULT 0.00,
    day_return DECIMAL(15,2) DEFAULT 0.00,
    day_return_percent DECIMAL(8,4) DEFAULT 0.00,
    
    -- Status and settings
    is_active BOOLEAN DEFAULT true,
    is_public BOOLEAN DEFAULT false,
    auto_rebalance BOOLEAN DEFAULT false,
    benchmark_symbol VARCHAR(10) DEFAULT 'SPY',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Positions - individual holdings within portfolios
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL CHECK (asset_type IN ('stock', 'etf', 'mutual_fund', 'bond', 'crypto', 'forex', 'option', 'future')),
    
    -- Position details
    quantity DECIMAL(18,8) NOT NULL DEFAULT 0,
    average_cost DECIMAL(15,6) NOT NULL DEFAULT 0,
    current_price DECIMAL(15,6) DEFAULT 0,
    market_value DECIMAL(15,2) DEFAULT 0,
    
    -- Cost basis and P&L
    total_cost DECIMAL(15,2) DEFAULT 0,
    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    day_pnl DECIMAL(15,2) DEFAULT 0,
    
    -- Position metrics
    weight_percent DECIMAL(8,4) DEFAULT 0,
    beta DECIMAL(8,4),
    dividend_yield DECIMAL(8,4),
    
    -- Metadata
    exchange VARCHAR(10),
    sector VARCHAR(50),
    industry VARCHAR(100),
    country VARCHAR(3),
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 📈 TRADING TABLES
-- ===========================================

-- Orders - all trading orders
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    
    -- Order identification
    client_order_id VARCHAR(100) UNIQUE,
    broker_order_id VARCHAR(100),
    parent_order_id UUID REFERENCES orders(id),
    
    -- Instrument details
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL CHECK (asset_type IN ('stock', 'etf', 'mutual_fund', 'bond', 'crypto', 'forex', 'option', 'future')),
    
    -- Order parameters
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell', 'short', 'cover')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit', 'trailing_stop')),
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(15,6),
    stop_price DECIMAL(15,6),
    trailing_amount DECIMAL(15,6),
    trailing_percent DECIMAL(8,4),
    
    -- Order status and execution
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'partial', 'filled', 'cancelled', 'rejected', 'expired')),
    filled_quantity DECIMAL(18,8) DEFAULT 0,
    remaining_quantity DECIMAL(18,8),
    average_fill_price DECIMAL(15,6),
    
    -- Financial details
    estimated_cost DECIMAL(15,2),
    actual_cost DECIMAL(15,2),
    commission DECIMAL(15,2) DEFAULT 0,
    fees DECIMAL(15,2) DEFAULT 0,
    
    -- Time specifications
    time_in_force VARCHAR(10) DEFAULT 'DAY' CHECK (time_in_force IN ('DAY', 'GTC', 'IOC', 'FOK')),
    good_till_date TIMESTAMP WITH TIME ZONE,
    
    -- Risk management
    risk_check_passed BOOLEAN DEFAULT true,
    risk_comments TEXT,
    
    -- Metadata
    exchange VARCHAR(10),
    route VARCHAR(20),
    order_source VARCHAR(20) DEFAULT 'web',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE
);

-- Order executions/fills
CREATE TABLE order_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Execution details
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(15,6) NOT NULL,
    value DECIMAL(15,2) NOT NULL,
    commission DECIMAL(15,2) DEFAULT 0,
    fees DECIMAL(15,2) DEFAULT 0,
    
    -- Market data
    exchange VARCHAR(10),
    market_center VARCHAR(20),
    liquidity_flag VARCHAR(10),
    
    -- Timing
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 📊 MARKET DATA TABLES (TimescaleDB)
-- ===========================================

-- Market quotes - real-time price data
CREATE TABLE market_quotes (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Price data
    bid_price DECIMAL(15,6),
    ask_price DECIMAL(15,6),
    bid_size DECIMAL(18,8),
    ask_size DECIMAL(18,8),
    last_price DECIMAL(15,6),
    last_size DECIMAL(18,8),
    
    -- Market status
    exchange VARCHAR(10),
    market_status VARCHAR(20),
    
    -- Additional data
    volume DECIMAL(18,8),
    vwap DECIMAL(15,6),
    high_price DECIMAL(15,6),
    low_price DECIMAL(15,6),
    open_price DECIMAL(15,6),
    close_price DECIMAL(15,6),
    
    -- Metadata
    data_source VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_quotes', 'timestamp', if_not_exists => TRUE);

-- OHLCV candle data
CREATE TABLE market_candles (
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- 1m, 5m, 15m, 1h, 1d, etc.
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- OHLCV data
    open_price DECIMAL(15,6) NOT NULL,
    high_price DECIMAL(15,6) NOT NULL,
    low_price DECIMAL(15,6) NOT NULL,
    close_price DECIMAL(15,6) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    
    -- Additional metrics
    vwap DECIMAL(15,6),
    trades_count INTEGER,
    
    -- Metadata
    data_source VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('market_candles', 'timestamp', if_not_exists => TRUE);

-- News and events
CREATE TABLE market_news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    headline TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    symbols TEXT[], -- Array of affected symbols
    
    -- Source information
    source VARCHAR(100),
    author VARCHAR(100),
    url TEXT,
    
    -- Classification
    category VARCHAR(50),
    sentiment DECIMAL(3,2), -- -1 to 1
    relevance_score DECIMAL(3,2), -- 0 to 1
    
    -- Impact assessment
    market_impact VARCHAR(20) CHECK (market_impact IN ('low', 'medium', 'high')),
    
    -- Timing
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 🤖 ML/AI TABLES
-- ===========================================

-- ML models registry
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- prediction, classification, clustering, etc.
    
    -- Model metadata
    description TEXT,
    features JSONB, -- List of input features
    target_variable VARCHAR(100),
    algorithm VARCHAR(100),
    hyperparameters JSONB,
    
    -- Performance metrics
    accuracy DECIMAL(8,4),
    precision_score DECIMAL(8,4),
    recall_score DECIMAL(8,4),
    f1_score DECIMAL(8,4),
    rmse DECIMAL(15,6),
    mae DECIMAL(15,6),
    
    -- Model lifecycle
    status VARCHAR(20) DEFAULT 'training' CHECK (status IN ('training', 'testing', 'production', 'deprecated')),
    training_start TIMESTAMP WITH TIME ZONE,
    training_end TIMESTAMP WITH TIME ZONE,
    
    -- File locations
    model_path TEXT,
    config_path TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML predictions
CREATE TABLE ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    symbol VARCHAR(20),
    
    -- Prediction details
    prediction_type VARCHAR(50), -- price, direction, volatility, etc.
    predicted_value DECIMAL(15,6),
    confidence_score DECIMAL(8,4),
    prediction_horizon INTEGER, -- minutes into the future
    
    -- Input features
    features JSONB,
    
    -- Validation
    actual_value DECIMAL(15,6),
    prediction_error DECIMAL(15,6),
    
    -- Timing
    prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    target_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('ml_predictions', 'prediction_timestamp', if_not_exists => TRUE);

-- ===========================================
-- 🛡️ RISK MANAGEMENT TABLES
-- ===========================================

-- Risk limits and rules
CREATE TABLE risk_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    
    -- Limit types
    limit_type VARCHAR(50) NOT NULL, -- position_size, sector_concentration, daily_loss, etc.
    limit_scope VARCHAR(20) DEFAULT 'portfolio' CHECK (limit_scope IN ('account', 'portfolio', 'position')),
    
    -- Limit values
    max_value DECIMAL(15,2),
    max_percentage DECIMAL(8,4),
    warning_threshold DECIMAL(8,4) DEFAULT 0.8,
    
    -- Configuration
    symbol VARCHAR(20), -- For symbol-specific limits
    sector VARCHAR(50), -- For sector limits
    asset_type VARCHAR(20), -- For asset type limits
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    enforcement_level VARCHAR(20) DEFAULT 'block' CHECK (enforcement_level IN ('warn', 'block')),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk events and violations
CREATE TABLE risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    order_id UUID REFERENCES orders(id) ON DELETE CASCADE,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL, -- limit_violation, margin_call, position_concentration, etc.
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'acknowledged', 'resolved', 'ignored')),
    
    -- Risk details
    limit_type VARCHAR(50),
    current_value DECIMAL(15,2),
    limit_value DECIMAL(15,2),
    violation_amount DECIMAL(15,2),
    
    -- Description and resolution
    description TEXT,
    resolution_notes TEXT,
    
    -- Timestamps
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 📊 ANALYTICS TABLES
-- ===========================================

-- Portfolio performance history
CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    
    -- Value metrics
    total_value DECIMAL(15,2) NOT NULL,
    cash_balance DECIMAL(15,2) NOT NULL,
    invested_value DECIMAL(15,2) NOT NULL,
    
    -- Performance metrics
    day_return DECIMAL(15,2),
    day_return_percent DECIMAL(8,4),
    total_return DECIMAL(15,2),
    total_return_percent DECIMAL(8,4),
    
    -- Risk metrics
    beta DECIMAL(8,4),
    alpha DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    volatility DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    
    -- Allocation metrics
    equity_allocation DECIMAL(8,4),
    bond_allocation DECIMAL(8,4),
    cash_allocation DECIMAL(8,4),
    alternative_allocation DECIMAL(8,4),
    
    -- Metadata
    benchmark_return DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User activity and audit log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Action details
    action_type VARCHAR(50) NOT NULL, -- login, order, transfer, settings_change, etc.
    entity_type VARCHAR(50), -- user, portfolio, order, etc.
    entity_id UUID,
    
    -- Change tracking
    old_values JSONB,
    new_values JSONB,
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    session_id UUID,
    api_endpoint TEXT,
    
    -- Metadata
    description TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    -- Timing
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('audit_log', 'occurred_at', if_not_exists => TRUE);

-- ===========================================
-- 📧 NOTIFICATIONS TABLES
-- ===========================================

-- Notification templates
CREATE TABLE notification_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    template_type VARCHAR(50) NOT NULL, -- email, push, sms
    
    -- Template content
    subject_template TEXT,
    body_template TEXT NOT NULL,
    
    -- Configuration
    is_active BOOLEAN DEFAULT true,
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    
    -- Metadata
    description TEXT,
    variables JSONB, -- List of template variables
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User notifications
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    template_id UUID REFERENCES notification_templates(id),
    
    -- Notification details
    type VARCHAR(50) NOT NULL, -- order_filled, portfolio_alert, system_maintenance, etc.
    channel VARCHAR(20) NOT NULL CHECK (channel IN ('email', 'push', 'sms', 'in_app')),
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    
    -- Content
    subject TEXT,
    message TEXT NOT NULL,
    data JSONB, -- Additional structured data
    
    -- Delivery tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'delivered', 'failed', 'read')),
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    read_at TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- 🔍 INDEXES FOR PERFORMANCE
-- ===========================================

-- User indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX idx_users_created_at ON users(created_at);

-- Session indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_active ON user_sessions(is_active) WHERE is_active = true;

-- Portfolio indexes
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_portfolios_active ON portfolios(is_active) WHERE is_active = true;

-- Position indexes
CREATE INDEX idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_active ON positions(is_active) WHERE is_active = true;

-- Order indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_portfolio_id ON orders(portfolio_id);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- Market data indexes
CREATE INDEX idx_market_quotes_symbol_timestamp ON market_quotes(symbol, timestamp DESC);
CREATE INDEX idx_market_candles_symbol_timeframe_timestamp ON market_candles(symbol, timeframe, timestamp DESC);

-- ML indexes
CREATE INDEX idx_ml_predictions_symbol_timestamp ON ml_predictions(symbol, prediction_timestamp DESC);
CREATE INDEX idx_ml_models_status ON ml_models(status);

-- Risk indexes
CREATE INDEX idx_risk_events_user_id ON risk_events(user_id);
CREATE INDEX idx_risk_events_status ON risk_events(status);
CREATE INDEX idx_risk_events_occurred_at ON risk_events(occurred_at);

-- Audit indexes
CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_action_type ON audit_log(action_type);
CREATE INDEX idx_audit_log_occurred_at ON audit_log(occurred_at DESC);

-- Notification indexes
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_status ON notifications(status);
CREATE INDEX idx_notifications_created_at ON notifications(created_at DESC);

-- ===========================================
-- 🔧 TRIGGERS AND FUNCTIONS
-- ===========================================

-- Update timestamps trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_risk_limits_updated_at BEFORE UPDATE ON risk_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_notification_templates_updated_at BEFORE UPDATE ON notification_templates FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER update_notifications_updated_at BEFORE UPDATE ON notifications FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Portfolio value calculation function
CREATE OR REPLACE FUNCTION calculate_portfolio_value(portfolio_uuid UUID)
RETURNS DECIMAL(15,2) AS $$
DECLARE
    total_value DECIMAL(15,2) := 0;
    cash_balance DECIMAL(15,2) := 0;
BEGIN
    -- Get cash balance
    SELECT p.cash_balance INTO cash_balance 
    FROM portfolios p 
    WHERE p.id = portfolio_uuid;
    
    -- Calculate total position value
    SELECT COALESCE(SUM(market_value), 0) INTO total_value
    FROM positions 
    WHERE portfolio_id = portfolio_uuid AND is_active = true;
    
    RETURN total_value + COALESCE(cash_balance, 0);
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- 📊 DATA RETENTION POLICIES
-- ===========================================

-- Retention policy for market quotes (keep 30 days of minute data)
SELECT add_retention_policy('market_quotes', INTERVAL '30 days');

-- Retention policy for audit logs (keep 2 years)
SELECT add_retention_policy('audit_log', INTERVAL '2 years');

-- Retention policy for ML predictions (keep 90 days)
SELECT add_retention_policy('ml_predictions', INTERVAL '90 days');

-- ===========================================
-- 🔐 SECURITY POLICIES
-- ===========================================

-- Enable Row Level Security on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Create policies (these would be customized based on your authentication system)
-- Example: Users can only see their own data
CREATE POLICY user_isolation_policy ON users
    FOR ALL TO authenticated_users
    USING (id = current_user_id());

CREATE POLICY portfolio_isolation_policy ON portfolios
    FOR ALL TO authenticated_users
    USING (user_id = current_user_id());

-- ===========================================
-- ✅ SCHEMA INITIALIZATION COMPLETE
-- ===========================================

-- Insert default data (this would be handled by seed scripts)
COMMENT ON DATABASE trading_db IS 'Octopus Trading Platform Database - Created by initial schema migration';

-- Log successful schema creation
INSERT INTO audit_log (action_type, entity_type, description, success) 
VALUES ('schema_migration', 'database', 'Initial schema created successfully', true); 