-- 👥 Default Users Seed Data - Octopus Trading Platform™
-- This script creates initial users for development and testing

-- ===========================================
-- 🔐 ADMIN USERS
-- ===========================================

-- System Administrator
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    is_active,
    is_verified,
    is_admin,
    email_verified_at,
    kyc_status,
    trading_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'admin',
    'admin@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: admin123
    'System',
    'Administrator',
    true,
    true,
    true,
    NOW(),
    'approved',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Platform Admin
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    is_active,
    is_verified,
    is_admin,
    email_verified_at,
    kyc_status,
    trading_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    'platform_admin',
    'platform@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: admin123
    'Platform',
    'Admin',
    true,
    true,
    true,
    NOW(),
    'approved',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- ===========================================
-- 🧪 DEMO/TEST USERS
-- ===========================================

-- Demo Trader
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    phone,
    is_active,
    is_verified,
    email_verified_at,
    kyc_status,
    risk_level,
    trading_enabled,
    margin_enabled,
    crypto_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000003',
    'demo_trader',
    'demo@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: demo123
    'Demo',
    'Trader',
    '+1-555-DEMO-001',
    true,
    true,
    NOW(),
    'approved',
    'medium',
    true,
    false,
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Test Investor
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    phone,
    is_active,
    is_verified,
    email_verified_at,
    kyc_status,
    risk_level,
    trading_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000004',
    'test_investor',
    'investor@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: investor123
    'Test',
    'Investor',
    '+1-555-TEST-002',
    true,
    true,
    NOW(),
    'approved',
    'low',
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Portfolio Manager
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    phone,
    is_active,
    is_verified,
    email_verified_at,
    kyc_status,
    risk_level,
    trading_enabled,
    margin_enabled,
    options_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000005',
    'portfolio_manager',
    'manager@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: manager123
    'Portfolio',
    'Manager',
    '+1-555-MGMT-003',
    true,
    true,
    NOW(),
    'approved',
    'high',
    true,
    true,
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Day Trader
INSERT INTO users (
    id,
    username,
    email,
    password_hash,
    first_name,
    last_name,
    phone,
    is_active,
    is_verified,
    email_verified_at,
    kyc_status,
    risk_level,
    trading_enabled,
    margin_enabled,
    crypto_enabled,
    created_at
) VALUES (
    '00000000-0000-0000-0000-000000000006',
    'day_trader',
    'daytrader@octopus.trading',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVgqrx.v6vF9R.Ni', -- password: trader123
    'Day',
    'Trader',
    '+1-555-TRADE-004',
    true,
    true,
    NOW(),
    'approved',
    'high',
    true,
    true,
    true,
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- ===========================================
-- 👥 USER PREFERENCES
-- ===========================================

-- Set preferences for demo trader
INSERT INTO user_preferences (
    user_id,
    theme,
    notifications_email,
    notifications_push,
    auto_save_portfolios,
    default_chart_timeframe,
    trading_confirmations,
    risk_warnings,
    preferences
) VALUES (
    '00000000-0000-0000-0000-000000000003',
    'dark',
    true,
    true,
    true,
    '1D',
    true,
    true,
    '{"dashboard_layout": "advanced", "chart_indicators": ["SMA", "RSI", "MACD"], "watchlist_symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD"]}'
) ON CONFLICT (user_id) DO NOTHING;

-- Set preferences for test investor
INSERT INTO user_preferences (
    user_id,
    theme,
    notifications_email,
    notifications_push,
    auto_save_portfolios,
    default_chart_timeframe,
    trading_confirmations,
    risk_warnings,
    preferences
) VALUES (
    '00000000-0000-0000-0000-000000000004',
    'light',
    true,
    false,
    true,
    '1W',
    true,
    true,
    '{"dashboard_layout": "simple", "chart_indicators": ["SMA"], "watchlist_symbols": ["VOO", "BND", "VTI"]}'
) ON CONFLICT (user_id) DO NOTHING;

-- Set preferences for portfolio manager
INSERT INTO user_preferences (
    user_id,
    theme,
    notifications_email,
    notifications_push,
    auto_save_portfolios,
    default_chart_timeframe,
    trading_confirmations,
    risk_warnings,
    preferences
) VALUES (
    '00000000-0000-0000-0000-000000000005',
    'dark',
    true,
    true,
    true,
    '1D',
    false,
    true,
    '{"dashboard_layout": "professional", "chart_indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger"], "watchlist_symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT"]}'
) ON CONFLICT (user_id) DO NOTHING;

-- ===========================================
-- 📊 DEFAULT PORTFOLIOS
-- ===========================================

-- Demo Portfolio for demo trader
INSERT INTO portfolios (
    id,
    user_id,
    name,
    description,
    currency,
    portfolio_type,
    investment_strategy,
    risk_tolerance,
    initial_value,
    current_value,
    cash_balance,
    buying_power,
    is_active,
    benchmark_symbol
) VALUES (
    '10000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000003',
    'Demo Portfolio',
    'Demo trading portfolio with virtual funds',
    'USD',
    'paper',
    'Growth',
    'moderate',
    100000.00,
    100000.00,
    50000.00,
    100000.00,
    true,
    'SPY'
) ON CONFLICT (id) DO NOTHING;

-- Conservative Portfolio for test investor
INSERT INTO portfolios (
    id,
    user_id,
    name,
    description,
    currency,
    portfolio_type,
    investment_strategy,
    risk_tolerance,
    initial_value,
    current_value,
    cash_balance,
    buying_power,
    is_active,
    benchmark_symbol
) VALUES (
    '10000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000004',
    'Conservative Growth',
    'Long-term conservative investment portfolio',
    'USD',
    'standard',
    'Conservative Growth',
    'conservative',
    50000.00,
    50000.00,
    25000.00,
    50000.00,
    true,
    'VTI'
) ON CONFLICT (id) DO NOTHING;

-- Aggressive Portfolio for portfolio manager
INSERT INTO portfolios (
    id,
    user_id,
    name,
    description,
    currency,
    portfolio_type,
    investment_strategy,
    risk_tolerance,
    initial_value,
    current_value,
    cash_balance,
    buying_power,
    is_active,
    benchmark_symbol
) VALUES (
    '10000000-0000-0000-0000-000000000003',
    '00000000-0000-0000-0000-000000000005',
    'Aggressive Growth',
    'High-risk, high-reward portfolio for experienced traders',
    'USD',
    'margin',
    'Aggressive Growth',
    'aggressive',
    250000.00,
    250000.00,
    100000.00,
    500000.00,
    true,
    'QQQ'
) ON CONFLICT (id) DO NOTHING;

-- Day Trading Portfolio
INSERT INTO portfolios (
    id,
    user_id,
    name,
    description,
    currency,
    portfolio_type,
    investment_strategy,
    risk_tolerance,
    initial_value,
    current_value,
    cash_balance,
    buying_power,
    is_active,
    benchmark_symbol
) VALUES (
    '10000000-0000-0000-0000-000000000004',
    '00000000-0000-0000-0000-000000000006',
    'Day Trading',
    'Active day trading portfolio with high turnover',
    'USD',
    'margin',
    'Day Trading',
    'aggressive',
    100000.00,
    100000.00,
    75000.00,
    400000.00,
    true,
    'SPY'
) ON CONFLICT (id) DO NOTHING;

-- ===========================================
-- 🛡️ DEFAULT RISK LIMITS
-- ===========================================

-- Position size limits for demo trader
INSERT INTO risk_limits (
    user_id,
    portfolio_id,
    limit_type,
    limit_scope,
    max_percentage,
    warning_threshold,
    is_active,
    enforcement_level
) VALUES 
    ('00000000-0000-0000-0000-000000000003', '10000000-0000-0000-0000-000000000001', 'position_size', 'position', 10.0, 0.8, true, 'warn'),
    ('00000000-0000-0000-0000-000000000003', '10000000-0000-0000-0000-000000000001', 'daily_loss', 'portfolio', 5.0, 0.8, true, 'block'),
    ('00000000-0000-0000-0000-000000000003', '10000000-0000-0000-0000-000000000001', 'sector_concentration', 'portfolio', 25.0, 0.8, true, 'warn');

-- Conservative limits for test investor
INSERT INTO risk_limits (
    user_id,
    portfolio_id,
    limit_type,
    limit_scope,
    max_percentage,
    warning_threshold,
    is_active,
    enforcement_level
) VALUES 
    ('00000000-0000-0000-0000-000000000004', '10000000-0000-0000-0000-000000000002', 'position_size', 'position', 5.0, 0.8, true, 'block'),
    ('00000000-0000-0000-0000-000000000004', '10000000-0000-0000-0000-000000000002', 'daily_loss', 'portfolio', 2.0, 0.8, true, 'block'),
    ('00000000-0000-0000-0000-000000000004', '10000000-0000-0000-0000-000000000002', 'sector_concentration', 'portfolio', 15.0, 0.8, true, 'block');

-- ===========================================
-- 📧 DEFAULT NOTIFICATION TEMPLATES
-- ===========================================

-- Order filled notification
INSERT INTO notification_templates (
    name,
    template_type,
    subject_template,
    body_template,
    priority,
    description,
    variables
) VALUES (
    'order_filled',
    'email',
    'Order Filled: {{symbol}} {{quantity}} shares',
    'Your {{side}} order for {{quantity}} shares of {{symbol}} has been filled at ${{price}} per share. Total value: ${{total_value}}.',
    'normal',
    'Notification sent when an order is completely filled',
    '["symbol", "quantity", "side", "price", "total_value"]'
) ON CONFLICT (name) DO NOTHING;

-- Risk limit violation
INSERT INTO notification_templates (
    name,
    template_type,
    subject_template,
    body_template,
    priority,
    description,
    variables
) VALUES (
    'risk_violation',
    'email',
    'Risk Limit Violation: {{limit_type}}',
    'A {{limit_type}} risk limit has been violated in your {{portfolio_name}} portfolio. Current value: {{current_value}}, Limit: {{limit_value}}.',
    'high',
    'Notification sent when a risk limit is violated',
    '["limit_type", "portfolio_name", "current_value", "limit_value"]'
) ON CONFLICT (name) DO NOTHING;

-- ===========================================
-- ✅ SEED DATA COMPLETE
-- ===========================================

-- Log successful seed data insertion
INSERT INTO audit_log (action_type, entity_type, description, success) 
VALUES ('seed_data', 'database', 'Default users and portfolios created successfully', true); 