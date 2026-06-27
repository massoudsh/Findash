-- ─────────────────────────────────────────────────────────────────────────────
-- Migration: Add Iranian Assets tables
-- Date: 2026-06-27
-- Run AFTER alembic baseline
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Enum type
CREATE TYPE asset_category AS ENUM (
    'gold', 'silver', 'currency', 'real_estate', 'crypto'
);

-- 2. Master assets table
CREATE TABLE IF NOT EXISTS assets (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(20)     UNIQUE NOT NULL,
    name_fa     VARCHAR(100)    NOT NULL,
    name_en     VARCHAR(100)    NOT NULL,
    category    asset_category  NOT NULL,
    unit        VARCHAR(20)     DEFAULT 'تومان',
    source_key  VARCHAR(100),
    is_active   BOOLEAN         DEFAULT TRUE,
    created_at  TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_assets_category ON assets (category);
CREATE INDEX IF NOT EXISTS ix_assets_symbol   ON assets (symbol);

-- 3. Price snapshots (latest price per asset)
CREATE TABLE IF NOT EXISTS asset_price_snapshots (
    id                  SERIAL PRIMARY KEY,
    asset_id            INTEGER REFERENCES assets(id),
    symbol              VARCHAR(20) UNIQUE NOT NULL,
    price               NUMERIC(20, 2) NOT NULL,
    price_toman         NUMERIC(20, 2) NOT NULL,
    change_24h          NUMERIC(20, 2) DEFAULT 0,
    change_percent_24h  NUMERIC(8, 4)  DEFAULT 0,
    high_24h            NUMERIC(20, 2),
    low_24h             NUMERIC(20, 2),
    source              VARCHAR(50),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_snapshot_updated ON asset_price_snapshots (updated_at);

-- 4. Price history (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS asset_price_history (
    id        SERIAL,
    asset_id  INTEGER REFERENCES assets(id),
    symbol    VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open      NUMERIC(20, 2),
    high      NUMERIC(20, 2),
    low       NUMERIC(20, 2),
    close     NUMERIC(20, 2) NOT NULL,
    volume    NUMERIC(20, 2) DEFAULT 0,
    interval  VARCHAR(5) DEFAULT '1d',
    PRIMARY KEY (id, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'asset_price_history',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_history_symbol_ts
    ON asset_price_history (symbol, timestamp DESC);

-- 5. User portfolio assets
CREATE TABLE IF NOT EXISTS portfolio_assets (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    quantity    NUMERIC(20, 8) NOT NULL,
    buy_price   NUMERIC(20, 2) NOT NULL,
    buy_date    TIMESTAMPTZ NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_portfolio_user_symbol
    ON portfolio_assets (user_id, symbol);

-- ─────────────────────────────────────────────────────────────────────────────
-- Seed: insert master asset list
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO assets (symbol, name_fa, name_en, category, source_key) VALUES
-- طلا
('XAU18',        'طلای ۱۸ عیار (هر گرم)',  'Gold 18K per gram',      'gold',        'geram18'),
('XAU24',        'طلای ۲۴ عیار (هر گرم)',  'Gold 24K per gram',      'gold',        'geram24'),
('COIN_FULL',    'سکه بهار آزادی',          'Bahar Azadi Coin',       'gold',        'sekeb'),
('COIN_HALF',    'نیم‌سکه',                 'Half Coin',              'gold',        'nim'),
('COIN_QUARTER', 'ربع‌سکه',                 'Quarter Coin',           'gold',        'rob'),
('COIN_OLD',     'سکه قدیم',               'Old Coin',               'gold',        'sekeq'),
('MESGHAL',      'مثقال طلا',               'Gold Mithqal',           'gold',        'mesghal'),
-- نقره
('XAG',          'نقره (هر گرم)',            'Silver per gram',        'silver',      'silver'),
-- ارز
('USD',          'دلار آمریکا',             'US Dollar',              'currency',    'price_dollar_rl'),
('EUR',          'یورو',                    'Euro',                   'currency',    'price_eur'),
('AED',          'درهم امارات',             'UAE Dirham',             'currency',    'price_aed'),
('GBP',          'پوند انگلیس',             'British Pound',          'currency',    'price_gbp'),
-- مسکن
('RE_TEHRAN',    'شاخص مسکن تهران',         'Tehran Real Estate Index','real_estate','real_estate_tehran'),
-- کریپتو
('BTC',          'بیت‌کوین',                'Bitcoin',                'crypto',      'crypto_bitcoin'),
('ETH',          'اتریوم',                  'Ethereum',               'crypto',      'crypto_ethereum'),
('USDT',         'تتر',                     'Tether',                 'crypto',      'crypto_tether')
ON CONFLICT (symbol) DO NOTHING;
