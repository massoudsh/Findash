"""
Direct PostgreSQL database connection and operations
Using psycopg2 without SQLAlchemy ORM
"""

import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration class"""
    
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5433"))
        self.database = os.getenv("DB_NAME", "trading_db")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "postgres")
        self.min_connections = int(os.getenv("DB_MIN_CONNECTIONS", "1"))
        self.max_connections = int(os.getenv("DB_MAX_CONNECTIONS", "20"))
    
    @property
    def connection_string(self) -> str:
        """Get the PostgreSQL connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for psycopg2"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password,
            "sslmode": "prefer",
            "connect_timeout": 10
        }

class PostgreSQLDatabase:
    """Direct PostgreSQL database operations"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **self.config.connection_params
            )
            logger.info("Database connection pool initialized successfully")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, connection=None, cursor_factory=None):
        """Get a cursor with optional connection and factory"""
        if connection:
            cursor = connection.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=cursor_factory)
                try:
                    yield cursor
                finally:
                    cursor.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: str = None) -> Optional[List[Dict]]:
        """
        Execute a query and optionally fetch results
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: 'one', 'all', or None for no fetch
        
        Returns:
            Query results if fetch is specified
        """
        with self.get_connection() as conn:
            with self.get_cursor(conn, cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                try:
                    cursor.execute(query, params)
                    
                    if fetch == 'one':
                        result = cursor.fetchone()
                        return dict(result) if result else None
                    elif fetch == 'all':
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                    else:
                        conn.commit()
                        return None
                        
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Query execution failed: {e}")
                    logger.error(f"Query: {query}")
                    logger.error(f"Params: {params}")
                    raise
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute query with multiple parameter sets"""
        with self.get_connection() as conn:
            with self.get_cursor(conn) as cursor:
                try:
                    cursor.executemany(query, params_list)
                    conn.commit()
                    return cursor.rowcount
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Batch execution failed: {e}")
                    raise
    
    def create_tables(self):
        """Create all necessary tables"""
        tables_sql = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(100) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            full_name VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            is_verified BOOLEAN DEFAULT FALSE,
            risk_tolerance VARCHAR(20) DEFAULT 'medium',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Portfolios table
        CREATE TABLE IF NOT EXISTS portfolios (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            initial_capital DECIMAL(15,2) DEFAULT 100000.00,
            current_value DECIMAL(15,2) DEFAULT 0.00,
            cash_balance DECIMAL(15,2) DEFAULT 100000.00,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Option positions table
        CREATE TABLE IF NOT EXISTS option_positions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            symbol VARCHAR(20) NOT NULL,
            option_type VARCHAR(10) NOT NULL CHECK (option_type IN ('call', 'put')),
            strike_price DECIMAL(10,2) NOT NULL,
            expiry_date TIMESTAMP WITH TIME ZONE NOT NULL,
            quantity INTEGER NOT NULL,
            premium_paid DECIMAL(10,4) NOT NULL,
            current_price DECIMAL(10,4),
            underlying_price DECIMAL(10,2) NOT NULL,
            implied_volatility DECIMAL(5,4) DEFAULT 0.2000,
            risk_free_rate DECIMAL(5,4) DEFAULT 0.0500,
            delta DECIMAL(8,6),
            gamma DECIMAL(8,6),
            theta DECIMAL(8,6),
            vega DECIMAL(8,6),
            rho DECIMAL(8,6),
            order_status VARCHAR(20) DEFAULT 'pending',
            opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP WITH TIME ZONE,
            pnl DECIMAL(12,2) DEFAULT 0.00,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Portfolio metrics table
        CREATE TABLE IF NOT EXISTS portfolio_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
            total_delta DECIMAL(10,4) DEFAULT 0.0000,
            total_gamma DECIMAL(10,4) DEFAULT 0.0000,
            total_theta DECIMAL(10,4) DEFAULT 0.0000,
            total_vega DECIMAL(10,4) DEFAULT 0.0000,
            total_rho DECIMAL(10,4) DEFAULT 0.0000,
            var_95 DECIMAL(12,2),
            var_99 DECIMAL(12,2),
            expected_shortfall DECIMAL(12,2),
            max_drawdown DECIMAL(8,4),
            sharpe_ratio DECIMAL(8,4),
            beta DECIMAL(8,4),
            concentration_index DECIMAL(8,4),
            largest_position_weight DECIMAL(8,4),
            total_return DECIMAL(8,4),
            daily_return DECIMAL(8,4),
            volatility DECIMAL(8,4),
            calculation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Risk reports table
        CREATE TABLE IF NOT EXISTS risk_reports (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
            report_type VARCHAR(50) NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            summary TEXT,
            recommendations TEXT,
            alerts JSONB,
            scenario_analysis JSONB,
            overall_risk_score DECIMAL(5,2),
            liquidity_risk_score DECIMAL(5,2),
            concentration_risk_score DECIMAL(5,2),
            market_risk_score DECIMAL(5,2),
            generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Market data table
        CREATE TABLE IF NOT EXISTS market_data (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            open_price DECIMAL(10,2),
            high_price DECIMAL(10,2),
            low_price DECIMAL(10,2),
            close_price DECIMAL(10,2),
            volume BIGINT,
            implied_volatility DECIMAL(5,4),
            beta DECIMAL(8,4),
            market_cap BIGINT,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Correlation matrices table
        CREATE TABLE IF NOT EXISTS correlation_matrices (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbols JSONB NOT NULL,
            correlation_data JSONB NOT NULL,
            period VARCHAR(10) DEFAULT '1y',
            calculation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Trading signals table
        CREATE TABLE IF NOT EXISTS trading_signals (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            signal_type VARCHAR(10) NOT NULL,
            strategy_name VARCHAR(100) NOT NULL,
            confidence DECIMAL(3,2),
            entry_price DECIMAL(10,2),
            target_price DECIMAL(10,2),
            stop_loss DECIMAL(10,2),
            reasoning TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE
        );

        -- API keys table
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key_name VARCHAR(100) NOT NULL,
            key_hash VARCHAR(255) NOT NULL,
            can_read BOOLEAN DEFAULT TRUE,
            can_write BOOLEAN DEFAULT FALSE,
            can_trade BOOLEAN DEFAULT FALSE,
            last_used TIMESTAMP WITH TIME ZONE,
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER DEFAULT 1000,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE
        );

        -- Audit logs table
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50),
            resource_id UUID,
            old_values JSONB,
            new_values JSONB,
            ip_address INET,
            user_agent TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
        CREATE INDEX IF NOT EXISTS idx_option_positions_user_id ON option_positions(user_id);
        CREATE INDEX IF NOT EXISTS idx_option_positions_portfolio_id ON option_positions(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_option_positions_symbol ON option_positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_option_positions_expiry ON option_positions(expiry_date);
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
        CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);

        -- Create function to update updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        -- Create triggers for updated_at
        DROP TRIGGER IF EXISTS update_users_updated_at ON users;
        CREATE TRIGGER update_users_updated_at 
            BEFORE UPDATE ON users 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

        DROP TRIGGER IF EXISTS update_portfolios_updated_at ON portfolios;
        CREATE TRIGGER update_portfolios_updated_at 
            BEFORE UPDATE ON portfolios 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

        DROP TRIGGER IF EXISTS update_option_positions_updated_at ON option_positions;
        CREATE TRIGGER update_option_positions_updated_at 
            BEFORE UPDATE ON option_positions 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        
        self.execute_query(tables_sql)
        logger.info("Database tables created successfully")
    
    def close_pool(self):
        """Close the connection pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")

# Global database instance
db_config = DatabaseConfig()
db = PostgreSQLDatabase(db_config)

def get_db() -> PostgreSQLDatabase:
    """Get the global database instance"""
    return db

def close_db():
    """Close the global database instance"""
    db.close_pool() 