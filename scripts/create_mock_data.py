#!/usr/bin/env python3
"""
Create Mock Data for Octopus Trading Platform Database
Creates tables and inserts realistic mock data for testing queries
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/trading_db"
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Sample symbols
SYMBOLS = [
    ("AAPL", "Apple Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com Inc."),
    ("TSLA", "Tesla Inc."),
    ("META", "Meta Platforms Inc."),
    ("NVDA", "NVIDIA Corporation"),
    ("BTC-USD", "Bitcoin"),
    ("ETH-USD", "Ethereum"),
    ("JPM", "JPMorgan Chase & Co."),
    ("V", "Visa Inc."),
    ("JNJ", "Johnson & Johnson"),
]

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_tables(engine):
    """Create all database tables"""
    print("📋 Creating database tables...")
    
    with engine.connect() as conn:
        # Create users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(64) UNIQUE NOT NULL,
                email VARCHAR(128) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                phone VARCHAR(20),
                is_verified BOOLEAN DEFAULT FALSE,
                risk_tolerance VARCHAR(20) DEFAULT 'moderate',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create portfolios table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(128) NOT NULL,
                description TEXT,
                initial_cash NUMERIC(15, 2) DEFAULT 10000.00,
                current_cash NUMERIC(15, 2) DEFAULT 10000.00,
                total_value NUMERIC(15, 2) DEFAULT 0.00,
                is_active BOOLEAN DEFAULT TRUE,
                risk_level VARCHAR(20) DEFAULT 'moderate',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create positions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
                symbol VARCHAR(32) NOT NULL,
                quantity NUMERIC(15, 6) NOT NULL,
                average_price NUMERIC(15, 6) NOT NULL,
                current_price NUMERIC(15, 6) DEFAULT 0.000000,
                market_value NUMERIC(15, 2) DEFAULT 0.00,
                unrealized_pnl NUMERIC(15, 2) DEFAULT 0.00,
                position_type VARCHAR(10) DEFAULT 'long',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(portfolio_id, symbol)
            )
        """))
        
        # Create trades table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
                symbol VARCHAR(32) NOT NULL,
                trade_type VARCHAR(4) NOT NULL,
                quantity NUMERIC(15, 6) NOT NULL,
                price NUMERIC(15, 6) NOT NULL,
                total_amount NUMERIC(15, 2) NOT NULL,
                fees NUMERIC(10, 2) DEFAULT 0.00,
                trade_date TIMESTAMP DEFAULT NOW(),
                settlement_date TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                order_id VARCHAR(100),
                notes TEXT
            )
        """))
        
        # Create market data table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS financial_time_series (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                price FLOAT,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                volume INTEGER,
                exchange VARCHAR(32),
                UNIQUE(time, symbol)
            )
        """))
        
        # Create portfolio snapshots table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
                snapshot_date TIMESTAMP DEFAULT NOW(),
                total_value NUMERIC(15, 2) NOT NULL,
                cash_value NUMERIC(15, 2) DEFAULT 0.00,
                positions_value NUMERIC(15, 2) DEFAULT 0.00,
                daily_pnl NUMERIC(15, 2) DEFAULT 0.00,
                daily_pnl_percent NUMERIC(8, 4) DEFAULT 0.0000,
                UNIQUE(portfolio_id, snapshot_date)
            )
        """))
        
        # Create risk metrics table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id SERIAL PRIMARY KEY,
                portfolio_id INTEGER UNIQUE REFERENCES portfolios(id) ON DELETE CASCADE,
                value_at_risk_1d NUMERIC(15, 2),
                value_at_risk_1w NUMERIC(15, 2),
                value_at_risk_1m NUMERIC(15, 2),
                sharpe_ratio NUMERIC(8, 4),
                beta NUMERIC(8, 4),
                volatility NUMERIC(8, 4),
                max_drawdown NUMERIC(8, 4),
                concentration_risk VARCHAR(20) DEFAULT 'medium',
                last_updated TIMESTAMP DEFAULT NOW()
            )
        """))
        
        conn.commit()
        print("  ✅ Tables created successfully")

def insert_mock_data(session):
    """Insert comprehensive mock data"""
    print("\n🌱 Inserting mock data...")
    
    # 1. Create Users
    print("\n👥 Creating users...")
    users_data = [
        ("trader_john", "john@example.com", "password123", "moderate"),
        ("investor_sarah", "sarah@example.com", "password123", "aggressive"),
        ("analyst_mike", "mike@example.com", "password123", "conservative"),
        ("day_trader", "trader@example.com", "password123", "aggressive"),
        ("long_term_investor", "investor@example.com", "password123", "conservative"),
    ]
    
    user_ids = []
    for username, email, password, risk_tolerance in users_data:
        result = session.execute(
            text("""
                INSERT INTO users (username, email, password_hash, risk_tolerance, is_verified, is_active)
                VALUES (:username, :email, :password_hash, :risk_tolerance, TRUE, TRUE)
                ON CONFLICT (username) DO UPDATE SET email = EXCLUDED.email
                RETURNING id
            """),
            {
                "username": username,
                "email": email,
                "password_hash": hash_password(password),
                "risk_tolerance": risk_tolerance
            }
        )
        user_id = result.fetchone()[0]
        user_ids.append(user_id)
        print(f"  ✅ User: {username} (ID: {user_id})")
    
    # 2. Create Portfolios
    print("\n💼 Creating portfolios...")
    portfolio_ids = []
    portfolio_names = [
        ("Growth Portfolio", "Aggressive growth strategy"),
        ("Dividend Portfolio", "Income-focused portfolio"),
        ("Balanced Portfolio", "Diversified mix"),
        ("Tech Focus", "Technology sector"),
        ("Blue Chip", "Large-cap stable companies"),
        ("Crypto Portfolio", "Cryptocurrency holdings"),
    ]
    
    for i, user_id in enumerate(user_ids):
        for j, (name, desc) in enumerate(portfolio_names[:2]):  # 2 portfolios per user
            initial_cash = Decimal(random.uniform(10000, 100000))
            current_cash = initial_cash * Decimal(random.uniform(0.2, 0.4))
            total_value = initial_cash * Decimal(random.uniform(0.95, 1.15))
            
            result = session.execute(
                text("""
                    INSERT INTO portfolios (user_id, name, description, initial_cash, 
                                          current_cash, total_value, is_active, risk_level)
                    VALUES (:user_id, :name, :description, :initial_cash, :current_cash,
                           :total_value, TRUE, :risk_level)
                    RETURNING id
                """),
                {
                    "user_id": user_id,
                    "name": name,
                    "description": desc,
                    "initial_cash": float(initial_cash),
                    "current_cash": float(current_cash),
                    "total_value": float(total_value),
                    "risk_level": ["low", "moderate", "high"][i % 3]
                }
            )
            portfolio_id = result.fetchone()[0]
            portfolio_ids.append(portfolio_id)
            print(f"  ✅ Portfolio: {name} (ID: {portfolio_id})")
    
    # 3. Create Positions
    print("\n📊 Creating positions...")
    for portfolio_id in portfolio_ids:
        num_positions = random.randint(3, 6)
        selected_symbols = random.sample(SYMBOLS, num_positions)
        
        for symbol, _ in selected_symbols:
            quantity = Decimal(random.uniform(10, 500))
            avg_price = Decimal(random.uniform(50, 300))
            current_price = avg_price * Decimal(random.uniform(0.85, 1.25))
            market_value = float(quantity * current_price)
            total_cost = float(quantity * avg_price)
            unrealized_pnl = market_value - total_cost
            
            session.execute(
                text("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, average_price,
                                          current_price, market_value, unrealized_pnl, is_active)
                    VALUES (:portfolio_id, :symbol, :quantity, :avg_price, :current_price,
                           :market_value, :unrealized_pnl, TRUE)
                    ON CONFLICT (portfolio_id, symbol) DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        current_price = EXCLUDED.current_price,
                        market_value = EXCLUDED.market_value,
                        unrealized_pnl = EXCLUDED.unrealized_pnl
                """),
                {
                    "portfolio_id": portfolio_id,
                    "symbol": symbol,
                    "quantity": float(quantity),
                    "avg_price": float(avg_price),
                    "current_price": float(current_price),
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl
                }
            )
        print(f"  ✅ Created {num_positions} positions for portfolio {portfolio_id}")
    
    # 4. Create Trades
    print("\n📈 Creating trades...")
    for portfolio_id in portfolio_ids:
        num_trades = random.randint(10, 25)
        
        for i in range(num_trades):
            symbol, _ = random.choice(SYMBOLS)
            trade_date = datetime.now() - timedelta(days=random.randint(1, 90))
            trade_type = random.choice(["BUY", "SELL"])
            quantity = Decimal(random.uniform(5, 100))
            price = Decimal(random.uniform(50, 300))
            total_amount = float(quantity * price)
            fees = total_amount * 0.001
            status = random.choice(["executed", "executed", "executed", "pending", "cancelled"])
            
            session.execute(
                text("""
                    INSERT INTO trades (portfolio_id, symbol, trade_type, quantity, price,
                                       total_amount, fees, trade_date, status, order_id)
                    VALUES (:portfolio_id, :symbol, :trade_type, :quantity, :price,
                           :total_amount, :fees, :trade_date, :status, :order_id)
                """),
                {
                    "portfolio_id": portfolio_id,
                    "symbol": symbol,
                    "trade_type": trade_type,
                    "quantity": float(quantity),
                    "price": float(price),
                    "total_amount": total_amount,
                    "fees": fees,
                    "trade_date": trade_date,
                    "status": status,
                    "order_id": f"ORD-{portfolio_id}-{i+1:04d}"
                }
            )
        print(f"  ✅ Created {num_trades} trades for portfolio {portfolio_id}")
    
    # 5. Create Market Data
    print("\n📉 Creating market data...")
    for symbol, _ in SYMBOLS:
        base_price = random.uniform(50, 300)
        for day in range(30):  # 30 days of data
            time = datetime.now() - timedelta(days=day)
            price = base_price * random.uniform(0.95, 1.05)
            open_price = price * random.uniform(0.98, 1.02)
            high = max(open_price, price) * random.uniform(1.0, 1.03)
            low = min(open_price, price) * random.uniform(0.97, 1.0)
            volume = random.randint(1000000, 10000000)
            
            session.execute(
                text("""
                    INSERT INTO financial_time_series (time, symbol, price, open, high, low, volume, exchange)
                    VALUES (:time, :symbol, :price, :open, :high, :low, :volume, :exchange)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        price = EXCLUDED.price,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        volume = EXCLUDED.volume
                """),
                {
                    "time": time,
                    "symbol": symbol,
                    "price": price,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "volume": volume,
                    "exchange": "NASDAQ" if "-" not in symbol else "CRYPTO"
                }
            )
        print(f"  ✅ Created 30 days of market data for {symbol}")
    
    # 6. Create Portfolio Snapshots
    print("\n📸 Creating portfolio snapshots...")
    for portfolio_id in portfolio_ids:
        for day in range(30):  # 30 days of snapshots
            snapshot_date = datetime.now() - timedelta(days=day)
            total_value = Decimal(random.uniform(10000, 150000))
            cash_value = total_value * Decimal(random.uniform(0.2, 0.4))
            positions_value = total_value - cash_value
            daily_pnl = total_value * Decimal(random.uniform(-0.05, 0.05))
            daily_pnl_percent = float(daily_pnl / total_value * 100) if total_value > 0 else 0
            
            session.execute(
                text("""
                    INSERT INTO portfolio_snapshots (portfolio_id, snapshot_date, total_value,
                                                    cash_value, positions_value, daily_pnl, daily_pnl_percent)
                    VALUES (:portfolio_id, :snapshot_date, :total_value, :cash_value,
                           :positions_value, :daily_pnl, :daily_pnl_percent)
                    ON CONFLICT (portfolio_id, snapshot_date) DO UPDATE SET
                        total_value = EXCLUDED.total_value,
                        cash_value = EXCLUDED.cash_value,
                        positions_value = EXCLUDED.positions_value,
                        daily_pnl = EXCLUDED.daily_pnl,
                        daily_pnl_percent = EXCLUDED.daily_pnl_percent
                """),
                {
                    "portfolio_id": portfolio_id,
                    "snapshot_date": snapshot_date,
                    "total_value": float(total_value),
                    "cash_value": float(cash_value),
                    "positions_value": float(positions_value),
                    "daily_pnl": float(daily_pnl),
                    "daily_pnl_percent": daily_pnl_percent
                }
            )
        print(f"  ✅ Created 30 days of snapshots for portfolio {portfolio_id}")
    
    # 7. Create Risk Metrics
    print("\n🛡️ Creating risk metrics...")
    for portfolio_id in portfolio_ids:
        session.execute(
            text("""
                INSERT INTO risk_metrics (portfolio_id, value_at_risk_1d, value_at_risk_1w,
                                         value_at_risk_1m, sharpe_ratio, beta, volatility,
                                         max_drawdown, concentration_risk)
                VALUES (:portfolio_id, :var_1d, :var_1w, :var_1m, :sharpe, :beta,
                       :volatility, :max_dd, :concentration)
                ON CONFLICT (portfolio_id) DO UPDATE SET
                    value_at_risk_1d = EXCLUDED.value_at_risk_1d,
                    value_at_risk_1w = EXCLUDED.value_at_risk_1w,
                    value_at_risk_1m = EXCLUDED.value_at_risk_1m,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    beta = EXCLUDED.beta,
                    volatility = EXCLUDED.volatility,
                    max_drawdown = EXCLUDED.max_drawdown
            """),
            {
                "portfolio_id": portfolio_id,
                "var_1d": random.uniform(100, 1000),
                "var_1w": random.uniform(500, 5000),
                "var_1m": random.uniform(2000, 20000),
                "sharpe": random.uniform(0.5, 2.5),
                "beta": random.uniform(0.8, 1.5),
                "volatility": random.uniform(0.1, 0.3),
                "max_dd": random.uniform(-0.2, -0.05),
                "concentration": random.choice(["low", "medium", "high"])
            }
        )
        print(f"  ✅ Created risk metrics for portfolio {portfolio_id}")
    
    session.commit()
    print("\n✅ Mock data insertion complete!")

def main():
    """Main function"""
    print("=" * 60)
    print("🐙 Octopus Trading Platform - Mock Data Generator")
    print("=" * 60)
    
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create tables
        create_tables(engine)
        
        # Insert mock data
        insert_mock_data(session)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 DATA SUMMARY")
        print("=" * 60)
        
        result = session.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM portfolios"))
        portfolio_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM positions"))
        position_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM trades"))
        trade_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM financial_time_series"))
        market_data_count = result.fetchone()[0]
        
        result = session.execute(text("SELECT COUNT(*) FROM portfolio_snapshots"))
        snapshot_count = result.fetchone()[0]
        
        print(f"  👥 Users: {user_count}")
        print(f"  💼 Portfolios: {portfolio_count}")
        print(f"  📊 Positions: {position_count}")
        print(f"  📈 Trades: {trade_count}")
        print(f"  📉 Market Data: {market_data_count}")
        print(f"  📸 Snapshots: {snapshot_count}")
        
        print("\n✅ Mock data created successfully!")
        print("\n💡 You can now query the database in psql:")
        print("   docker exec -it octopus-db psql -U postgres -d trading_db")
        print("\n📝 Try these queries:")
        print("   SELECT * FROM users;")
        print("   SELECT * FROM portfolios;")
        print("   SELECT * FROM trades LIMIT 10;")
        print("   SELECT * FROM positions;")
        
        session.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

