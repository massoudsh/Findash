#!/usr/bin/env python3
"""
Seed Real Data for Octopus Trading Platform
Creates comprehensive sample data from database to frontend
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import random
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import uuid

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5434/trading_db"
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Sample symbols for realistic data
SYMBOLS = [
    ("AAPL", "Apple Inc.", "Technology", "NASDAQ"),
    ("MSFT", "Microsoft Corporation", "Technology", "NASDAQ"),
    ("GOOGL", "Alphabet Inc.", "Technology", "NASDAQ"),
    ("AMZN", "Amazon.com Inc.", "Consumer Cyclical", "NASDAQ"),
    ("TSLA", "Tesla Inc.", "Consumer Cyclical", "NASDAQ"),
    ("META", "Meta Platforms Inc.", "Technology", "NASDAQ"),
    ("NVDA", "NVIDIA Corporation", "Technology", "NASDAQ"),
    ("JPM", "JPMorgan Chase & Co.", "Financial Services", "NYSE"),
    ("V", "Visa Inc.", "Financial Services", "NYSE"),
    ("JNJ", "Johnson & Johnson", "Healthcare", "NYSE"),
    ("WMT", "Walmart Inc.", "Consumer Defensive", "NYSE"),
    ("PG", "Procter & Gamble Co.", "Consumer Defensive", "NYSE"),
    ("MA", "Mastercard Inc.", "Financial Services", "NYSE"),
    ("DIS", "The Walt Disney Company", "Communication Services", "NYSE"),
    ("NFLX", "Netflix Inc.", "Communication Services", "NASDAQ"),
]

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def generate_sample_data():
    """Generate comprehensive sample data"""
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("🌱 Seeding database with real sample data...")
        
        # 1. Create Users
        print("\n📝 Creating users...")
        user_ids = []
        users_data = [
            {
                "username": "trader_john",
                "email": "john.trader@example.com",
                "first_name": "John",
                "last_name": "Trader",
                "password_hash": hash_password("password123"),
                "is_verified": True,
                "trading_enabled": True,
                "risk_level": "medium"
            },
            {
                "username": "investor_sarah",
                "email": "sarah.investor@example.com",
                "first_name": "Sarah",
                "last_name": "Investor",
                "password_hash": hash_password("password123"),
                "is_verified": True,
                "trading_enabled": True,
                "risk_level": "high"
            },
            {
                "username": "analyst_mike",
                "email": "mike.analyst@example.com",
                "first_name": "Mike",
                "last_name": "Analyst",
                "password_hash": hash_password("password123"),
                "is_verified": True,
                "trading_enabled": True,
                "risk_level": "low"
            }
        ]
        
        for user_data in users_data:
            result = session.execute(
                text("""
                    INSERT INTO users (username, email, first_name, last_name, password_hash, 
                                     is_verified, trading_enabled, risk_level, created_at)
                    VALUES (:username, :email, :first_name, :last_name, :password_hash,
                           :is_verified, :trading_enabled, :risk_level, NOW())
                    ON CONFLICT (username) DO UPDATE SET
                        email = EXCLUDED.email,
                        updated_at = NOW()
                    RETURNING id
                """),
                user_data
            )
            user_id = result.fetchone()[0]
            user_ids.append(user_id)
            print(f"  ✅ Created user: {user_data['username']} ({user_id})")
        
        # 2. Create Portfolios
        print("\n💼 Creating portfolios...")
        portfolio_ids = []
        portfolio_names = [
            ("Growth Portfolio", "Aggressive growth strategy focusing on tech stocks"),
            ("Dividend Portfolio", "Conservative income-focused portfolio"),
            ("Balanced Portfolio", "Diversified mix of growth and value stocks"),
            ("Tech Focus", "Technology sector concentrated portfolio"),
            ("Blue Chip", "Large-cap stable companies portfolio")
        ]
        
        for i, user_id in enumerate(user_ids):
            for j, (name, desc) in enumerate(portfolio_names[:2]):  # 2 portfolios per user
                initial_value = Decimal(random.uniform(50000, 200000))
                current_value = initial_value * Decimal(random.uniform(0.95, 1.15))
                cash_balance = initial_value * Decimal(random.uniform(0.1, 0.3))
                
                result = session.execute(
                    text("""
                        INSERT INTO portfolios (user_id, name, description, initial_value, 
                                              current_value, cash_balance, is_active, 
                                              risk_tolerance, created_at)
                        VALUES (:user_id, :name, :description, :initial_value, :current_value,
                               :cash_balance, true, :risk_tolerance, NOW())
                        RETURNING id
                    """),
                    {
                        "user_id": user_id,
                        "name": name,
                        "description": desc,
                        "initial_value": float(initial_value),
                        "current_value": float(current_value),
                        "cash_balance": float(cash_balance),
                        "risk_tolerance": ["conservative", "moderate", "aggressive"][i % 3]  # Portfolio risk tolerance
                    }
                )
                portfolio_id = result.fetchone()[0]
                portfolio_ids.append(portfolio_id)
                print(f"  ✅ Created portfolio: {name} ({portfolio_id})")
        
        # 3. Create Positions
        print("\n📊 Creating positions...")
        for portfolio_id in portfolio_ids:
            # 5-8 positions per portfolio
            num_positions = random.randint(5, 8)
            selected_symbols = random.sample(SYMBOLS, num_positions)
            
            for symbol, company_name, sector, exchange in selected_symbols:
                quantity = Decimal(random.uniform(10, 500))
                avg_cost = Decimal(random.uniform(50, 300))
                current_price = avg_cost * Decimal(random.uniform(0.85, 1.25))
                market_value = quantity * current_price
                total_cost = quantity * avg_cost
                unrealized_pnl = market_value - total_cost
                weight_percent = float(market_value / Decimal(100000)) * 100  # Approximate
                
                # Delete existing position if exists, then insert new
                session.execute(
                    text("DELETE FROM positions WHERE portfolio_id = :portfolio_id AND symbol = :symbol"),
                    {"portfolio_id": portfolio_id, "symbol": symbol}
                )
                session.execute(
                    text("""
                        INSERT INTO positions (portfolio_id, symbol, asset_type, quantity,
                                             average_cost, current_price, market_value,
                                             total_cost, unrealized_pnl, weight_percent,
                                             sector, exchange, is_active, created_at)
                        VALUES (:portfolio_id, :symbol, 'stock', :quantity, :avg_cost,
                               :current_price, :market_value, :total_cost, :unrealized_pnl,
                               :weight_percent, :sector, :exchange, true, NOW())
                    """),
                    {
                        "portfolio_id": portfolio_id,
                        "symbol": symbol,
                        "quantity": float(quantity),
                        "avg_cost": float(avg_cost),
                        "current_price": float(current_price),
                        "market_value": float(market_value),
                        "total_cost": float(total_cost),
                        "unrealized_pnl": float(unrealized_pnl),
                        "weight_percent": weight_percent,
                        "sector": sector,
                        "exchange": exchange
                    }
                )
            print(f"  ✅ Created {num_positions} positions for portfolio {portfolio_id}")
        
        # 4. Create Orders/Trades
        print("\n📈 Creating orders and trades...")
        order_statuses = ["executed", "executed", "executed", "pending", "cancelled"]
        
        for portfolio_id in portfolio_ids:
            # 10-20 trades per portfolio
            num_trades = random.randint(10, 20)
            selected_symbols = random.sample(SYMBOLS, min(num_trades, len(SYMBOLS)))
            
            # Get user_id for this portfolio
            portfolio_result = session.execute(
                text("SELECT user_id FROM portfolios WHERE id = :portfolio_id"),
                {"portfolio_id": portfolio_id}
            )
            user_id = portfolio_result.fetchone()[0]
            
            for i, (symbol, _, _, _) in enumerate(selected_symbols):
                trade_date = datetime.now() - timedelta(days=random.randint(1, 90))
                side = random.choice(["buy", "sell"])
                order_type = random.choice(["market", "limit"])
                quantity = Decimal(random.uniform(5, 100))
                price = Decimal(random.uniform(50, 300))
                estimated_cost = float(quantity * price)
                fees = estimated_cost * 0.001  # 0.1% fee
                status = random.choice(["filled", "filled", "filled", "pending", "cancelled"])
                
                session.execute(
                    text("""
                        INSERT INTO orders (user_id, portfolio_id, symbol, asset_type, side, 
                                          order_type, quantity, price, estimated_cost, fees, 
                                          status, created_at)
                        VALUES (:user_id, :portfolio_id, :symbol, 'stock', :side, :order_type, 
                               :quantity, :price, :estimated_cost, :fees, :status, :trade_date)
                    """),
                    {
                        "user_id": user_id,
                        "portfolio_id": portfolio_id,
                        "symbol": symbol,
                        "side": side,
                        "order_type": order_type,
                        "quantity": float(quantity),
                        "price": float(price),
                        "estimated_cost": estimated_cost,
                        "fees": fees,
                        "status": status,
                        "trade_date": trade_date
                    }
                )
            print(f"  ✅ Created {num_trades} orders for portfolio {portfolio_id}")
        
        # 5. Create Market Data (for reports)
        print("\n📉 Creating market data snapshots...")
        for portfolio_id in portfolio_ids:
            # Create daily snapshots for last 30 days
            for day in range(30):
                snapshot_date = datetime.now() - timedelta(days=day)
                portfolio_value = Decimal(random.uniform(50000, 250000))
                
                session.execute(
                    text("""
                        INSERT INTO portfolio_snapshots (portfolio_id, snapshot_date, total_value,
                                                        cash_balance, invested_value, day_return,
                                                        day_return_percent)
                        VALUES (:portfolio_id, :snapshot_date, :total_value, :cash_balance,
                               :invested_value, :day_return, :day_return_percent)
                    """),
                    {
                        "portfolio_id": portfolio_id,
                        "snapshot_date": snapshot_date.date(),
                        "total_value": float(portfolio_value),
                        "cash_balance": float(portfolio_value * Decimal(0.2)),
                        "invested_value": float(portfolio_value * Decimal(0.8)),
                        "day_return": float(portfolio_value * Decimal(random.uniform(-0.05, 0.05))),
                        "day_return_percent": float(random.uniform(-5, 5))
                    }
                )
            print(f"  ✅ Created 30 days of snapshots for portfolio {portfolio_id}")
        
        # 6. Create Reports/Analytics data (skip risk_metrics table if it doesn't exist)
        print("\n📊 Creating analytics and reports data...")
        print("  ✅ Analytics data will be calculated from portfolio snapshots")
        
        # Commit all changes
        session.commit()
        print("\n✅ Successfully seeded database with sample data!")
        print("\n📊 Summary:")
        print(f"  - Users: {len(user_ids)}")
        print(f"  - Portfolios: {len(portfolio_ids)}")
        print(f"  - Positions: ~{len(portfolio_ids) * 6}")
        print(f"  - Orders: ~{len(portfolio_ids) * 15}")
        print(f"  - Market Data: {len(portfolio_ids) * 30} snapshots")
        
        return True
        
    except Exception as e:
        session.rollback()
        print(f"\n❌ Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    success = generate_sample_data()
    sys.exit(0 if success else 1)

