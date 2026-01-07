#!/usr/bin/env python3
"""
Quick database query script for Octopus Trading Platform
Usage: python quick_query.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.postgres_connection import get_db_session
from src.database.models import User, Portfolio, Trade, Position, MarketData
from datetime import datetime, timedelta

def main():
    """Run example queries"""
    
    print("=" * 60)
    print("Octopus Trading Platform - Database Query Examples")
    print("=" * 60)
    
    # Query 1: Users
    print("\n1. USERS")
    print("-" * 60)
    try:
        with get_db_session() as db:
            users = db.query(User).limit(10).all()
            if users:
                for user in users:
                    print(f"  ID: {user.id} | {user.username} | {user.email} | Active: {user.is_active}")
            else:
                print("  No users found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Query 2: Portfolios
    print("\n2. PORTFOLIOS")
    print("-" * 60)
    try:
        with get_db_session() as db:
            portfolios = db.query(Portfolio).join(User).limit(10).all()
            if portfolios:
                for p in portfolios:
                    print(f"  ID: {p.id} | {p.name} | Owner: {p.user.username} | Value: ${p.total_value}")
            else:
                print("  No portfolios found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Query 3: Recent Trades
    print("\n3. RECENT TRADES (Last 10)")
    print("-" * 60)
    try:
        with get_db_session() as db:
            trades = db.query(Trade).order_by(Trade.trade_date.desc()).limit(10).all()
            if trades:
                for trade in trades:
                    print(f"  {trade.trade_date} | {trade.trade_type} | {trade.quantity} {trade.symbol} @ ${trade.price} | Status: {trade.status}")
            else:
                print("  No trades found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Query 4: Active Positions
    print("\n4. ACTIVE POSITIONS")
    print("-" * 60)
    try:
        with get_db_session() as db:
            positions = db.query(Position).filter(Position.is_active == True).limit(10).all()
            if positions:
                for pos in positions:
                    print(f"  {pos.symbol} | {pos.quantity} shares | Avg: ${pos.average_price} | Current: ${pos.current_price} | P&L: ${pos.unrealized_pnl}")
            else:
                print("  No active positions found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Query 5: Table Counts
    print("\n5. TABLE COUNTS")
    print("-" * 60)
    try:
        with get_db_session() as db:
            user_count = db.query(User).count()
            portfolio_count = db.query(Portfolio).count()
            trade_count = db.query(Trade).count()
            position_count = db.query(Position).count()
            market_data_count = db.query(MarketData).count() if hasattr(MarketData, '__tablename__') else 0
            
            print(f"  Users: {user_count}")
            print(f"  Portfolios: {portfolio_count}")
            print(f"  Trades: {trade_count}")
            print(f"  Positions: {position_count}")
            print(f"  Market Data: {market_data_count}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Query complete!")
    print("=" * 60)
    print("\nFor more query options, see QUERY_DATABASE.md")

if __name__ == "__main__":
    main()

