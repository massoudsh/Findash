"""
Database initialization script for Quantum Trading Matrix
"""

import asyncio
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, User, Portfolio, RiskLevel
from passlib.context import CryptContext
from datetime import datetime

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_database_engine(database_url: str = None):
    """Create database engine"""
    if not database_url:
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/trading_db")
    
    engine = create_engine(database_url, echo=True)
    return engine

def init_database(database_url: str = None):
    """Initialize database with tables and seed data"""
    engine = create_database_engine(database_url)
    
    # Create all tables
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        # Create demo user if not exists
        existing_user = session.query(User).filter(User.email == "demo@quantumtrading.com").first()
        if not existing_user:
            print("Creating demo user...")
            demo_user = User(
                email="demo@quantumtrading.com",
                username="demo_user",
                hashed_password=hash_password("demo123"),
                full_name="Demo User",
                is_active=True,
                is_verified=True,
                risk_tolerance=RiskLevel.MEDIUM
            )
            session.add(demo_user)
            session.commit()
            
            # Create demo portfolio
            print("Creating demo portfolio...")
            demo_portfolio = Portfolio(
                user_id=demo_user.id,
                name="Demo Options Portfolio",
                description="Sample portfolio for testing options trading features",
                initial_capital=100000.0,
                current_value=100000.0,
                cash_balance=50000.0,
                is_active=True
            )
            session.add(demo_portfolio)
            session.commit()
            
            print(f"Demo user created with ID: {demo_user.id}")
            print(f"Demo portfolio created with ID: {demo_portfolio.id}")
        else:
            print("Demo user already exists")
        
        # Create admin user if not exists
        existing_admin = session.query(User).filter(User.email == "admin@quantumtrading.com").first()
        if not existing_admin:
            print("Creating admin user...")
            admin_user = User(
                email="admin@quantumtrading.com",
                username="admin",
                hashed_password=hash_password("admin123"),
                full_name="Admin User",
                is_active=True,
                is_verified=True,
                risk_tolerance=RiskLevel.HIGH
            )
            session.add(admin_user)
            session.commit()
            print(f"Admin user created with ID: {admin_user.id}")
        else:
            print("Admin user already exists")
            
    except Exception as e:
        print(f"Error during database initialization: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    print("Database initialization completed successfully!")

def reset_database(database_url: str = None):
    """Reset database - DROP ALL TABLES and recreate"""
    engine = create_database_engine(database_url)
    
    print("WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Operation cancelled")
        return
    
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Tables dropped!")
    
    # Recreate
    init_database(database_url)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_database()
    else:
        init_database() 