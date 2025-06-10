"""
PostgreSQL database initialization script
Using direct PostgreSQL connection instead of SQLAlchemy
"""

import os
import sys
from datetime import datetime

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Handle both direct execution and module imports
try:
    # Try relative imports first (when run as module)
    from .postgres_connection import get_db, DatabaseConfig
    from .repositories import UserRepository, PortfolioRepository
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database.postgres_connection import get_db, DatabaseConfig
    from database.repositories import UserRepository, PortfolioRepository

def init_database(reset: bool = False):
    """Initialize the database with tables and seed data"""
    
    print("🚀 Initializing Quantum Trading Matrix™ Database...")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    # Get database instance
    db = get_db()
    
    # Test connection
    try:
        db.execute_query("SELECT 1", fetch='one')
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    # Drop tables if reset is requested
    if reset:
        print("⚠️  Resetting database (dropping all tables)...")
        drop_tables(db)
    
    # Create tables
    print("📋 Creating database tables...")
    try:
        db.create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False
    
    # Create seed data
    print("🌱 Creating seed data...")
    try:
        create_seed_data()
        print("✅ Seed data created successfully")
    except Exception as e:
        print(f"❌ Failed to create seed data: {e}")
        return False
    
    print("🎉 Database initialization completed successfully!")
    return True

def drop_tables(db):
    """Drop all tables (for reset)"""
    drop_sql = """
    -- Drop tables in reverse order due to foreign key constraints
    DROP TABLE IF EXISTS audit_logs CASCADE;
    DROP TABLE IF EXISTS api_keys CASCADE;
    DROP TABLE IF EXISTS trading_signals CASCADE;
    DROP TABLE IF EXISTS correlation_matrices CASCADE;
    DROP TABLE IF EXISTS market_data CASCADE;
    DROP TABLE IF EXISTS risk_reports CASCADE;
    DROP TABLE IF EXISTS portfolio_metrics CASCADE;
    DROP TABLE IF EXISTS option_positions CASCADE;
    DROP TABLE IF EXISTS portfolios CASCADE;
    DROP TABLE IF EXISTS users CASCADE;
    
    -- Drop functions and triggers
    DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
    """
    
    db.execute_query(drop_sql)
    print("✅ All tables dropped")

def create_seed_data():
    """Create initial seed data"""
    user_repo = UserRepository()
    portfolio_repo = PortfolioRepository()
    
    # Create demo user
    demo_user = user_repo.get_user_by_email("demo@quantumtrading.com")
    if not demo_user:
        print("👤 Creating demo user...")
        demo_user = user_repo.create_user(
            email="demo@quantumtrading.com",
            username="demo_user",
            password="demo123",
            full_name="Demo User",
            risk_tolerance="medium"
        )
        
        # Verify the user
        user_repo.update_user(demo_user.id, is_verified=True)
        
        print(f"✅ Demo user created: {demo_user.email} (ID: {demo_user.id})")
    else:
        print(f"ℹ️  Demo user already exists: {demo_user.email}")
    
    # Create admin user
    admin_user = user_repo.get_user_by_email("admin@quantumtrading.com")
    if not admin_user:
        print("👨‍💼 Creating admin user...")
        admin_user = user_repo.create_user(
            email="admin@quantumtrading.com",
            username="admin",
            password="admin123",
            full_name="Admin User",
            risk_tolerance="high"
        )
        
        # Verify the user
        user_repo.update_user(admin_user.id, is_verified=True)
        
        print(f"✅ Admin user created: {admin_user.email} (ID: {admin_user.id})")
    else:
        print(f"ℹ️  Admin user already exists: {admin_user.email}")
    
    # Create demo portfolio for demo user
    demo_portfolios = portfolio_repo.get_user_portfolios(demo_user.id)
    if not demo_portfolios:
        print("💼 Creating demo portfolio...")
        demo_portfolio = portfolio_repo.create_portfolio(
            user_id=demo_user.id,
            name="Demo Options Portfolio",
            description="Sample portfolio for testing options trading features",
            initial_capital=100000.0
        )
        print(f"✅ Demo portfolio created: {demo_portfolio.name} (ID: {demo_portfolio.id})")
    else:
        print(f"ℹ️  Demo portfolio already exists: {demo_portfolios[0].name}")
    
    # Create admin portfolio
    admin_portfolios = portfolio_repo.get_user_portfolios(admin_user.id)
    if not admin_portfolios:
        print("💼 Creating admin portfolio...")
        admin_portfolio = portfolio_repo.create_portfolio(
            user_id=admin_user.id,
            name="Admin Testing Portfolio",
            description="Portfolio for administrative testing and monitoring",
            initial_capital=1000000.0
        )
        print(f"✅ Admin portfolio created: {admin_portfolio.name} (ID: {admin_portfolio.id})")
    else:
        print(f"ℹ️  Admin portfolio already exists: {admin_portfolios[0].name}")

def verify_database_setup():
    """Verify that the database is properly set up"""
    print("🔍 Verifying database setup...")
    
    db = get_db()
    user_repo = UserRepository()
    portfolio_repo = PortfolioRepository()
    
    # Check table existence
    tables_query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
    """
    
    tables = db.execute_query(tables_query, fetch='all')
    expected_tables = [
        'api_keys', 'audit_logs', 'correlation_matrices', 'market_data',
        'option_positions', 'portfolio_metrics', 'portfolios', 
        'risk_reports', 'trading_signals', 'users'
    ]
    
    found_tables = [table['table_name'] for table in tables]
    missing_tables = set(expected_tables) - set(found_tables)
    
    if missing_tables:
        print(f"❌ Missing tables: {missing_tables}")
        return False
    else:
        print(f"✅ All {len(expected_tables)} tables found")
    
    # Check users
    demo_user = user_repo.get_user_by_email("demo@quantumtrading.com")
    admin_user = user_repo.get_user_by_email("admin@quantumtrading.com")
    
    if not demo_user:
        print("❌ Demo user not found")
        return False
    
    if not admin_user:
        print("❌ Admin user not found")
        return False
    
    print("✅ Seed users verified")
    
    # Check portfolios
    demo_portfolios = portfolio_repo.get_user_portfolios(demo_user.id)
    admin_portfolios = portfolio_repo.get_user_portfolios(admin_user.id)
    
    if not demo_portfolios:
        print("❌ Demo portfolio not found")
        return False
    
    if not admin_portfolios:
        print("❌ Admin portfolio not found")
        return False
    
    print("✅ Seed portfolios verified")
    
    # Check indexes
    indexes_query = """
    SELECT indexname 
    FROM pg_indexes 
    WHERE schemaname = 'public' 
    AND indexname LIKE 'idx_%'
    ORDER BY indexname
    """
    
    indexes = db.execute_query(indexes_query, fetch='all')
    if len(indexes) >= 10:  # We created multiple indexes
        print(f"✅ Database indexes verified ({len(indexes)} indexes)")
    else:
        print(f"⚠️  Expected more indexes, found {len(indexes)}")
    
    print("✅ Database setup verification completed successfully!")
    return True

def get_database_info():
    """Get database information and statistics"""
    print("📊 Database Information:")
    
    db = get_db()
    
    # Database version
    version_result = db.execute_query("SELECT version()", fetch='one')
    print(f"PostgreSQL Version: {version_result['version'].split(',')[0]}")
    
    # Database size
    size_query = """
    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
    """
    size_result = db.execute_query(size_query, fetch='one')
    print(f"Database Size: {size_result['db_size']}")
    
    # Table statistics
    stats_query = """
    SELECT 
        schemaname,
        relname as tablename,
        n_tup_ins as inserts,
        n_tup_upd as updates,
        n_tup_del as deletes,
        n_live_tup as live_tuples
    FROM pg_stat_user_tables 
    ORDER BY relname
    """
    
    try:
        stats = db.execute_query(stats_query, fetch='all')
        
        print("\nTable Statistics:")
        print("-" * 60)
        print(f"{'Table':<20} {'Inserts':<8} {'Updates':<8} {'Deletes':<8} {'Live Rows':<10}")
        print("-" * 60)
        
        for stat in stats:
            print(f"{stat['tablename']:<20} {stat['inserts']:<8} {stat['updates']:<8} {stat['deletes']:<8} {stat['live_tuples']:<10}")
    except Exception as e:
        print(f"⚠️  Could not retrieve table statistics: {e}")

def main():
    """Main function for database initialization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize Quantum Trading Matrix Database')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset database (drop all tables and recreate)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify database setup')
    parser.add_argument('--info', action='store_true',
                       help='Show database information')
    
    args = parser.parse_args()
    
    if args.reset:
        confirm = input("⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("❌ Operation cancelled")
            sys.exit(1)
    
    try:
        if args.info:
            get_database_info()
        elif args.verify:
            success = verify_database_setup()
            sys.exit(0 if success else 1)
        else:
            success = init_database(reset=args.reset)
            if success and not args.reset:
                verify_database_setup()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 