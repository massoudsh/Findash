#!/usr/bin/env python3
"""
Final comprehensive test for Quantum Trading Matrix™
Tests all components and provides clear status
"""

import os
import sys

def test_critical_imports():
    """Test all critical imports"""
    print("🧪 Testing Critical Imports")
    print("-" * 30)
    
    critical_modules = [
        ("psycopg2", "PostgreSQL driver"),
        ("fastapi", "FastAPI framework"), 
        ("pandas", "Data analysis"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Visualization"),
        ("scipy", "Scientific computing"),
        ("passlib", "Password hashing")
    ]
    
    failed = []
    for module, desc in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed.append(module)
    
    return len(failed) == 0

def test_database_modules():
    """Test database module imports"""
    print("\n🗄️ Testing Database Modules")
    print("-" * 30)
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from database.postgres_connection import get_db
        print("✅ Database connection module")
        
        from database.repositories import UserRepository
        print("✅ Database repositories")
        
        return True
    except Exception as e:
        print(f"❌ Database modules failed: {e}")
        return False

def test_options_module():
    """Test options trading module"""
    print("\n📈 Testing Options Module")
    print("-" * 30)
    
    try:
        from options_risk_integration import BlackScholesCalculator
        print("✅ Options trading module imported")
        
        # Test calculation
        price = BlackScholesCalculator.option_price(100, 100, 0.25, 0.05, 0.2, 'call')
        print(f"✅ Black-Scholes calculation: ${price:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Options module failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🔌 Testing Database Connection")
    print("-" * 30)
    
    try:
        from database.postgres_connection import get_db
        
        db = get_db()
        result = db.execute_query("SELECT 1 as test", fetch='one')
        
        if result and result['test'] == 1:
            print("✅ PostgreSQL connection working")
            return True
        else:
            print("❌ PostgreSQL connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure PostgreSQL is running: docker compose up -d db")
        return False

def test_fastapi_app():
    """Test FastAPI application"""
    print("\n🌐 Testing FastAPI Application")
    print("-" * 30)
    
    try:
        from main import app
        print("✅ FastAPI app imported")
        
        # Check some key routes
        routes = [route.path for route in app.routes]
        key_routes = ["/health", "/", "/options/price"]
        
        missing = []
        for route in key_routes:
            if route in routes:
                print(f"✅ Route {route}")
            else:
                print(f"❌ Route {route} missing")
                missing.append(route)
        
        return len(missing) == 0
    except Exception as e:
        print(f"❌ FastAPI app failed: {e}")
        return False

def test_database_initialization():
    """Test database initialization"""
    print("\n🏗️ Testing Database Initialization")
    print("-" * 30)
    
    try:
        # Import the init module
        from database import postgres_init
        
        # Test if we can get database info
        db = postgres_init.get_db()
        version = db.execute_query("SELECT version()", fetch='one')
        print(f"✅ Database version: {version['version'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def main():
    """Run comprehensive test suite"""
    print("🚀 Quantum Trading Matrix™ - Final System Test")
    print("=" * 60)
    
    tests = [
        ("Critical Imports", test_critical_imports),
        ("Database Modules", test_database_modules), 
        ("Options Trading", test_options_module),
        ("Database Connection", test_database_connection),
        ("FastAPI Application", test_fastapi_app),
        ("Database Initialization", test_database_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                status = "PASSED"
            else:
                status = "FAILED"
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            status = "ERROR"
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n🚀 Ready to use:")
        print("1. Initialize database: python3 database/postgres_init.py")
        print("2. Start API server: python3 main.py")
        print("3. Visit docs: http://localhost:8000/docs")
        print("4. Run example: python3 examples/postgresql_usage.py")
        return 0
    else:
        print("❌ Some systems need attention")
        print("\n🔧 Next steps:")
        print("1. Check missing dependencies above")
        print("2. Ensure PostgreSQL is running: docker compose up -d db")
        print("3. Review error messages for specific issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 