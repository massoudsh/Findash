#!/usr/bin/env python3
"""
Test script to verify Quantum Trading Matrix™ setup
Tests PostgreSQL connection, dependencies, and basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    test_modules = [
        ("psycopg2", "PostgreSQL driver"),
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("pandas", "Data analysis"),
        ("numpy", "Numerical computing"),
        ("yfinance", "Market data"),
        ("passlib", "Password hashing"),
        ("pydantic", "Data validation"),
        ("scipy", "Scientific computing"),
    ]
    
    failed_imports = []
    
    for module_name, description in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} ({description})")
        except ImportError as e:
            print(f"❌ {module_name} ({description}) - {e}")
            failed_imports.append(module_name)
    
    return len(failed_imports) == 0

def test_database_imports():
    """Test database module imports"""
    print("\n🗄️ Testing database module imports...")
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from database.postgres_connection import get_db
        print("✅ Database connection module")
        
        from database.repositories import UserRepository, PortfolioRepository
        print("✅ Database repositories")
        
        return True
    except ImportError as e:
        print(f"❌ Database imports failed: {e}")
        return False

def test_options_module():
    """Test options trading module"""
    print("\n📈 Testing options trading module...")
    
    try:
        from options_risk_integration import BlackScholesCalculator, OptionsPortfolio
        print("✅ Options trading module")
        
        # Test Black-Scholes calculation
        price = BlackScholesCalculator.option_price(100, 100, 0.25, 0.05, 0.2, 'call')
        print(f"✅ Black-Scholes calculation: ${price:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Options module failed: {e}")
        return False

def test_database_connection():
    """Test actual database connection"""
    print("\n🔌 Testing database connection...")
    
    try:
        from database.postgres_connection import get_db
        
        db = get_db()
        result = db.execute_query("SELECT 1 as test", fetch='one')
        
        if result and result['test'] == 1:
            print("✅ PostgreSQL connection successful")
            return True
        else:
            print("❌ PostgreSQL connection failed - invalid response")
            return False
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure PostgreSQL is running: docker compose up -d db")
        return False

def test_fastapi_app():
    """Test if FastAPI app can be imported and created"""
    print("\n🌐 Testing FastAPI application...")
    
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
        
        # Test that the app has endpoints
        routes = [route.path for route in app.routes]
        expected_routes = ["/health", "/", "/portfolio/options/add"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
        
        return True
    except Exception as e:
        print(f"❌ FastAPI app failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Quantum Trading Matrix™ - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_imports),
        ("Database Modules", test_database_imports),
        ("Options Trading", test_options_module),
        ("Database Connection", test_database_connection),
        ("FastAPI Application", test_fastapi_app),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready!")
        print("\n🚀 Next steps:")
        print("1. Initialize database: python3 database/postgres_init.py")
        print("2. Start API server: python3 main.py") 
        print("3. Visit API docs: http://localhost:8000/docs")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 