#!/usr/bin/env python3
"""
Simple test script to diagnose backend startup issues
"""

import sys
import os
import traceback

print("🔍 Testing backend components step by step...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("\n1. Testing basic imports...")
    import fastapi
    import uvicorn
    print("✅ FastAPI and Uvicorn imports OK")
    
    print("\n2. Testing environment setup...")
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✅ Environment loaded")
    
    print("\n3. Testing core config...")
    from src.core.config import get_settings
    settings = get_settings()
    print(f"✅ Settings loaded - Environment: {settings.environment}")
    
    print("\n4. Testing database config...")
    try:
        print(f"Database URL: {settings.database.url}")
        print("✅ Database config OK")
    except Exception as e:
        print(f"⚠️ Database config issue: {e}")
    
    print("\n5. Testing cache config...")
    try:
        print(f"Redis URL: {settings.redis.url}")
        print("✅ Redis config OK")
    except Exception as e:
        print(f"⚠️ Redis config issue: {e}")
        
    print("\n6. Testing main app import...")
    from src.main_refactored import app
    print("✅ Main app imported successfully")
    
    print("\n7. Testing database connection...")
    try:
        from src.database.postgres_connection import init_db_connection
        init_db_connection()
        print("✅ Database connection initialized")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        
    print("\n8. Testing cache connection...")
    try:
        from src.core.cache import TradingCache
        cache = TradingCache()
        print("⚠️ Cache connection attempted (may fail without Redis)")
    except Exception as e:
        print(f"⚠️ Cache connection failed: {e}")
    
    print("\n✅ All basic tests passed! Ready to start server.")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    print("\n🔍 Full traceback:")
    traceback.print_exc()
    sys.exit(1)
    
print("\n🚀 Starting server...")
if __name__ == "__main__":
    uvicorn.run(
        "src.main_refactored:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for testing
        log_level="info"
    ) 