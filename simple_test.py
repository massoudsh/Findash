#!/usr/bin/env python3
"""Simple test - no fancy output to avoid PowerShell issues"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("TESTING IMPORTS...")

# Test psycopg2
try:
    import psycopg2
    print("psycopg2: OK")
except:
    print("psycopg2: FAIL")

# Test fastapi
try:
    import fastapi
    print("fastapi: OK")
except:
    print("fastapi: FAIL")

# Test matplotlib
try:
    import matplotlib
    print("matplotlib: OK")
except:
    print("matplotlib: FAIL")

# Test database modules
try:
    from database.postgres_connection import get_db
    print("database.postgres_connection: OK")
except Exception as e:
    print(f"database.postgres_connection: FAIL - {e}")

# Test options module
try:
    from options_risk_integration import BlackScholesCalculator
    print("options_risk_integration: OK")
except Exception as e:
    print(f"options_risk_integration: FAIL - {e}")

# Test main app
try:
    from main import app
    print("main app: OK")
except Exception as e:
    print(f"main app: FAIL - {e}")

# Test database connection
try:
    from database.postgres_connection import get_db
    db = get_db()
    result = db.execute_query("SELECT 1", fetch='one')
    if result:
        print("database connection: OK")
    else:
        print("database connection: FAIL")
except Exception as e:
    print(f"database connection: FAIL - {e}")

print("TEST COMPLETE") 