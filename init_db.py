#!/usr/bin/env python3
"""Simple database initialization script"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from database.postgres_init import init_database, drop_tables, get_db
    
    print("Resetting database...")
    db = get_db()
    drop_tables(db)
    
    print("Initializing database...")
    success = init_database()
    
    if success:
        print("Database initialization complete!")
    else:
        print("Database initialization failed!")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 