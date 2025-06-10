# 🚀 Quick Fix Guide - PostgreSQL Migration

## The Problem
You're seeing import errors because some Python dependencies are missing and there are module path issues.

## ✅ **What's Already Working**
Based on our tests:
- ✅ PostgreSQL connection (psycopg2 installed and working)
- ✅ FastAPI framework 
- ✅ Core dependencies (pandas, numpy, yfinance, pydantic)
- ✅ Docker PostgreSQL database is running

## ❌ **What Needs Fixing**
- ❌ Missing packages: `passlib`, `scipy`, `scikit-learn`
- ❌ Python path issues for database imports

## 🔧 **Manual Fix Steps**

### Step 1: Install Missing Dependencies
```bash
# Run these commands one by one in Terminal (not PowerShell)
python3 -m pip install passlib
python3 -m pip install scipy  
python3 -m pip install scikit-learn
python3 -m pip install bcrypt
python3 -m pip install python-jose[cryptography]
```

### Step 2: Fix Import Paths
The database module imports are failing. Here's the fix:

**Update `database/repositories.py`** - Change line 11:
```python
# Change this:
from database.postgres_connection import get_db

# To this:
from .postgres_connection import get_db
```

**Update `database/postgres_init.py`** - Change lines 13-14:
```python
# Change these:
from database.postgres_connection import get_db, DatabaseConfig
from database.repositories import UserRepository, PortfolioRepository

# To these:
from .postgres_connection import get_db, DatabaseConfig
from .repositories import UserRepository, PortfolioRepository
```

### Step 3: Update Main Application
**Update `main.py`** - Change lines 12-16:
```python
# Change these imports:
from database.repositories import (
    UserRepository, PortfolioRepository, OptionPositionRepository,
    MarketDataRepository, APIKeyRepository, AuditLogRepository
)
from database.postgres_connection import get_db

# To these:
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.repositories import (
    UserRepository, PortfolioRepository, OptionPositionRepository,
    MarketDataRepository, APIKeyRepository, AuditLogRepository
)
from database.postgres_connection import get_db
```

### Step 4: Test the Fix
```bash
# Run this test
python3 test_setup.py
```

You should see:
```
✅ All tests passed! Your setup is ready!
```

### Step 5: Initialize Database
```bash
# Initialize the PostgreSQL database
python3 database/postgres_init.py

# Verify the setup
python3 database/postgres_init.py --verify
```

### Step 6: Start the Application
```bash
# Start the FastAPI server
python3 main.py
```

Then visit:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 🧪 **Test API Endpoints**

### Test Option Price Calculation:
```bash
curl -X POST "http://localhost:8000/options/price" \
  -H "Content-Type: application/json" \
  -d '{
    "underlying_price": 100,
    "strike": 100,
    "time_to_expiry": 0.25,
    "volatility": 0.2
  }'
```

### Test Portfolio Management:
```bash
# Add option position (this will use demo user)
curl -X POST "http://localhost:8000/portfolio/options/add" \
  -H "Authorization: Bearer demo-token" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "option_type": "call",
    "strike": 150,
    "expiry_days": 30,
    "quantity": 10,
    "premium": 5.0,
    "underlying_price": 155,
    "volatility": 0.25
  }'
```

## 🐛 **If You Still Get Errors**

### Error: "No module named 'passlib'"
```bash
python3 -m pip install --user passlib[bcrypt]
```

### Error: "No module named 'scipy'"
```bash
python3 -m pip install --user scipy
```

### Error: "Database connection failed"
```bash
# Make sure PostgreSQL is running
docker compose up -d db

# Check if it's healthy
docker ps | grep postgres
```

### Error: "ModuleNotFoundError: No module named 'database'"
Add this to the top of the file having issues:
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## 🎯 **Expected Results**

After fixing everything, you should be able to:

1. ✅ **Run Database Init**: `python3 database/postgres_init.py`
2. ✅ **Start API Server**: `python3 main.py`
3. ✅ **Access API Docs**: http://localhost:8000/docs
4. ✅ **Test Options Trading**: Use the API endpoints
5. ✅ **Run Examples**: `python3 examples/postgresql_usage.py`

## 🚀 **Key Benefits You'll Get**

- **Performance**: 🚀 Direct PostgreSQL queries (no ORM overhead)
- **Scalability**: 📈 Connection pooling for concurrent users
- **Control**: 🎛️ Fine-tuned SQL optimization
- **Features**: 🐘 Native PostgreSQL features (JSONB, triggers, UUIDs)
- **Debugging**: 🔍 See exact SQL queries being executed

## 📞 **If You Need Help**

If you're still having issues:
1. Run `python3 test_setup.py` and share the output
2. Check Docker: `docker ps | grep postgres`
3. Check Python path: `python3 -c "import sys; print(sys.path)"`
4. Test imports manually: `python3 -c "import psycopg2, fastapi; print('OK')"`

Your PostgreSQL migration is almost complete! 🎉 