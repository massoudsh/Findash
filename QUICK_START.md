# 🚀 Octopus Trading Platform - Quick Start Guide

**Get your trading platform running in under 5 minutes using FREE services only!**

## ⚡ Ultra-Quick Start (Recommended)

```bash
# 1. Clone and enter the project
cd Modules

# 2. Run the automated setup
python quick_start.py

# 3. That's it! Platform will be running at:
#    - Frontend: http://localhost:3000
#    - API: http://localhost:8000
#    - Docs: http://localhost:8000/docs
```

## 🔑 Demo Credentials

**Ready-to-use accounts:**
- **Admin**: `admin@octopus.trading` / `SecureAdmin2025!`
- **Trader**: `trader@octopus.trading` / `TraderPro2025!`
- **Demo User**: `demo@octopus.trading` / `DemoUser2025!`

## 📋 Manual Setup (If needed)

### Prerequisites
- Python 3.8+
- Docker (optional but recommended)
- Node.js 16+ (for frontend)

### Step 1: Environment Setup
```bash
# Copy secure environment template
cp env.example env.local

# Generate secure secrets (replace in env.local)
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

### Step 2: Start Services
**With Docker (Easy):**
```bash
# Start PostgreSQL
docker run -d --name octopus-postgres \
  -e POSTGRES_DB=trading_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 postgres:15-alpine

# Start Redis
docker run -d --name octopus-redis \
  -p 6379:6379 redis:7-alpine
```

**Without Docker:**
- Install PostgreSQL: https://www.postgresql.org/download/
- Install Redis: https://redis.io/download

### Step 3: Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies (optional)
cd frontend-nextjs
npm install
cd ..
```

### Step 4: Start the Platform
```bash
# Start backend
python -m uvicorn src.main_refactored:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (in another terminal)
cd frontend-nextjs
npm run dev
```

## 🆓 Free Data Sources (Already Configured)

The platform uses **100% FREE** data sources:

### Stock Market Data
- **Yahoo Finance** - Primary source (unlimited)
- **Alpha Vantage** - 25 calls/day (free tier)
- **Finnhub** - 60 calls/minute (free tier)
- **IEX Cloud** - Free tier available

### Cryptocurrency Data
- **CoinGecko** - 100 calls/minute (no API key needed)
- **Binance Public API** - 1200 requests/minute
- **CryptoCompare** - Free tier available

### Getting Better API Keys (Optional)
1. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
2. **Finnhub**: https://finnhub.io/register
3. **IEX Cloud**: https://iexcloud.io/
4. **News API**: https://newsapi.org/

Update keys in `env.local` file:
```bash
ALPHA_VANTAGE_API_KEY=your_real_key_here
FINNHUB_API_KEY=your_real_key_here
NEWS_API_KEY=your_real_key_here
```

## 🛡️ Security Features (Production-Ready)

✅ **Secure by Default:**
- Auto-generated 32+ character secrets
- bcrypt password hashing (12 rounds)
- JWT tokens with proper expiration
- Rate limiting (30 req/min with burst protection)
- Failed login attempt tracking
- Environment-based configuration

✅ **No Hardcoded Secrets:**
- All credentials in environment variables
- Demo passwords configurable
- Secure fallbacks for missing configs

## 📊 Platform Features

### Trading Engine
- Real-time market data from multiple free sources
- Portfolio management and tracking
- Risk management tools
- Backtesting capabilities

### Analytics
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market sentiment analysis
- Performance metrics
- Custom dashboards

### User Management
- Multi-user support with roles
- Authentication with NextAuth.js
- API key management
- Audit logging

## 🔧 Configuration Options

### Rate Limiting (Free-Tier Friendly)
```bash
RATE_LIMIT_PER_MINUTE=30        # API calls per minute
RATE_LIMIT_BURST=10             # Burst limit
MAX_LOGIN_ATTEMPTS=5            # Account lockout threshold
```

### Trading Settings
```bash
DEFAULT_PORTFOLIO_VALUE=100000  # Starting portfolio value
MAX_POSITION_SIZE=0.1           # Maximum position size (10%)
RISK_FREE_RATE=0.05            # Risk-free rate for calculations
```

### Data Sources
```bash
# Enable/disable specific sources
ENABLE_YAHOO_FINANCE=true
ENABLE_ALPHA_VANTAGE=true
ENABLE_COINGECKO=true
ENABLE_FINNHUB=true
```

## 🚀 Deployment Options

### 1. Local Development
Perfect for testing and development:
```bash
python quick_start.py
```

### 2. Cloud Deployment (Free Tiers)
- **Heroku**: 550 free hours/month
- **Railway**: $5/month with free trial
- **Render**: Free tier available
- **DigitalOcean**: $5/month droplet

### 3. VPS Deployment
Any VPS with 1GB RAM and Docker support.

## 🆘 Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check if ports are busy
netstat -tulpn | grep :8000

# Check environment
python -c "from src.core.config import get_settings; print(get_settings())"
```

**Database connection failed:**
```bash
# Test PostgreSQL connection
psql postgresql://postgres:password@localhost:5432/trading_db

# Check Docker containers
docker ps
```

**Frontend issues:**
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**Rate limiting issues:**
```bash
# Check Redis connection
redis-cli ping

# Monitor rate limits
redis-cli monitor
```

### Support Resources
- 📚 API Documentation: http://localhost:8000/docs
- 🐛 Issues: Create GitHub issues for bugs
- 💬 Community: Join discussions for help

## 📈 Scaling to Production

### 1. Get Better API Keys
Replace demo keys with real free-tier API keys for higher limits.

### 2. Enable Production Security
```bash
# In env.local
ENVIRONMENT=production
DEBUG=false
FORCE_HTTPS=true
SECURE_COOKIES=true
```

### 3. Add Monitoring (Free)
- **Prometheus + Grafana**: Built-in support
- **Sentry**: Free error tracking
- **Uptime monitoring**: UptimeRobot (free)

### 4. Database Scaling
- **PostgreSQL optimization**: Built-in connection pooling
- **Redis clustering**: For high availability
- **TimescaleDB**: For time-series data (free extension)

## 🎯 Roadmap to Institutional Features

**Phase 1 (Current): MVP Launch**
- ✅ Secure authentication
- ✅ Real-time data
- ✅ Basic portfolio management
- ✅ Free data sources

**Phase 2: User Growth**
- User registration system
- Email notifications
- Advanced charting
- Social features

**Phase 3: Monetization**
- Premium data feeds
- Advanced analytics
- Institutional APIs
- White-label solutions

**Phase 4: Institutional**
- Professional compliance
- Audit trails
- Enterprise security
- Custom deployment

---

## 🎉 You're Ready!

Your trading platform is now running with:
- **Production-grade security**
- **Real market data**
- **Multiple free data sources**
- **Scalable architecture**
- **Zero monthly costs**

**Start trading at: http://localhost:3000**

*Built for rapid deployment and scaling to institutional grade.* 