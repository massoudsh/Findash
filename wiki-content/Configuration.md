# Configuration

Complete guide to configuring the Octopus Trading Platform.

## Environment Variables

### Core Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENVIRONMENT` | No | development | Environment mode (development, staging, production) |
| `DEBUG` | No | true | Enable debug mode |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Security

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | **Yes** | - | Application secret key |
| `JWT_SECRET_KEY` | **Yes** | - | JWT token signing key |
| `JWT_ACCESS_TOKEN_EXPIRES` | No | 3600 | Token expiration in seconds |
| `FORCE_HTTPS` | No | false | Force HTTPS redirects |
| `SECURE_COOKIES` | No | false | Use secure cookies |
| `HSTS_MAX_AGE` | No | 0 | HSTS header max age |

### Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No* | - | PostgreSQL connection string |
| `DB_HOST` | No | localhost | Database host |
| `DB_PORT` | No | 5432 | Database port |
| `DB_NAME` | No | trading_db | Database name |
| `DB_USER` | No | postgres | Database user |
| `DB_PASSWORD` | No | - | Database password |
| `DB_POOL_SIZE` | No | 10 | Connection pool size |
| `DB_MAX_OVERFLOW` | No | 20 | Max overflow connections |

*Required for production, optional for development mode

### Redis

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | No* | - | Redis connection string |
| `REDIS_HOST` | No | localhost | Redis host |
| `REDIS_PORT` | No | 6379 | Redis port |
| `REDIS_PASSWORD` | No | - | Redis password |
| `REDIS_DB` | No | 0 | Redis database number |

### Celery

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CELERY_BROKER_URL` | No | redis://localhost:6379/0 | Celery broker URL |
| `CELERY_RESULT_BACKEND` | No | redis://localhost:6379/1 | Result backend URL |
| `CELERY_TASK_ALWAYS_EAGER` | No | false | Run tasks synchronously |

### API Keys

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPHA_VANTAGE_API_KEY` | No | - | Alpha Vantage API key |
| `NEWS_API_KEY` | No | - | News API key |
| `TWITTER_API_KEY` | No | - | Twitter API key |
| `TWITTER_API_SECRET` | No | - | Twitter API secret |

### CORS

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORS_ORIGINS` | No | * | Allowed origins (comma-separated) |
| `CORS_ALLOW_CREDENTIALS` | No | true | Allow credentials |

### Monitoring

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROMETHEUS_ENABLED` | No | true | Enable Prometheus metrics |
| `GRAFANA_ADMIN_USER` | No | admin | Grafana admin username |
| `GRAFANA_ADMIN_PASSWORD` | No | admin | Grafana admin password |

---

## Configuration Files

### `.env` File Example

```bash
# ===========================================
# Octopus Trading Platform Configuration
# ===========================================

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Security (REQUIRED - generate secure values)
SECRET_KEY=your-super-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production
JWT_ACCESS_TOKEN_EXPIRES=3600

# Database (optional for dev mode)
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_db
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis (optional for dev mode)
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# External APIs
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
NEWS_API_KEY=your-news-api-key

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin
```

### Production `.env` Example

```bash
# ===========================================
# Production Configuration
# ===========================================

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Security
SECRET_KEY=<64-character-random-string>
JWT_SECRET_KEY=<64-character-random-string>
JWT_ACCESS_TOKEN_EXPIRES=1800
FORCE_HTTPS=true
SECURE_COOKIES=true
HSTS_MAX_AGE=31536000

# Database
DATABASE_URL=postgresql://octopus_app:secure_password@db:5432/trading_db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://:secure_redis_password@redis:6379/0

# CORS
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

---

## Pydantic Settings

The application uses Pydantic for configuration management:

```python
# src/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Security
    secret_key: str
    jwt_secret_key: str
    jwt_access_token_expires: int = 3600
    
    # Database
    database_url: Optional[str] = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # Redis
    redis_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Get settings instance
def get_settings() -> Settings:
    return Settings()
```

---

## Docker Configuration

### docker-compose-core.yml

**Services included (9 total):**
- API Server
- Frontend
- PostgreSQL/TimescaleDB
- Redis
- Celery Worker
- Celery Beat
- Flower
- Prometheus
- Grafana

**Use case:** Development and small deployments

### docker-compose-complete.yml

**Additional services (24 total):**
- API Gateway (Kong/APISIX)
- Service Discovery
- Kafka
- Elasticsearch
- Keycloak (Auth)
- pgAdmin
- Redis Commander
- Additional workers

**Use case:** Production and enterprise deployments

---

## Frontend Configuration

### `next.config.js`

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost', 'your-domain.com'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL,
  },
};

module.exports = nextConfig;
```

### Frontend Environment Variables

```bash
# frontend-nextjs/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=Octopus Trading
```

---

## Logging Configuration

### Python Logging

```python
# src/core/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
        },
    },
}
```

---

## Rate Limiting Configuration

```python
# Rate limit settings
RATE_LIMIT_DEFAULT = "100/minute"
RATE_LIMIT_BURST = "20/10seconds"
RATE_LIMIT_BY_KEY = {
    "auth": "10/minute",
    "trading": "50/minute",
    "market_data": "200/minute",
}
```

---

## Cache Configuration

```python
# Cache TTL settings (seconds)
CACHE_TTL = {
    "market_data": 60,        # 1 minute
    "portfolio": 300,         # 5 minutes
    "user_session": 3600,     # 1 hour
    "api_response": 60,       # 1 minute
}

# Cache namespaces
CACHE_NAMESPACES = {
    "market": "market:",
    "portfolio": "portfolio:",
    "user": "user:",
    "session": "session:",
}
```

---

## Agent Configuration

```yaml
# config/agents.yaml
agents:
  data_collector:
    enabled: true
    refresh_interval: 60
    sources:
      - yahoo_finance
      - alpha_vantage
      - news_api
      
  risk_manager:
    enabled: true
    var_confidence_levels: [0.95, 0.99, 0.999]
    max_portfolio_var: 0.05
    correlation_lookback: 252
    
  strategy_agent:
    enabled: true
    strategies:
      - momentum
      - mean_reversion
      - technical_analysis
    signal_fusion: weighted_average
    
  ml_models:
    enabled: true
    models:
      - prophet
      - random_forest
    prediction_horizons: [1, 7, 30]
```

---

## Security Best Practices

### Generating Secrets

```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Generate JWT_SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
```

### Environment Variable Security

1. **Never commit `.env` files** to version control
2. **Use different secrets** for each environment
3. **Rotate secrets** regularly in production
4. **Use a secrets manager** (AWS Secrets Manager, HashiCorp Vault) for production

---

## Next Steps

- [[Getting Started]] - Installation guide
- [[Deployment]] - Production deployment
- [[Architecture]] - System architecture
