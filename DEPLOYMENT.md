# 🚀 Quantum Trading Matrix™ - Production Deployment Guide

This guide covers the complete deployment process for the Quantum Trading Matrix™ platform, from development to production.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **RAM**: Minimum 8GB, Recommended 16GB+
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Storage**: Minimum 50GB SSD, Recommended 200GB+ SSD
- **Network**: Stable internet connection with low latency

### Software Dependencies
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: Latest version
- **SSL Certificate**: For HTTPS (Let's Encrypt recommended)

### API Keys & Accounts
- Alpha Vantage API key (for market data)
- Alpaca API credentials (for paper/live trading)
- Sentry account (for error monitoring)
- AWS/GCP account (for cloud deployment)

## 🛠️ Pre-Deployment Setup

### 1. Server Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/quantum-trading-matrix.git
cd quantum-trading-matrix
```

### 3. Environment Configuration
```bash
# Copy production environment template
cp config/production.env .env

# Edit environment variables
nano .env
```

### 4. SSL Certificate Setup (for HTTPS)
```bash
# Using Let's Encrypt (recommended)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com

# Or use your own certificates
mkdir -p nginx/ssl
# Copy your SSL certificates to nginx/ssl/
```

## 🚀 Deployment Methods

### Method 1: Automated Deployment (Recommended)

The automated deployment script handles all aspects of the deployment process:

```bash
# Make deployment script executable
chmod +x scripts/deploy.sh

# Run deployment with all checks
./scripts/deploy.sh

# Quick deployment (skip tests and backup)
./scripts/deploy.sh --force --skip-tests --skip-backup

# Dry run (preview what will be deployed)
./scripts/deploy.sh --dry-run
```

### Method 2: Manual Deployment

#### Step 1: Build Docker Images
```bash
# Build API image
docker build -t quantumtradingmatrix/api:latest .

# Build frontend image (if separate)
docker build -t quantumtradingmatrix/frontend:latest frontend/
```

#### Step 2: Database Setup
```bash
# Start database
docker-compose -f deploy/production.yml up -d db

# Initialize database
docker exec qtm-api python database/init_db.py
```

#### Step 3: Deploy Application
```bash
# Deploy all services
docker-compose -f deploy/production.yml up -d

# Check status
docker-compose -f deploy/production.yml ps
```

## 🔧 Configuration

### Environment Variables

Update `.env` file with your production values:

```bash
# Application
ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-minimum-32-chars
DEBUG=false

# Database
DATABASE_URL=postgresql://qtm_user:secure-password@localhost:5432/qtm_prod
DB_PASSWORD=your-secure-database-password

# Redis
REDIS_URL=redis://:redis-password@localhost:6379/0
REDIS_PASSWORD=your-secure-redis-password

# API Keys
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret

# Monitoring
SENTRY_DSN=your-sentry-dsn
GRAFANA_PASSWORD=your-grafana-password
```

### Nginx Configuration

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS Configuration
    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # API
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # WebSocket support
        location /ws/ {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## 📊 Monitoring & Observability

### Built-in Monitoring Stack

The deployment includes:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **ELK Stack**: Log aggregation and analysis
- **Sentry**: Error tracking

### Access Monitoring Dashboards

```bash
# Grafana (monitoring dashboards)
http://yourdomain.com:3000
Username: admin
Password: [GRAFANA_PASSWORD from .env]

# Prometheus (metrics)
http://yourdomain.com:9090

# Kibana (logs)
http://yourdomain.com:5601
```

### Health Checks

```bash
# Application health
curl https://yourdomain.com/health

# Database health
docker exec qtm-postgres pg_isready -U qtm_user

# Redis health
docker exec qtm-redis redis-cli ping
```

## 🔒 Security Considerations

### 1. Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
```

### 2. SSL/TLS Configuration
- Use strong SSL certificates (Let's Encrypt or commercial)
- Enable HSTS headers
- Use secure cipher suites

### 3. Database Security
- Use strong passwords
- Enable SSL connections
- Regular backups with encryption

### 4. API Security
- Rate limiting enabled
- JWT tokens with short expiration
- API key rotation policy

## 📦 Backup & Recovery

### Automated Backups

Backups are automatically created daily:
```bash
# Manual backup
docker exec qtm-postgres pg_dump -U qtm_user qtm_prod > backup_$(date +%Y%m%d).sql

# Restore from backup
docker exec -i qtm-postgres psql -U qtm_user qtm_prod < backup_20231201.sql
```

### Data Persistence

Important data is persisted in Docker volumes:
- Database data: `/opt/qtm/data/postgres`
- Redis data: `/opt/qtm/data/redis`
- Application logs: `/opt/qtm/logs`

## 🚦 Deployment Verification

### 1. Service Health Checks
```bash
# Check all containers are running
docker ps

# Check service logs
docker logs qtm-api
docker logs qtm-postgres
docker logs qtm-redis
```

### 2. API Functionality Tests
```bash
# Health endpoint
curl https://yourdomain.com/api/health

# Options pricing
curl -X POST https://yourdomain.com/api/options/price \
  -H "Content-Type: application/json" \
  -d '{"underlying_price": 100, "strike": 100, "time_to_expiry": 0.25, "volatility": 0.2}'

# Portfolio status
curl https://yourdomain.com/api/portfolio/greeks
```

### 3. Performance Tests
```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 https://yourdomain.com/api/health

# Response time monitoring
curl -w "@curl-format.txt" -o /dev/null -s https://yourdomain.com/api/health
```

## 🔄 Updates & Maintenance

### Rolling Updates
```bash
# Build new image
docker build -t quantumtradingmatrix/api:v1.1 .

# Update service
docker service update --image quantumtradingmatrix/api:v1.1 qtm-api
```

### Scheduled Maintenance
```bash
# Stop services gracefully
docker-compose -f deploy/production.yml down

# Perform maintenance
# ...

# Restart services
docker-compose -f deploy/production.yml up -d
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker logs qtm-api

# Check resources
docker system df
docker system prune  # Clean up if needed
```

#### 2. Database Connection Issues
```bash
# Check database status
docker exec qtm-postgres pg_isready -U qtm_user

# Reset database connection
docker restart qtm-postgres
```

#### 3. High Memory Usage
```bash
# Monitor resource usage
docker stats

# Adjust memory limits in docker-compose.yml
```

#### 4. SSL Certificate Issues
```bash
# Check certificate expiry
openssl x509 -in /path/to/cert.pem -text -noout | grep "Not After"

# Renew Let's Encrypt certificate
sudo certbot renew
```

### Log Analysis
```bash
# Application logs
docker logs qtm-api --tail 100 -f

# System logs
journalctl -u docker.service -f

# Nginx logs
docker logs qtm-nginx --tail 100 -f
```

## 📞 Support & Resources

### Documentation
- [API Documentation](https://yourdomain.com/docs)
- [User Guide](./docs/USER_GUIDE.md)
- [Architecture Overview](./docs/ARCHITECTURE.md)

### Monitoring & Alerts
- Set up email/Slack alerts for critical issues
- Monitor key metrics: response time, error rate, CPU/memory usage
- Regular health checks and automated testing

### Backup & Disaster Recovery
- Test restore procedures regularly
- Maintain offsite backups
- Document recovery procedures

---

## 🎯 Next Steps

After successful deployment:

1. **Configure monitoring alerts**
2. **Set up automated testing**
3. **Implement CI/CD pipeline**
4. **Scale infrastructure as needed**
5. **Optimize performance based on metrics**

For additional support or questions, please refer to the project documentation or create an issue in the GitHub repository. 