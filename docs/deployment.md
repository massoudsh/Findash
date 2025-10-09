# 🚀 Production Deployment Guide - Octopus Trading Platform™

## Overview

This guide covers the complete production deployment of the Octopus Trading Platform, from infrastructure setup to monitoring and maintenance.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS or RHEL 8+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 500GB SSD minimum, 1TB recommended
- **CPU**: 8 cores minimum, 16 cores recommended
- **Network**: Gigabit internet connection

### Required Software
- Docker 24.0+
- Docker Compose 2.20+
- NGINX 1.20+
- SSL certificates (Let's Encrypt or commercial)
- PostgreSQL 15+ with TimescaleDB
- Redis 7+

## Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Microservices │
│   (NGINX)       │───▶│   (Kong)        │───▶│   (FastAPI)     │
│   Port 80/443   │    │   Port 8000     │    │   Port 8010     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Event Stream  │    │   Database      │
│   (Next.js)     │    │   (Kafka)       │    │   (PostgreSQL)  │
│   Port 3000     │    │   Port 9092     │    │   Port 5432     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Deployment Methods

### Method 1: Docker Compose (Recommended for Single Server)

#### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Create application directory
sudo mkdir -p /opt/octopus-trading
sudo chown $USER:$USER /opt/octopus-trading
```

#### 2. Application Setup
```bash
# Clone repository
cd /opt/octopus-trading
git clone https://github.com/your-org/octopus-trading-platform.git .

# Copy production environment
cp env.example .env

# Generate secure secrets
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env
```

#### 3. Environment Configuration
```bash
# Edit production environment
nano .env
```

**Production .env Configuration:**
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security (Generated above)
SECRET_KEY=your-generated-secret-key
JWT_SECRET_KEY=your-generated-jwt-secret

# Database
DATABASE_URL=postgresql://octopus_app:secure_password@db:5432/trading_db
DB_PASSWORD=secure_database_password

# Redis
REDIS_URL=redis://:secure_redis_password@redis:6379/0
REDIS_PASSWORD=secure_redis_password

# External APIs
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
NEWS_API_KEY=your-news-api-key

# Production Security
FORCE_HTTPS=true
SECURE_COOKIES=true
HSTS_MAX_AGE=31536000

# CORS (Update with your domain)
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Email (Optional)
SMTP_HOST=smtp.yourdomain.com
SMTP_USER=noreply@yourdomain.com
SMTP_PASSWORD=your-smtp-password

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

#### 4. SSL Certificate Setup
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com -d www.your-domain.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### 5. NGINX Configuration
```bash
# Create NGINX config
sudo nano /etc/nginx/sites-available/octopus-trading
```

**NGINX Configuration:**
```nginx
# /etc/nginx/sites-available/octopus-trading
upstream api_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

upstream frontend_backend {
    server 127.0.0.1:3000;
    keepalive 32;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    # API Routes
    location /api/ {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket Routes
    location /ws/ {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend Routes
    location / {
        proxy_pass http://frontend_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

```bash
# Enable site and restart NGINX
sudo ln -s /etc/nginx/sites-available/octopus-trading /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 6. Deploy Application
```bash
# Start all services
docker compose -f docker-compose-complete.yml up -d

# Verify deployment
docker compose ps
docker compose logs -f
```

### Method 2: Kubernetes (Recommended for Multi-Server)

#### 1. Kubernetes Manifests

**Namespace:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: octopus-trading
```

**ConfigMap:**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: octopus-config
  namespace: octopus-trading
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

**Secrets:**
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: octopus-secrets
  namespace: octopus-trading
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  DATABASE_URL: <base64-encoded-db-url>
```

**API Deployment:**
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: octopus-api
  namespace: octopus-trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: octopus-api
  template:
    metadata:
      labels:
        app: octopus-api
    spec:
      containers:
      - name: api
        image: octopus-trading-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: octopus-config
        - secretRef:
            name: octopus-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Database Setup

### PostgreSQL Configuration
```bash
# Create database user
sudo -u postgres psql
CREATE ROLE octopus_app WITH LOGIN PASSWORD 'secure_password';
CREATE DATABASE trading_db OWNER octopus_app;
\q

# Install TimescaleDB extension
sudo -u postgres psql -d trading_db
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q
```

### Database Migration
```bash
# Run initial migrations
docker compose exec api python -m alembic upgrade head

# Verify tables
docker compose exec db psql -U octopus_app -d trading_db -c "\dt"
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'octopus-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'octopus-postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'octopus-redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```bash
# Import trading-specific dashboards
curl -X POST \
  http://admin:password@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/trading-overview.json
```

## Security Hardening

### Firewall Configuration
```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### Container Security
```bash
# Run containers as non-root
# Add to Dockerfile:
RUN groupadd -r octopus && useradd -r -g octopus octopus
USER octopus
```

### Database Security
```sql
-- Restrict database permissions
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO octopus_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO octopus_app;
```

## Backup & Recovery

### Automated Backups
```bash
# Create backup script
sudo nano /opt/scripts/backup-octopus.sh
```

**Backup Script:**
```bash
#!/bin/bash
# /opt/scripts/backup-octopus.sh

BACKUP_DIR="/opt/backups/octopus"
DATE=$(date +%Y%m%d_%H%M%S)
DB_BACKUP="$BACKUP_DIR/db_backup_$DATE.sql"
FILES_BACKUP="$BACKUP_DIR/files_backup_$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
docker compose exec -T db pg_dump -U octopus_app trading_db > $DB_BACKUP

# Files backup
tar -czf $FILES_BACKUP \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='.venv' \
  /opt/octopus-trading

# Upload to S3 (optional)
aws s3 cp $DB_BACKUP s3://your-backup-bucket/database/
aws s3 cp $FILES_BACKUP s3://your-backup-bucket/files/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

```bash
# Make executable and schedule
sudo chmod +x /opt/scripts/backup-octopus.sh
sudo crontab -e
# Add: 0 2 * * * /opt/scripts/backup-octopus.sh
```

### Recovery Procedures
```bash
# Database recovery
docker compose exec -T db psql -U octopus_app -d trading_db < backup_file.sql

# Files recovery
tar -xzf files_backup.tar.gz -C /opt/
```

## Performance Optimization

### Database Tuning
```bash
# PostgreSQL configuration optimizations
sudo nano /etc/postgresql/15/main/postgresql.conf
```

**PostgreSQL Settings:**
```ini
# Memory
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 256MB

# Checkpoints
checkpoint_completion_target = 0.9
checkpoint_timeout = 10min
max_wal_size = 4GB

# Connections
max_connections = 200
```

### Application Scaling
```bash
# Scale API instances
docker compose up -d --scale api=3

# Use connection pooling
# Add to .env:
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

## Health Checks & Monitoring

### Application Health Endpoints
- **API Health**: `GET /health`
- **Database Health**: `GET /health/db`
- **Redis Health**: `GET /health/redis`
- **Comprehensive Health**: `GET /health/all`

### Monitoring Alerts
```yaml
# monitoring/alert_rules.yml
groups:
- name: octopus_alerts
  rules:
  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status=~"5.."}[5m])) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: DatabaseDown
    expr: up{job="octopus-postgres"} == 0
    for: 1m
    annotations:
      summary: "Database is down"
      
  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
    for: 10m
    annotations:
      summary: "High memory usage"
```

## Troubleshooting

### Common Issues

1. **API Not Responding**
   ```bash
   # Check container status
   docker compose ps
   docker compose logs api
   
   # Check resource usage
   docker stats
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   docker compose exec api python -c "
   from src.database.postgres_connection import init_db_connection
   init_db_connection()
   print('Database connection successful')
   "
   ```

3. **SSL Certificate Issues**
   ```bash
   # Renew certificates
   sudo certbot renew
   sudo systemctl reload nginx
   ```

### Log Analysis
```bash
# Application logs
docker compose logs -f api

# NGINX logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# System logs
sudo journalctl -f -u docker
```

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system resources
- Check application logs
- Verify backup completion

**Weekly:**
- Review security logs
- Update system packages
- Database maintenance

**Monthly:**
- SSL certificate renewal check
- Security audit
- Performance review

### Update Procedures
```bash
# Application updates
git pull origin main
docker compose build
docker compose up -d

# System updates
sudo apt update && sudo apt upgrade
sudo reboot
```

## Compliance & Auditing

### Audit Trail
- All user actions logged to `audit_log` table
- Security events tracked in `security_events`
- API access logged via NGINX

### Regulatory Requirements
- **Data Retention**: 7 years for financial data
- **Encryption**: AES-256 for data at rest
- **Access Controls**: Role-based permissions
- **Audit Logs**: Immutable, cryptographically signed

---

## Support & Escalation

### Contact Information
- **DevOps Team**: devops@octopus.trading
- **Security Team**: security@octopus.trading
- **Emergency**: +1-555-EMERGENCY

### Escalation Procedures
1. **Level 1**: Application issues → Development team
2. **Level 2**: Infrastructure issues → DevOps team
3. **Level 3**: Security incidents → Security team
4. **Level 4**: Business impact → Management

---

*Deployment guide version: 1.2*  
*Last updated: January 2025* 