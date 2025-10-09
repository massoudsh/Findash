"""
Production Deployment Configuration
Advanced deployment and scaling configuration

This module handles:
- Docker containerization
- Kubernetes deployment manifests
- Production environment variables
- Monitoring and logging configuration
- Auto-scaling and load balancing
- Security and performance settings
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ScalingMode(Enum):
    MANUAL = "manual"
    AUTO = "auto"
    PREDICTIVE = "predictive"

@dataclass
class DatabaseConfig:
    """Database configuration for production"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "require"
    connection_pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis configuration for caching and sessions"""
    host: str
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    connection_pool_size: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    jwt_secret: str
    jwt_expiration_hours: int = 24
    password_hash_rounds: int = 12
    rate_limit_per_minute: int = 100
    cors_origins: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    ssl_redirect: bool = True
    hsts_max_age: int = 31536000

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = True
    jaeger_endpoint: str = "http://jaeger:14268/api/traces"
    log_level: str = "INFO"
    structured_logging: bool = True
    metrics_retention_days: int = 30

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    mode: ScalingMode = ScalingMode.AUTO
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_percentage: int = 70
    target_memory_percentage: int = 80
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period_seconds: int = 300

class ProductionConfig:
    """Production deployment configuration"""
    
    def __init__(self, environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION):
        self.environment = environment
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from environment variables"""
        
        # Database Configuration
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "quantum_trading"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            connection_pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10"))
        )
        
        # Redis Configuration
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            connection_pool_size=int(os.getenv("REDIS_POOL_SIZE", "50"))
        )
        
        # Security Configuration
        self.security = SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", ""),
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
            cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
            allowed_hosts=os.getenv("ALLOWED_HOSTS", "").split(",") if os.getenv("ALLOWED_HOSTS") else []
        )
        
        # Monitoring Configuration
        self.monitoring = MonitoringConfig(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            grafana_enabled=os.getenv("GRAFANA_ENABLED", "true").lower() == "true",
            jaeger_enabled=os.getenv("JAEGER_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
        )
        
        # Scaling Configuration
        self.scaling = ScalingConfig(
            mode=ScalingMode(os.getenv("SCALING_MODE", "auto")),
            min_replicas=int(os.getenv("MIN_REPLICAS", "2")),
            max_replicas=int(os.getenv("MAX_REPLICAS", "10")),
            target_cpu_percentage=int(os.getenv("TARGET_CPU_PERCENTAGE", "70")),
            target_memory_percentage=int(os.getenv("TARGET_MEMORY_PERCENTAGE", "80"))
        )
        
        # Application Configuration
        self.app_config = {
            "host": os.getenv("APP_HOST", "0.0.0.0"),
            "port": int(os.getenv("APP_PORT", "8000")),
            "workers": int(os.getenv("WORKERS", "4")),
            "worker_class": os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker"),
            "timeout": int(os.getenv("TIMEOUT", "120")),
            "keepalive": int(os.getenv("KEEPALIVE", "2")),
            "max_requests": int(os.getenv("MAX_REQUESTS", "1000")),
            "max_requests_jitter": int(os.getenv("MAX_REQUESTS_JITTER", "100"))
        }
        
        # Broker Configuration
        self.broker_config = {
            "interactive_brokers": {
                "host": os.getenv("IB_HOST", "localhost"),
                "port": int(os.getenv("IB_PORT", "7497")),
                "client_id": int(os.getenv("IB_CLIENT_ID", "1")),
                "timeout": int(os.getenv("IB_TIMEOUT", "60"))
            },
            "alpaca": {
                "api_key": os.getenv("ALPACA_API_KEY", ""),
                "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
                "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                "data_url": os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
            },
            "td_ameritrade": {
                "api_key": os.getenv("TDA_API_KEY", ""),
                "refresh_token": os.getenv("TDA_REFRESH_TOKEN", ""),
                "account_id": os.getenv("TDA_ACCOUNT_ID", "")
            }
        }
    
    def get_docker_config(self) -> Dict[str, Any]:
        """Generate Docker configuration"""
        
        return {
            "dockerfile": """
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY alembic.ini .
COPY setup.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app

USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "src.main:app"]
""",
            "docker_compose": """
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: quantum_trading
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
""",
            "gunicorn_config": """
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('APP_PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = 1000
timeout = int(os.getenv('TIMEOUT', 120))
keepalive = int(os.getenv('KEEPALIVE', 2))

# Restart workers after this many requests
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', 100))

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'quantum_trading_matrix'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
preload_app = True
"""
        }
    
    def get_kubernetes_config(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifests"""
        
        return {
            "namespace": """
apiVersion: v1
kind: Namespace
metadata:
  name: quantum-trading
""",
            "deployment": f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-trading-app
  namespace: quantum-trading
  labels:
    app: quantum-trading
spec:
  replicas: {self.scaling.min_replicas}
  selector:
    matchLabels:
      app: quantum-trading
  template:
    metadata:
      labels:
        app: quantum-trading
    spec:
      containers:
      - name: quantum-trading
        image: quantum-trading:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "{self.environment.value}"
        - name: DB_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: quantum-trading-secrets
              key: db-password
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
""",
            "service": """
apiVersion: v1
kind: Service
metadata:
  name: quantum-trading-service
  namespace: quantum-trading
spec:
  selector:
    app: quantum-trading
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
""",
            "hpa": f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantum-trading-hpa
  namespace: quantum-trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantum-trading-app
  minReplicas: {self.scaling.min_replicas}
  maxReplicas: {self.scaling.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {self.scaling.target_cpu_percentage}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {self.scaling.target_memory_percentage}
""",
            "postgres": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: quantum-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: quantum_trading
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: quantum-trading-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: quantum-trading
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
""",
            "redis": """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: quantum-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: quantum-trading
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
"""
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Generate monitoring configuration"""
        
        return {
            "prometheus": """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quantum-trading'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
""",
            "grafana_dashboard": """
{
  "dashboard": {
    "title": "Octopus",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error rate"
          }
        ]
      },
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
            "legendFormat": "CPU %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory MB"
          }
        ]
      }
    ]
  }
}
""",
            "alert_rules": """
groups:
  - name: quantum_trading_alerts
    rules:
      - alert: HighRequestRate
        expr: rate(http_requests_total[5m]) > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High request rate detected
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
      
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage detected
      
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 1.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage detected
"""
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate production configuration"""
        
        errors = []
        
        # Check required secrets
        if not self.security.secret_key:
            errors.append("SECRET_KEY is required")
        
        if not self.security.jwt_secret:
            errors.append("JWT_SECRET is required")
        
        if not self.database.password:
            errors.append("DB_PASSWORD is required")
        
        # Check broker configurations
        brokers_configured = 0
        
        if self.broker_config["alpaca"]["api_key"]:
            brokers_configured += 1
        
        if self.broker_config["td_ameritrade"]["api_key"]:
            brokers_configured += 1
        
        if brokers_configured == 0:
            errors.append("At least one broker must be configured")
        
        # Check scaling configuration
        if self.scaling.min_replicas > self.scaling.max_replicas:
            errors.append("min_replicas cannot be greater than max_replicas")
        
        # Check security
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if not self.security.ssl_redirect:
                errors.append("SSL redirect should be enabled in production")
            
            if len(self.security.cors_origins) == 0:
                errors.append("CORS origins should be configured for production")
        
        return errors

# Global configuration instance
config = ProductionConfig()

def get_config() -> ProductionConfig:
    """Get global configuration instance"""
    return config