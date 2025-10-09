"""
Configuration management for Octopus Trading Platform™
Secure, production-ready configuration with proper environment variable support
"""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
from dataclasses import dataclass, field


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    url: str = Field("postgresql://postgres:postgres@localhost:5432/trading_db", env="DATABASE_URL")
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    name: str = Field("trading_db", env="DB_NAME")
    user: str = Field("postgres", env="DB_USER")
    password: str = Field("postgres", env="DB_PASSWORD")
    pool_size: int = Field(20, env="DB_POOL_SIZE")
    max_overflow: int = Field(30, env="DB_MAX_OVERFLOW")
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")
    
    class Config:
        env_prefix = "REDIS_"


class CelerySettings(BaseSettings):
    """Celery configuration settings"""
    broker_url: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    result_backend: str = Field("redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    accept_content: str = Field("json", env="CELERY_ACCEPT_CONTENT")  # Changed from List[str] to str
    result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    timezone: str = Field("UTC", env="CELERY_TIMEZONE")
    enable_utc: bool = Field(True, env="CELERY_ENABLE_UTC")
    worker_prefetch_multiplier: int = Field(1, env="CELERY_WORKER_PREFETCH_MULTIPLIER")
    task_acks_late: bool = Field(True, env="CELERY_TASK_ACKS_LATE")
    worker_max_tasks_per_child: int = Field(1000, env="CELERY_WORKER_MAX_TASKS_PER_CHILD")
    
    @property
    def accept_content_list(self) -> List[str]:
        """Convert accept_content string to list for Celery"""
        return [content.strip() for content in self.accept_content.split(",")]
    
    class Config:
        env_prefix = "CELERY_"


class AuthSettings(BaseSettings):
    """Authentication and security settings - Production secure defaults"""
    secret_key: str = Field(..., env="SECRET_KEY")  # Required field
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")  # Required field
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(60, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    bcrypt_rounds: int = Field(12, env="BCRYPT_ROUNDS")
    
    @validator("secret_key", "jwt_secret_key")
    def validate_secret_keys(cls, v):
        if len(v) < 32:
            raise ValueError("Secret keys must be at least 32 characters for security")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


class APISettings(BaseSettings):
    """API server configuration"""
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(4, env="API_WORKERS")
    reload: bool = Field(True, env="API_RELOAD")
    
    class Config:
        env_prefix = "API_"


class ExternalAPISettings(BaseSettings):
    """External API keys and endpoints"""
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    yahoo_finance_api_key: Optional[str] = Field(None, env="YAHOO_FINANCE_API_KEY")
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")


class EmailSettings(BaseSettings):
    """Email configuration"""
    smtp_host: str = Field("smtp.gmail.com", env="SMTP_HOST")
    smtp_port: int = Field(587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    smtp_tls: bool = Field(True, env="SMTP_TLS")
    smtp_ssl: bool = Field(False, env="SMTP_SSL")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings"""
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(3001, env="GRAFANA_PORT")
    grafana_admin_password: str = Field("admin123", env="GRAFANA_ADMIN_PASSWORD")
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")


class FileStorageSettings(BaseSettings):
    """File storage configuration"""
    data_dir: str = Field("./data", env="DATA_DIR")
    logs_dir: str = Field("./logs", env="LOGS_DIR")
    models_dir: str = Field("./models", env="MODELS_DIR")
    upload_dir: str = Field("./uploads", env="UPLOAD_DIR")
    
    def __post_init__(self):
        # Create directories if they don't exist
        for directory in [self.data_dir, self.logs_dir, self.models_dir, self.upload_dir]:
            os.makedirs(directory, exist_ok=True)


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration"""
    per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    burst: int = Field(20, env="RATE_LIMIT_BURST")


class CORSSettings(BaseSettings):
    """CORS configuration"""
    origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8000"], 
        env="CORS_ORIGINS"
    )
    allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    
    @validator("origins", pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class TradingSettings(BaseSettings):
    """Trading configuration"""
    default_portfolio_value: float = Field(100000.0, env="DEFAULT_PORTFOLIO_VALUE")
    max_position_size: float = Field(0.1, env="MAX_POSITION_SIZE")
    risk_free_rate: float = Field(0.05, env="RISK_FREE_RATE")
    default_volatility: float = Field(0.2, env="DEFAULT_VOLATILITY")
    
    @validator("max_position_size")
    def validate_max_position_size(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        return v


class TestSettings(BaseSettings):
    """Testing configuration"""
    database_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/test_trading_db",
        env="TEST_DATABASE_URL"
    )
    pytest_workers: int = Field(4, env="PYTEST_WORKERS")
    coverage_threshold: int = Field(80, env="COVERAGE_THRESHOLD")


class Settings(BaseSettings):
    """Main application settings"""
    
    # Core settings
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Nested settings as fields
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    api: APISettings = Field(default_factory=APISettings)
    external_apis: ExternalAPISettings = Field(default_factory=ExternalAPISettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    file_storage: FileStorageSettings = Field(default_factory=FileStorageSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    testing: TestSettings = Field(default_factory=TestSettings)
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_environments = ["development", "testing", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance - moved to avoid import-time issues
# Use get_settings() function instead of this global variable
# settings = get_settings() 