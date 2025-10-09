"""
Unified Security Configuration for Hybrid Architecture
Provides shared JWT settings and security constants for both FastAPI and Django services
"""

import os
from datetime import timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class SecurityLevel(str, Enum):
    """Security level configurations"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class JWTConfiguration:
    """Unified JWT configuration for both services"""
    
    # Secret keys - MUST be same for both services
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"))
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-change-in-production"))
    
    # Algorithm - standardized across services  
    algorithm: str = "HS256"
    
    # Token expiration - UNIFIED TIMES
    access_token_expire_minutes: int = 60  # Standardized to 60 minutes
    refresh_token_expire_days: int = 7
    api_key_expire_days: int = 365
    
    # Token types
    access_token_type: str = "access"
    refresh_token_type: str = "refresh"
    
    # Security settings
    bcrypt_rounds: int = 12
    require_https: bool = False  # Set to True in production
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 200
    
    def get_access_token_expire_timedelta(self) -> timedelta:
        """Get access token expiration as timedelta for Django"""
        return timedelta(minutes=self.access_token_expire_minutes)
    
    def get_refresh_token_expire_timedelta(self) -> timedelta:
        """Get refresh token expiration as timedelta for Django"""
        return timedelta(days=self.refresh_token_expire_days)
    
    def validate_secrets(self) -> List[str]:
        """Validate security configuration"""
        errors = []
        
        if len(self.secret_key) < 32:
            errors.append("SECRET_KEY must be at least 32 characters")
        
        if len(self.jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")
        
        if self.secret_key == "dev-secret-key-change-in-production":
            errors.append("SECRET_KEY must be changed from default in production")
        
        if self.jwt_secret_key == "dev-jwt-secret-change-in-production":
            errors.append("JWT_SECRET_KEY must be changed from default in production")
        
        return errors
    
    def to_fastapi_config(self) -> Dict[str, Any]:
        """Export configuration for FastAPI service"""
        return {
            "secret_key": self.secret_key,
            "jwt_secret_key": self.jwt_secret_key,
            "jwt_algorithm": self.algorithm,
            "jwt_access_token_expire_minutes": self.access_token_expire_minutes,
            "jwt_refresh_token_expire_days": self.refresh_token_expire_days,
            "bcrypt_rounds": self.bcrypt_rounds
        }
    
    def to_django_config(self) -> Dict[str, Any]:
        """Export configuration for Django service"""
        return {
            "SECRET_KEY": self.secret_key,
            "JWT_ALGORITHM": self.algorithm,
            "JWT_ACCESS_TOKEN_LIFETIME": self.get_access_token_expire_timedelta(),
            "JWT_REFRESH_TOKEN_LIFETIME": self.get_refresh_token_expire_timedelta(),
            "JWT_ROTATE_REFRESH_TOKENS": True,
            "JWT_BLACKLIST_AFTER_ROTATION": True,
            "JWT_AUTH_HEADER_TYPES": ("Bearer",),
            "JWT_SECRET_KEY": self.jwt_secret_key
        }


@dataclass  
class CORSConfiguration:
    """Unified CORS configuration"""
    
    allowed_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",  # Next.js frontend
        "http://127.0.0.1:3000",
        "http://localhost:8080",  # NGINX gateway
        "http://127.0.0.1:8080"
    ])
    
    allowed_methods: List[str] = field(default_factory=lambda: [
        "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"
    ])
    
    allowed_headers: List[str] = field(default_factory=lambda: [
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "X-Requested-With", "X-CSRFToken"
    ])
    
    allow_credentials: bool = True
    max_age: int = 86400  # 24 hours
    
    def to_fastapi_config(self) -> Dict[str, Any]:
        """Export CORS config for FastAPI"""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allowed_methods,
            "allow_headers": self.allowed_headers,
            "max_age": self.max_age
        }
    
    def to_django_config(self) -> Dict[str, Any]:
        """Export CORS config for Django"""
        return {
            "CORS_ALLOWED_ORIGINS": self.allowed_origins,
            "CORS_ALLOW_CREDENTIALS": self.allow_credentials,
            "CORS_ALLOWED_HEADERS": self.allowed_headers,
            "CORS_ALLOW_ALL_ORIGINS": False
        }


@dataclass
class SecurityMiddlewareConfig:
    """Security middleware configuration"""
    
    # HTTPS settings
    force_https: bool = False
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    
    # Content security
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    
    # Session security
    session_cookie_secure: bool = False  # Set to True in production
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "Lax"
    csrf_cookie_secure: bool = False  # Set to True in production
    csrf_cookie_httponly: bool = True
    csrf_cookie_samesite: str = "Lax"
    
    def to_django_middleware_config(self) -> Dict[str, Any]:
        """Export security middleware config for Django"""
        return {
            "SECURE_HSTS_SECONDS": self.hsts_max_age if self.force_https else 0,
            "SECURE_HSTS_INCLUDE_SUBDOMAINS": self.hsts_include_subdomains,
            "SECURE_SSL_REDIRECT": self.force_https,
            "X_FRAME_OPTIONS": self.x_frame_options,
            "SESSION_COOKIE_SECURE": self.session_cookie_secure,
            "SESSION_COOKIE_HTTPONLY": self.session_cookie_httponly,
            "SESSION_COOKIE_SAMESITE": self.session_cookie_samesite,
            "CSRF_COOKIE_SECURE": self.csrf_cookie_secure,
            "CSRF_COOKIE_HTTPONLY": self.csrf_cookie_httponly,
            "CSRF_COOKIE_SAMESITE": self.csrf_cookie_samesite
        }


class UnifiedSecurityConfig:
    """Unified security configuration manager"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.DEVELOPMENT):
        self.security_level = security_level
        self.jwt = JWTConfiguration()
        self.cors = CORSConfiguration()
        self.middleware = SecurityMiddlewareConfig()
        
        # Apply security level specific settings
        self._apply_security_level_settings()
    
    def _apply_security_level_settings(self):
        """Apply security settings based on environment"""
        if self.security_level == SecurityLevel.PRODUCTION:
            # Production security hardening
            self.jwt.require_https = True
            self.jwt.bcrypt_rounds = 14
            self.jwt.rate_limit_requests_per_minute = 60  # Stricter rate limiting
            
            self.middleware.force_https = True
            self.middleware.session_cookie_secure = True
            self.middleware.csrf_cookie_secure = True
            
            # Restrict CORS in production
            self.cors.allowed_origins = [
                os.getenv("FRONTEND_URL", "https://trading.yourcompany.com"),
                os.getenv("API_GATEWAY_URL", "https://api.yourcompany.com")
            ]
            
        elif self.security_level == SecurityLevel.STAGING:
            # Staging security settings
            self.jwt.require_https = True
            self.jwt.bcrypt_rounds = 13
            
            self.middleware.force_https = True
            self.middleware.session_cookie_secure = True
            self.middleware.csrf_cookie_secure = True
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate entire security configuration"""
        validation_results = {
            "jwt_errors": self.jwt.validate_secrets(),
            "warnings": [],
            "recommendations": []
        }
        
        # Additional validations
        if self.security_level == SecurityLevel.PRODUCTION:
            if not self.jwt.require_https:
                validation_results["warnings"].append("HTTPS should be required in production")
            
            if not self.middleware.force_https:
                validation_results["warnings"].append("HTTPS should be enforced in production")
        
        # Recommendations
        if self.jwt.access_token_expire_minutes > 120:
            validation_results["recommendations"].append("Consider shorter access token expiration")
        
        if len(self.cors.allowed_origins) > 10:
            validation_results["recommendations"].append("Consider restricting CORS origins")
        
        return validation_results
    
    def export_fastapi_config(self) -> Dict[str, Any]:
        """Export complete configuration for FastAPI service"""
        return {
            "auth": self.jwt.to_fastapi_config(),
            "cors": self.cors.to_fastapi_config(),
            "security_level": self.security_level.value
        }
    
    def export_django_config(self) -> Dict[str, Any]:
        """Export complete configuration for Django service"""
        config = {}
        config.update(self.jwt.to_django_config())
        config.update(self.cors.to_django_config())
        config.update(self.middleware.to_django_middleware_config())
        config["SECURITY_LEVEL"] = self.security_level.value
        return config


# Global configuration instances
def get_security_config(security_level: str = None) -> UnifiedSecurityConfig:
    """Get unified security configuration"""
    if security_level:
        level = SecurityLevel(security_level)
    else:
        level = SecurityLevel(os.getenv("SECURITY_LEVEL", "development"))
    
    return UnifiedSecurityConfig(level)


# Export for easy imports
__all__ = [
    "UnifiedSecurityConfig",
    "JWTConfiguration", 
    "CORSConfiguration",
    "SecurityMiddlewareConfig",
    "SecurityLevel",
    "get_security_config"
] 