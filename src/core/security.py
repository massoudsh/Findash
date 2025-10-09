"""
Octopus Trading Platform™ - Security Module
Production-grade security with authentication, authorization, and protection mechanisms
"""

import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
import logging

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
import redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Redis for rate limiting and session management
try:
    redis_client = redis.Redis.from_url(settings.redis.url, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# ==============================================================================
# PASSWORD & HASHING UTILITIES
# ==============================================================================

def hash_password(password: str) -> str:
    """Securely hash a password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    return secrets.token_urlsafe(length)

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()

# ==============================================================================
# JWT TOKEN MANAGEMENT
# ==============================================================================

class TokenData(BaseModel):
    """JWT token payload data"""
    user_id: str
    email: str
    roles: List[str] = []
    permissions: List[str] = []

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.auth.jwt_access_token_expire_minutes)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    return jwt.encode(to_encode, settings.auth.jwt_secret_key, algorithm=settings.auth.jwt_algorithm)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.auth.jwt_refresh_token_expire_days)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    return jwt.encode(to_encode, settings.auth.jwt_secret_key, algorithm=settings.auth.jwt_algorithm)

def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.auth.jwt_secret_key, algorithms=[settings.auth.jwt_algorithm])
        
        # Validate token type
        token_type = payload.get("type")
        if token_type != "access":
            return None
            
        # Extract user data
        user_id = payload.get("sub")
        email = payload.get("email")
        roles = payload.get("roles", [])
        permissions = payload.get("permissions", [])
        
        if user_id is None or email is None:
            return None
            
        return TokenData(
            user_id=user_id,
            email=email,
            roles=roles,
            permissions=permissions
        )
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

# ==============================================================================
# AUTHENTICATION DEPENDENCIES
# ==============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Dependency to get current authenticated user"""
    
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception
            
        return token_data
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

async def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency to ensure user is active"""
    # Add additional checks here (user status, account expiry, etc.)
    return current_user

def require_permissions(required_permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs if available
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=403,
                    detail="Permission check failed: No user context"
                )
            
            # Check permissions
            user_permissions = set(current_user.permissions)
            required_perms = set(required_permissions)
            
            if not required_perms.issubset(user_permissions):
                missing_perms = required_perms - user_permissions
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Missing: {list(missing_perms)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ==============================================================================
# RATE LIMITING
# ==============================================================================

class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is allowed under rate limit"""
        if not self.redis:
            return True  # Allow if Redis is not available
            
        try:
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - window
            
            # Remove expired entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_count = self.redis.zcard(key)
            
            if current_count >= limit:
                return False
                
            # Add current request
            self.redis.zadd(key, {str(current_time): current_time})
            self.redis.expire(key, window)
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow on error
            
    async def get_remaining(self, key: str, limit: int, window: int = 60) -> int:
        """Get remaining requests for key"""
        if not self.redis:
            return limit
            
        try:
            current_time = int(datetime.utcnow().timestamp())
            window_start = current_time - window
            
            self.redis.zremrangebyscore(key, 0, window_start)
            current_count = self.redis.zcard(key)
            
            return max(0, limit - current_count)
            
        except Exception:
            return limit

# Global rate limiter instance
rate_limiter = RateLimiter(redis_client)

# ==============================================================================
# SECURITY MIDDLEWARE
# ==============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
        # CSP header
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 100):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
            
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
            
        rate_limit_key = f"rate_limit:{client_ip}"
        
        # Check rate limit
        if not await rate_limiter.is_allowed(rate_limit_key, self.calls_per_minute, 60):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded: {self.calls_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
            
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await rate_limiter.get_remaining(rate_limit_key, self.calls_per_minute, 60)
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(datetime.utcnow().timestamp()) + 60)
        
        return response

# ==============================================================================
# API KEY AUTHENTICATION
# ==============================================================================

class APIKeyManager:
    """Manage API keys for service authentication"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        
    def generate_api_key(self, user_id: str, name: str = "") -> str:
        """Generate a new API key"""
        api_key = f"otf_{generate_secure_token(32)}"  # otf = octopus trading platform
        
        # Store API key metadata
        if self.redis:
            key_data = {
                "user_id": user_id,
                "name": name,
                "created_at": datetime.utcnow().isoformat(),
                "last_used": "",
                "active": "true"
            }
            self.redis.hset(f"api_key:{hash_api_key(api_key)}", mapping=key_data)
            
        return api_key
        
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return metadata"""
        if not self.redis:
            return None
            
        try:
            key_hash = hash_api_key(api_key)
            key_data = self.redis.hgetall(f"api_key:{key_hash}")
            
            if not key_data or key_data.get("active") != "true":
                return None
                
            # Update last used timestamp
            self.redis.hset(f"api_key:{key_hash}", "last_used", datetime.utcnow().isoformat())
            
            return key_data
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
            
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if not self.redis:
            return False
            
        try:
            key_hash = hash_api_key(api_key)
            return self.redis.hset(f"api_key:{key_hash}", "active", "false") is not None
        except Exception:
            return False

# Global API key manager
api_key_manager = APIKeyManager(redis_client)

# ==============================================================================
# SECURITY UTILITIES
# ==============================================================================

def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks"""
    return hmac.compare_digest(a, b)

def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input"""
    if not input_str:
        return ""
        
    # Remove control characters and limit length
    sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\n\r\t')
    return sanitized[:max_length]

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    import ipaddress
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

# ==============================================================================
# AUDIT LOGGING
# ==============================================================================

class SecurityAuditLogger:
    """Log security events for monitoring and compliance"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.logger = logging.getLogger("security_audit")
        
    def log_auth_event(self, event_type: str, user_id: Optional[str], ip_address: str, 
                      details: Optional[Dict[str, Any]] = None):
        """Log authentication events"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {}
        }
        
        self.logger.info(f"Security event: {event}")
        
        # Store in Redis for real-time monitoring
        if self.redis:
            try:
                self.redis.lpush("security_events", str(event))
                self.redis.ltrim("security_events", 0, 9999)  # Keep last 10k events
            except Exception as e:
                self.logger.error(f"Failed to store security event: {e}")

# Global security audit logger
security_audit = SecurityAuditLogger(redis_client)

def rate_limit_dependency(request: Request):
    """FastAPI dependency for per-IP rate limiting (100 req/min)"""
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_ip}"
    allowed = rate_limiter.is_allowed(key, 100, 60)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    return True 