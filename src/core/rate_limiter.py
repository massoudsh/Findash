"""
Free Redis-based Rate Limiter
Protects APIs from abuse without using paid services
"""

import redis
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
import json

logger = logging.getLogger(__name__)

class FreeRateLimiter:
    """Simple Redis-based rate limiter using free tier"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for rate limiting: {e}")
            self.redis_client = None
    
    def check_rate_limit(self, 
                        identifier: str, 
                        limit: int = 30, 
                        window_seconds: int = 60,
                        burst_limit: int = 10) -> Dict[str, Any]:
        """
        Check if request is within rate limits
        
        Args:
            identifier: Unique identifier (IP, user_id, etc.)
            limit: Requests per window
            window_seconds: Time window in seconds
            burst_limit: Maximum burst requests
            
        Returns:
            Dict with rate limit info
        """
        
        if not self.redis_client:
            # If Redis is not available, allow all requests
            return {
                "allowed": True,
                "remaining": limit,
                "reset_time": int(time.time()) + window_seconds,
                "retry_after": None
            }
        
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        # Create Redis keys
        requests_key = f"rate_limit:{identifier}:requests"
        burst_key = f"rate_limit:{identifier}:burst"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old requests outside the window
            pipe.zremrangebyscore(requests_key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(requests_key)
            
            # Get burst count
            pipe.get(burst_key)
            
            # Execute pipeline
            results = pipe.execute()
            current_requests = results[1]
            burst_count = int(results[2] or 0)
            
            # Check burst limit first (short-term protection)
            if burst_count >= burst_limit:
                burst_ttl = self.redis_client.ttl(burst_key)
                if burst_ttl == -1:  # No TTL set
                    self.redis_client.expire(burst_key, 10)  # 10 seconds
                    burst_ttl = 10
                
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": current_time + burst_ttl,
                    "retry_after": burst_ttl,
                    "reason": "burst_limit_exceeded"
                }
            
            # Check regular rate limit
            if current_requests >= limit:
                # Find the oldest request to determine when limit resets
                oldest_request = self.redis_client.zrange(requests_key, 0, 0, withscores=True)
                if oldest_request:
                    reset_time = int(oldest_request[0][1]) + window_seconds
                    retry_after = max(1, reset_time - current_time)
                else:
                    reset_time = current_time + window_seconds
                    retry_after = window_seconds
                
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": reset_time,
                    "retry_after": retry_after,
                    "reason": "rate_limit_exceeded"
                }
            
            # Allow the request - record it
            pipe = self.redis_client.pipeline()
            
            # Add current request to sorted set
            pipe.zadd(requests_key, {str(current_time): current_time})
            
            # Increment burst counter
            pipe.incr(burst_key)
            pipe.expire(burst_key, 10)  # 10-second burst window
            
            # Set TTL for requests key
            pipe.expire(requests_key, window_seconds)
            
            pipe.execute()
            
            remaining = max(0, limit - current_requests - 1)
            
            return {
                "allowed": True,
                "remaining": remaining,
                "reset_time": current_time + window_seconds,
                "retry_after": None
            }
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # If Redis fails, allow the request
            return {
                "allowed": True,
                "remaining": limit,
                "reset_time": current_time + window_seconds,
                "retry_after": None
            }
    
    def check_login_attempts(self, identifier: str, max_attempts: int = 5) -> Dict[str, Any]:
        """Check failed login attempts with exponential backoff"""
        
        if not self.redis_client:
            return {"allowed": True, "attempts": 0, "lockout_time": None}
        
        attempts_key = f"login_attempts:{identifier}"
        lockout_key = f"login_lockout:{identifier}"
        
        try:
            # Check if currently locked out
            lockout_until = self.redis_client.get(lockout_key)
            if lockout_until:
                lockout_time = int(lockout_until)
                current_time = int(time.time())
                
                if current_time < lockout_time:
                    return {
                        "allowed": False,
                        "attempts": max_attempts,
                        "lockout_time": lockout_time,
                        "retry_after": lockout_time - current_time
                    }
                else:
                    # Lockout expired, clear it
                    self.redis_client.delete(lockout_key, attempts_key)
            
            # Get current attempts
            attempts = int(self.redis_client.get(attempts_key) or 0)
            
            return {
                "allowed": attempts < max_attempts,
                "attempts": attempts,
                "lockout_time": None
            }
            
        except Exception as e:
            logger.error(f"Login attempts check error: {e}")
            return {"allowed": True, "attempts": 0, "lockout_time": None}
    
    def record_failed_login(self, identifier: str, max_attempts: int = 5):
        """Record a failed login attempt"""
        
        if not self.redis_client:
            return
        
        attempts_key = f"login_attempts:{identifier}"
        lockout_key = f"login_lockout:{identifier}"
        
        try:
            # Increment attempts
            attempts = self.redis_client.incr(attempts_key)
            
            # Set TTL for attempts (reset after 1 hour)
            self.redis_client.expire(attempts_key, 3600)
            
            # If max attempts reached, create lockout
            if attempts >= max_attempts:
                # Exponential backoff: 2^(attempts-max_attempts) * 5 minutes
                backoff_minutes = min(2 ** (attempts - max_attempts) * 5, 60)  # Max 1 hour
                lockout_until = int(time.time()) + (backoff_minutes * 60)
                
                self.redis_client.set(lockout_key, lockout_until, ex=backoff_minutes * 60)
                
                logger.warning(f"Account locked for {backoff_minutes} minutes: {identifier}")
                
        except Exception as e:
            logger.error(f"Failed login recording error: {e}")
    
    def clear_login_attempts(self, identifier: str):
        """Clear login attempts after successful login"""
        
        if not self.redis_client:
            return
        
        attempts_key = f"login_attempts:{identifier}"
        lockout_key = f"login_lockout:{identifier}"
        
        try:
            self.redis_client.delete(attempts_key, lockout_key)
        except Exception as e:
            logger.error(f"Clear login attempts error: {e}")

# Global rate limiter instance
rate_limiter = FreeRateLimiter()

def get_client_identifier(request: Request) -> str:
    """Get unique identifier for rate limiting"""
    
    # Try to get user ID if authenticated
    if hasattr(request.state, 'user') and request.state.user:
        return f"user:{request.state.user.get('id', 'unknown')}"
    
    # Use IP address as fallback
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in case of multiple proxies
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    
    return f"ip:{ip}"

def create_rate_limit_dependency(requests_per_minute: int = 30, burst_limit: int = 10):
    """Create a FastAPI dependency for rate limiting"""
    
    async def rate_limit_check(request: Request):
        identifier = get_client_identifier(request)
        
        # Check rate limit
        result = rate_limiter.check_rate_limit(
            identifier=identifier,
            limit=requests_per_minute,
            window_seconds=60,
            burst_limit=burst_limit
        )
        
        if not result["allowed"]:
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(requests_per_minute),
                "X-RateLimit-Remaining": str(result["remaining"]),
                "X-RateLimit-Reset": str(result["reset_time"]),
            }
            
            if result.get("retry_after"):
                headers["Retry-After"] = str(result["retry_after"])
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Try again in {result.get('retry_after', 60)} seconds.",
                    "retry_after": result.get("retry_after"),
                    "reason": result.get("reason", "rate_limit_exceeded")
                },
                headers=headers
            )
        
        return True
    
    return rate_limit_check

# Standard rate limit dependencies
standard_rate_limit = create_rate_limit_dependency(30, 10)  # 30/min, 10 burst
strict_rate_limit = create_rate_limit_dependency(10, 5)    # 10/min, 5 burst
auth_rate_limit = create_rate_limit_dependency(5, 2)       # 5/min, 2 burst for auth endpoints 