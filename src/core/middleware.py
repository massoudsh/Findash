"""
Middleware components for Quantum Trading Matrix™
Provides logging, error handling, rate limiting, and security middleware
"""

import time
import uuid
import json
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import structlog
from src.core.config import get_settings
from src.core.security import rate_limiter

settings = get_settings()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, request_id, e, process_time)
            raise
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        await self._log_response(request, response, request_id, process_time)
        
        return response
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request details
        logger.info(
            "incoming_request",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
            headers=dict(request.headers) if settings.debug else None
        )
    
    async def _log_response(self, request: Request, response: Response, request_id: str, process_time: float):
        """Log outgoing response"""
        logger.info(
            "outgoing_response",
            request_id=request_id,
            status_code=response.status_code,
            process_time=process_time,
            method=request.method,
            path=request.url.path
        )
    
    async def _log_error(self, request: Request, request_id: str, error: Exception, process_time: float):
        """Log request error"""
        logger.error(
            "request_error",
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            process_time=process_time,
            method=request.method,
            path=request.url.path,
            exc_info=True
        )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except ValueError as e:
            # Handle validation errors
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        except PermissionError as e:
            # Handle permission errors
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "permission_denied",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        except ConnectionError as e:
            # Handle connection errors (database, external APIs)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "service_unavailable",
                    "message": "External service temporarily unavailable",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error_message=str(e),
                request_id=getattr(request.state, "request_id", None),
                exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred" if not settings.debug else str(e),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting"""
    
    def __init__(self, app, calls_per_minute: int = None):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute or settings.rate_limit.per_minute
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Create identifier for rate limiting
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        identifier = f"{client_ip}:{user_agent}"
        
        # Check rate limit
        if not rate_limiter.is_allowed(identifier):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {self.calls_per_minute} requests per minute.",
                    "request_id": getattr(request.state, "request_id", None)
                },
                headers={
                    "Retry-After": "60"
                }
            )
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for cache control headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Set cache control based on endpoint
        if request.url.path.startswith("/api/market-data"):
            # Market data can be cached briefly
            response.headers["Cache-Control"] = "public, max-age=60"
        elif request.url.path.startswith("/api/auth"):
            # Auth endpoints should not be cached
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            # API docs can be cached
            response.headers["Cache-Control"] = "public, max-age=3600"
        else:
            # Default: no cache for dynamic content
            response.headers["Cache-Control"] = "no-cache"
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for limiting request size"""
    
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "request_too_large",
                        "message": f"Request size {content_length} bytes exceeds limit of {self.max_size} bytes",
                        "request_id": getattr(request.state, "request_id", None)
                    }
                )
        
        return await call_next(request)


# Middleware factory functions
def create_request_logging_middleware():
    """Create request logging middleware"""
    return RequestLoggingMiddleware


def create_error_handling_middleware():
    """Create error handling middleware"""
    return ErrorHandlingMiddleware


def create_rate_limiting_middleware(calls_per_minute: int = None):
    """Create rate limiting middleware"""
    def middleware_factory(app):
        return RateLimitingMiddleware(app, calls_per_minute)
    return middleware_factory


def create_security_headers_middleware():
    """Create security headers middleware"""
    return SecurityHeadersMiddleware


def create_cache_control_middleware():
    """Create cache control middleware"""
    return CacheControlMiddleware


def create_request_size_limit_middleware(max_size: int = 10 * 1024 * 1024):
    """Create request size limit middleware"""
    def middleware_factory(app):
        return RequestSizeLimitMiddleware(app, max_size)
    return middleware_factory 