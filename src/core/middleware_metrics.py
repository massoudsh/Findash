"""
FastAPI Middleware for Prometheus Metrics Collection
Automatically tracks HTTP requests, response times, and errors
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.monitoring.metrics import metrics_collector

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics for Prometheus"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        # Skip metrics endpoint to avoid recursion
        if endpoint == "/metrics":
            return await call_next(request)
        
        # Get request size if available
        request_size = None
        if hasattr(request, '_body'):
            try:
                body = await request.body()
                request_size = len(body) if body else 0
            except Exception:
                pass
        
        status_code = 200
        response_size = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Get response size if available
            if hasattr(response, 'body'):
                try:
                    response_size = len(response.body) if response.body else 0
                except Exception:
                    pass
            
            duration = time.time() - start_time
            
            # Record metrics
            metrics_collector.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )
            
            return response
            
        except Exception as e:
            status_code = 500
            duration = time.time() - start_time
            
            # Record error metrics
            metrics_collector.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )
            
            # Re-raise the exception
            raise

