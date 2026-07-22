"""
Octopus Trading Platform™ - Production FastAPI Application
Secure, maintainable, single-service architecture
Handles: Market Data, Real-time Processing, ML/AI, WebSockets
"""

# Load environment variables first, before any other imports
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import JSONResponse
import uuid
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.core.initialization import SystemInitializer
from src.core.config import get_settings
from src.core.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware
from src.core.middleware_metrics import MetricsMiddleware
from src.monitoring.metrics import metrics_collector

# Use the comprehensive WebSocket implementation (singleton from module)
from src.realtime.websockets import websocket_manager

from src.api.routes import ml_models
from src.api.routes import portfolios
# Unified Auth API (single implementation via professional_auth)
from src.api.endpoints.professional_auth import router as auth_router

# Unified Market Data API (replaces professional_market_data, market_data_workflow, real_market_data, simple_real_data)
from src.api.endpoints.unified_market_data import router as unified_market_data_router

from src.api.endpoints.comprehensive_api import router as comprehensive_router
from src.api.endpoints.risk import router as risk_router
from src.api.endpoints.llm_simple import llm_router
from src.api.endpoints.macro_data import router as macro_router
from src.api.endpoints.alpha_vantage_mcp import router as alpha_vantage_mcp_router
from src.api.endpoints.onchain_data import router as onchain_router
from src.api.endpoints.social_data import router as social_router
from src.api.endpoints.agents import router as agents_router
from src.api.endpoints.wallet import router as wallet_router
from src.api.endpoints.security import router as security_router
from src.api.endpoints.scenarios import router as scenarios_router
# Unified WebSocket API (replaces websocket_realtime, websocket, realtime endpoints)
from src.api.endpoints.unified_websocket import router as unified_websocket_router, set_websocket_manager

# Trading Bots, Agent Panels, and Backtesting APIs
from src.api.endpoints.trading_bots import router as trading_bots_router
from src.api.endpoints.agent_panels import router as agent_panels_router
from src.api.endpoints.backtesting import router as backtesting_router
from src.api.endpoints.strategies_crud import router as strategies_crud_router
from src.api.endpoints.startup_tracker import router as startup_tracker_router
from src.api.endpoints.allocation_copilot import router as allocation_copilot_router
from src.api.endpoints.search import router as search_router
from src.api.endpoints.payment_zarinpal import router as zarinpal_router
from src.api.endpoints.iran_market import router as iran_market_router
from src.api.endpoints.otp_auth import router as otp_auth_router

# Orphaned routers (previously unregistered)
from src.api.endpoints.fundamental_data import router as fundamental_router
from src.api.endpoints.data_collection import router as data_collection_router
from src.api.endpoints.portfolio_api import portfolio_router as portfolio_api_router
from src.api.endpoints.strategies import router as strategies_router
from src.api.endpoints.agents_v2 import router as agents_v2_router
from src.api.routes.analysis import router as analysis_router
from src.api.routes.trading import router as trading_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Global system initializer
system_initializer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for production deployment"""
    global system_initializer
    
    logger.info("🚀 Starting Octopus Trading Platform...")
    
    # Initialize system components
    system_initializer = SystemInitializer()
    await system_initializer.initialize_performance_components()
    
    # Initialize WebSocket manager for real-time updates
    await websocket_manager.initialize()
    set_websocket_manager(websocket_manager)
    
    # Start background tasks for real-time processing
    asyncio.create_task(system_initializer.start_market_data_streams())
    
    logger.info("✅ Octopus Trading Platform ready for production")
    
    yield
    
    # Cleanup
    await system_initializer.cleanup()
    logger.info("🛑 Octopus Trading Platform shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="🐙 Octopus Trading Platform",
    description="Professional trading platform with AI-powered analytics and real-time market data",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add unified error handling middleware (order matters - metrics first)
app.add_middleware(MetricsMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Add secure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,  # Secure: No wildcards
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# Initialize Prometheus monitoring
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Custom metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for monitoring and load balancers"""
    # Simple health check without blocking operations
    return {
        "status": "healthy",
        "service": "octopus-trading-platform",
        "version": "3.4.0",
        "environment": settings.environment,
    }

@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with component status"""
    try:
        component_status = await system_initializer.get_component_status() if system_initializer else {}
        return {
            "status": "healthy",
            "service": "octopus-trading-platform",
            "version": "3.4.0",
            "environment": settings.environment,
            "components": component_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Try to import LLM router (has heavy ML dependencies - torch, peft)
try:
    from src.api.endpoints.llm import llm_router as llm_full_router
except ImportError as e:
    logger.warning(f"LLM router not available (missing dependency: {e})")
    llm_full_router = None

# API Routes

# Unified Authentication API (professional_auth is the main implementation)
app.include_router(auth_router, tags=["Unified Authentication"])

# Unified Market Data API (consolidates all market data endpoints)
app.include_router(unified_market_data_router, tags=["Unified Market Data"])

# Other APIs
app.include_router(ml_models.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(portfolios.router, prefix="/portfolios", tags=["Portfolio Management"])
app.include_router(comprehensive_router, prefix="/api", tags=["Comprehensive API"])
app.include_router(risk_router, tags=["Skfolio Risk Management"])
app.include_router(llm_router, prefix="/llm", tags=["LLM & AI Analytics"])

# New real data source APIs
app.include_router(macro_router, prefix="/api/macro", tags=["Real Macro Data"])
app.include_router(onchain_router, prefix="/api/onchain", tags=["Real On-Chain Data"])
app.include_router(social_router, prefix="/api/social", tags=["Real Social Data"])
app.include_router(alpha_vantage_mcp_router, tags=["Alpha Vantage MCP"])

# Phase 3: Backend Integration APIs
app.include_router(agents_router, tags=["Agent Monitoring"])
app.include_router(wallet_router, tags=["Wallet & Funding"])
app.include_router(zarinpal_router, tags=["ZarinPal Payment"])
app.include_router(iran_market_router, tags=["Iran Market Data"])
app.include_router(otp_auth_router, tags=["OTP Authentication"])
app.include_router(security_router, tags=["Security & Access Control"])
app.include_router(scenarios_router, tags=["Market Scenarios"])
# Unified WebSocket API (consolidates all WebSocket endpoints)
app.include_router(unified_websocket_router, tags=["Unified WebSocket"])

# Trading Bots, Agent Panels, and Backtesting APIs
app.include_router(trading_bots_router, tags=["Trading Bots"])
app.include_router(agent_panels_router, tags=["Agent Panels"])
app.include_router(backtesting_router, tags=["Backtesting"])
app.include_router(strategies_crud_router, tags=["Strategies CRUD"])
app.include_router(startup_tracker_router, tags=["Startup Tracker"])
app.include_router(allocation_copilot_router, tags=["Asset Allocation Copilot"])
app.include_router(search_router, tags=["Search"])

# Previously orphaned routers
app.include_router(fundamental_router, prefix="/api/fundamental", tags=["Fundamental Data"])
app.include_router(data_collection_router, prefix="/api/data-collection", tags=["Data Collection"])
app.include_router(portfolio_api_router, prefix="/api/portfolio", tags=["Portfolio API"])
app.include_router(strategies_router, prefix="/api/strategies", tags=["Strategies"])
app.include_router(agents_v2_router, tags=["Agent Monitoring V2"])
if llm_full_router is not None:
    app.include_router(llm_full_router, prefix="/api/llm", tags=["LLM & AI Analytics"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["Market Analysis"])
app.include_router(trading_router, prefix="/api/trading", tags=["Trading Operations"])

# Main WebSocket endpoint for real-time data streaming
# This uses the unified websocket manager and is compatible with unified_websocket_router
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time market data"""
    client_id = f"ws_{uuid.uuid4()}"
    try:
        await websocket_manager.connect(websocket, client_id)
        logger.info(f"Main WebSocket client connected: {client_id}")
        
        while True:
            data = await websocket.receive_text()
            await websocket_manager.handle_message(client_id, data)
    except WebSocketDisconnect:
        logger.info(f"Main WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if client_id:
            await websocket_manager.disconnect(client_id)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    return {
        "message": "🐙 Octopus Trading Platform - Production Ready",
        "version": "3.4.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws",
        "features": {
            "real_time_data": "✅",
            "ai_analytics": "✅", 
            "risk_management": "✅",
            "portfolio_optimization": "✅",
            "backtesting": "✅",
            "secure_authentication": "✅"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    if isinstance(exc.detail, dict):
        # Endpoints/dependencies that already build a structured error body
        # (e.g. rate limiting, account lockout) should surface it as-is
        # instead of being nested under a generic "http_error" wrapper.
        content = {**exc.detail, "request_id": getattr(request.state, "request_id", None)}
    else:
        content = {
            "error": "http_error",
            "message": exc.detail,
            "request_id": getattr(request.state, "request_id", None)
        }
    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred" if not settings.debug else str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main_refactored:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production: No reload
        workers=1,     # Single worker for development
        log_level="info"
    ) 