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

from fastapi import FastAPI, HTTPException, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.core.initialization import SystemInitializer
from src.core.config import get_settings
from src.core.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from src.core.exceptions import error_handler

# Use the comprehensive WebSocket implementation
from src.realtime.websockets import WebSocketManager

# Create global websocket manager instance
websocket_manager = WebSocketManager()

from src.api.routes import market_data
from src.api.routes import realtime
from src.api.routes import ml_models
from src.api.routes import websocket
from src.api.routes import portfolios
from src.api.endpoints.professional_auth import router as auth_router
from src.api.endpoints.professional_market_data import router as market_router
from src.api.endpoints.comprehensive_api import router as comprehensive_router
from src.api.endpoints.risk import router as risk_router
from src.api.endpoints.llm_simple import llm_router
from src.api.endpoints.simple_real_data import simple_data_router
from src.api.endpoints.macro_data import router as macro_router
from src.api.endpoints.onchain_data import router as onchain_router
from src.api.endpoints.social_data import router as social_router
from src.api.endpoints.agents import router as agents_router
from src.api.endpoints.wallet import router as wallet_router
from src.api.endpoints.security import router as security_router
from src.api.endpoints.scenarios import router as scenarios_router
from src.api.endpoints.websocket_realtime import router as ws_realtime_router, set_websocket_manager
from src.api.endpoints.market_data_workflow import router as market_data_workflow_router

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

# Add unified error handling middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)

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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for monitoring and load balancers"""
    try:
        component_status = await system_initializer.get_component_status() if system_initializer else {}
        return {
            "status": "healthy",
            "service": "octopus-trading-platform",
            "version": "3.0.0",
            "environment": settings.environment,
            "components": component_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# API Routes
app.include_router(auth_router, tags=["Professional Authentication"])
app.include_router(market_router, tags=["Professional Market Data"])
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["Market Data"])
app.include_router(realtime.router, prefix="/api/v1/realtime", tags=["Real-time Data"])
app.include_router(ml_models.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(websocket.router, prefix="/api/v1/websocket", tags=["WebSocket"])
app.include_router(portfolios.router, prefix="/portfolios", tags=["Portfolio Management"])
app.include_router(comprehensive_router, prefix="/api", tags=["Comprehensive API"])
app.include_router(risk_router, tags=["Skfolio Risk Management"])
app.include_router(llm_router, prefix="/llm", tags=["LLM & AI Analytics"])
app.include_router(simple_data_router, prefix="/api/simple", tags=["Real Market Data"])

# New real data source APIs
app.include_router(macro_router, prefix="/api/macro", tags=["Real Macro Data"])
app.include_router(onchain_router, prefix="/api/onchain", tags=["Real On-Chain Data"])
app.include_router(social_router, prefix="/api/social", tags=["Real Social Data"])

# Phase 3: Backend Integration APIs
app.include_router(agents_router, tags=["Agent Monitoring"])
app.include_router(wallet_router, tags=["Wallet & Funding"])
app.include_router(security_router, tags=["Security & Access Control"])
app.include_router(scenarios_router, tags=["Market Scenarios"])
app.include_router(ws_realtime_router, tags=["WebSocket Real-time"])

# Market Data Workflow API
app.include_router(market_data_workflow_router, tags=["Market Data Workflow"])

# WebSocket endpoint for real-time data streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Handle incoming messages from client
            data = await websocket.receive_text()
            # Process subscription requests, etc.
            await websocket_manager.handle_message(websocket, data)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with platform information"""
    return {
        "message": "🐙 Octopus Trading Platform - Production Ready",
        "version": "3.0.0",
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
app.add_exception_handler(HTTPException, error_handler)

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