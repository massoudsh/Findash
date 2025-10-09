"""
API Routes Module
Organizes all API endpoints into logical, maintainable routers.
"""

# Import all routers for easy access
from .market_data import router as market_data_router
from .realtime import router as realtime_router
from .ml_models import router as ml_models_router
from .websocket import router as websocket_router
from .portfolios import router as portfolios_router
from .trading import router as trading_router
from .analysis import router as analysis_router

__all__ = [
    "market_data_router",
    "realtime_router",
    "ml_models_router", 
    "websocket_router",
    "portfolios_router",
    "trading_router",
    "analysis_router"
] 