"""
API Routes Module
Organizes all API endpoints into logical, maintainable routers.
"""

# Import all routers for easy access
from .ml_models import router as ml_models_router
from .portfolios import router as portfolios_router
from .trading import router as trading_router
from .analysis import router as analysis_router

__all__ = [
    "ml_models_router", 
    "portfolios_router",
    "trading_router",
    "analysis_router"
] 