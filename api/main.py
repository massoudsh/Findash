from fastapi import FastAPI
from datetime import datetime
import uvicorn

from .core.config import settings
from .routes import market_data, analysis, news, strategies
from .auth import routes as auth_routes

app = FastAPI(
    title="Quantum Trading Matrix API",
    description="API for accessing trading strategies, market data, and analysis.",
    version="1.0.0",
    debug=settings.api.DEBUG,
)

# Register routers
app.include_router(auth_routes.router)
app.include_router(market_data.router)
app.include_router(analysis.router)
app.include_router(news.router)
app.include_router(strategies.router)

@app.get("/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

def start():
    """Start the API server."""
    uvicorn.run(
        "api.main:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        reload=settings.api.DEBUG,
    )

if __name__ == "__main__":
    start() 