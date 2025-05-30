from .routes import market_data, analysis, news

# Register routers
app.include_router(market_data.router)
app.include_router(analysis.router)
app.include_router(news.router)

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
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8000),
        reload=config.get("api.debug", False)
    )

if __name__ == "__main__":
    start() 