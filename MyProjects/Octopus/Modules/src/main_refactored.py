"""
Octopus Trading Platform — FastAPI entry point
Usage:
    uvicorn src.main_refactored:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.api.routes.assets import router as assets_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🐙 Octopus starting up...")
    yield
    logger.info("🐙 Octopus shutting down...")


app = FastAPI(
    title="Octopus Trading Platform",
    description="AI-Powered Trading Platform with Iranian Market Assets",
    version="0.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(assets_router)

# TODO: register other existing routers here as project grows
# app.include_router(market_data_router)
# app.include_router(trades_router)
# app.include_router(portfolio_router)
# app.include_router(risk_router)
# app.include_router(ai_models_router)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": app.version}
