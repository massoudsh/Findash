"""Market data API routes."""

from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime

from core.models import MarketData
from M1.Scraping import fetch_real_time_data
from ..main import verify_api_key

router = APIRouter(
    prefix="/market",
    tags=["Market Data"],
    dependencies=[Depends(verify_api_key)]
)

@router.get("/price/{symbol}", response_model=MarketData)
async def get_market_price(
    symbol: str,
    include_volume: bool = Query(True, description="Include trading volume")
):
    """Get real-time market price for a symbol."""
    data = await fetch_real_time_data(symbol)
    if not include_volume:
        data.volume = None
    return data

@router.get("/batch", response_model=List[MarketData])
async def get_batch_prices(
    symbols: List[str] = Query(..., min_items=1, max_items=10)
):
    """Get real-time market prices for multiple symbols."""
    return [
        await fetch_real_time_data(symbol)
        for symbol in symbols
    ] 