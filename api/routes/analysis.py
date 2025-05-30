"""Analysis API routes."""

from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta

from core.models import TradingSignal, SocialMetrics
from ..main import verify_api_key

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
    dependencies=[Depends(verify_api_key)]
)

@router.get("/signals/{symbol}", response_model=List[TradingSignal])
async def get_trading_signals(
    symbol: str,
    lookback: int = Query(7, ge=1, le=30, description="Lookback period in days"),
    min_confidence: float = Query(0.5, ge=0, le=1)
):
    """Get trading signals for a symbol."""
    # Implementation...

@router.get("/social/{symbol}", response_model=SocialMetrics)
async def get_social_metrics(
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get social media metrics for a symbol."""
    # Implementation... 