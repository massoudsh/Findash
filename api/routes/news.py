"""News API routes."""

from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime

from core.models import NewsArticle
from ..main import verify_api_key

router = APIRouter(
    prefix="/news",
    tags=["News"],
    dependencies=[Depends(verify_api_key)]
)

@router.get("/articles/{symbol}", response_model=List[NewsArticle])
async def get_news_articles(
    symbol: str,
    limit: int = Query(10, ge=1, le=100),
    min_sentiment: Optional[float] = Query(None, ge=-1, le=1)
):
    """Get news articles for a symbol."""
    # Implementation... 