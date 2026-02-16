"""
Platform search API. Intended to be backed by Elasticsearch for fast full-text search
across pages, symbols, strategies, and content. Until Elasticsearch is configured,
returns results from a static platform index.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["Search"])

# Static platform index (replace with Elasticsearch query when available)
PLATFORM_INDEX = [
    {"id": "dashboard", "title": "Dashboard", "path": "/dashboard", "type": "page"},
    {"id": "trading", "title": "Trading Center", "path": "/trading", "type": "page"},
    {"id": "options", "title": "Options", "path": "/options", "type": "page"},
    {"id": "trades", "title": "Live Trading", "path": "/trades", "type": "page"},
    {"id": "portfolio", "title": "Portfolio", "path": "/portfolio", "type": "page"},
    {"id": "strategies", "title": "Strategies", "path": "/strategies", "type": "page"},
    {"id": "risk", "title": "Risk Assessment", "path": "/risk", "type": "page"},
    {"id": "backtesting", "title": "Backtesting", "path": "/backtesting", "type": "page"},
    {"id": "market-data", "title": "Market Data", "path": "/market-data", "type": "page"},
    {"id": "realtime", "title": "Real-time", "path": "/realtime", "type": "page"},
    {"id": "reports", "title": "Reports", "path": "/reports", "type": "page"},
    {"id": "settings", "title": "Settings", "path": "/settings", "type": "page"},
]


@router.get("")
async def search(
    q: str = Query(..., min_length=2),
    limit: int = Query(15, ge=1, le=50),
) -> dict:
    """
    Search anything in the platform. When Elasticsearch is configured, this endpoint
    will perform fast full-text search across pages, symbols, strategies, and content.
    """
    q_lower = q.lower().strip()
    results: List[dict] = []
    for item in PLATFORM_INDEX:
        if q_lower in item["title"].lower() or q_lower in item["path"].lower():
            results.append(item)
            if len(results) >= limit:
                break
    return {"results": results}
