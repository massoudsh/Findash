"""
Alpha Vantage MCP API Endpoints

Exposes the Alpha Vantage remote MCP server (real-time quotes, historical
daily/intraday time series, symbol search) through this platform's own REST
API, so the frontend and internal services don't need to speak MCP directly.

See: src/data_processing/api_clients/alpha_vantage_mcp_client.py
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from src.data_processing.api_clients.alpha_vantage_mcp_client import (
    get_alpha_vantage_mcp_client,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alpha-vantage-mcp", tags=["Alpha Vantage MCP"])


@router.get("/tools")
async def list_mcp_tools() -> Dict[str, Any]:
    """List the data tools exposed by the Alpha Vantage MCP server."""
    try:
        client = get_alpha_vantage_mcp_client()
        tools = await client.list_tools()
        return {"status": "success", "tools": tools}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Alpha Vantage MCP list_tools failed: {e}")
        raise HTTPException(status_code=502, detail=f"Alpha Vantage MCP server error: {e}")


@router.get("/quote")
async def get_quote(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL")
) -> Dict[str, Any]:
    """Real-time quote for a symbol via the Alpha Vantage MCP server."""
    try:
        client = get_alpha_vantage_mcp_client()
        data = await client.get_quote(symbol.upper())
        return {"status": "success", "symbol": symbol.upper(), "data": data}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Alpha Vantage MCP quote failed for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Alpha Vantage MCP server error: {e}")


@router.get("/historical/daily")
async def get_daily_history(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    outputsize: str = Query(
        "compact", description="'compact' (last 100 days) or 'full' (20+ years)"
    ),
) -> Dict[str, Any]:
    """Historical daily OHLCV time series via the Alpha Vantage MCP server."""
    try:
        client = get_alpha_vantage_mcp_client()
        data = await client.get_daily_history(symbol.upper(), outputsize=outputsize)
        return {"status": "success", "symbol": symbol.upper(), "data": data}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Alpha Vantage MCP daily history failed for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Alpha Vantage MCP server error: {e}")


@router.get("/historical/intraday")
async def get_intraday_history(
    symbol: str = Query(..., description="Ticker symbol, e.g. AAPL"),
    interval: str = Query("5min", description="1min/5min/15min/30min/60min"),
    outputsize: str = Query(
        "compact", description="'compact' (last 100 points) or 'full'"
    ),
) -> Dict[str, Any]:
    """Intraday OHLCV time series via the Alpha Vantage MCP server."""
    try:
        client = get_alpha_vantage_mcp_client()
        data = await client.get_intraday(
            symbol.upper(), interval=interval, outputsize=outputsize
        )
        return {"status": "success", "symbol": symbol.upper(), "data": data}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Alpha Vantage MCP intraday history failed for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Alpha Vantage MCP server error: {e}")


@router.get("/search")
async def search_symbol(
    keywords: str = Query(..., description="Company name or keyword to search for")
) -> Dict[str, Any]:
    """Search for ticker symbols matching a keyword via the Alpha Vantage MCP server."""
    try:
        client = get_alpha_vantage_mcp_client()
        data = await client.search_symbol(keywords)
        return {"status": "success", "keywords": keywords, "data": data}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Alpha Vantage MCP symbol search failed for '{keywords}': {e}")
        raise HTTPException(status_code=502, detail=f"Alpha Vantage MCP server error: {e}")
