"""
Alpha Vantage MCP Client

Connects to Alpha Vantage's official remote Model Context Protocol (MCP) server
(https://mcp.alphavantage.co/mcp) over the Streamable HTTP transport, so this
platform can call Alpha Vantage's data tools (real-time quotes, historical time
series, symbol search, etc.) the same way an MCP-compatible AI agent would.

This complements the existing plain-REST `AlphaVantageClient`
(src/data_processing/api_clients/alpha_vantage_client.py): the MCP client is the
integration point for agent/tool-calling use cases, while the REST client remains
useful for simple direct HTTP calls.

Uses the free Alpha Vantage API key (same key as the REST API,
ALPHA_VANTAGE_API_KEY) passed as a query parameter to the MCP endpoint.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

MCP_SERVER_BASE_URL = "https://mcp.alphavantage.co/mcp"


class AlphaVantageMCPClient:
    """
    Thin async client around the Alpha Vantage remote MCP server.

    Each call opens a short-lived MCP session (the server is stateless-friendly
    over Streamable HTTP), lists/calls the requested tool, and returns plain
    Python data structures instead of raw MCP protocol objects.
    """

    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise ValueError("An Alpha Vantage API key is required for the MCP client")
        self.api_key = api_key
        self.timeout = timeout
        self.server_url = f"{MCP_SERVER_BASE_URL}?apikey={api_key}"

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        async with streamablehttp_client(self.server_url, timeout=self.timeout) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List the data tools exposed by the Alpha Vantage MCP server."""
        async with self._session() as session:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in result.tools
            ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an arbitrary Alpha Vantage MCP tool by name (e.g. 'GLOBAL_QUOTE',
        'TIME_SERIES_DAILY', 'TIME_SERIES_INTRADAY', 'SYMBOL_SEARCH', ...).
        """
        async with self._session() as session:
            result = await session.call_tool(name, arguments)

            if result.isError:
                error_text = "; ".join(
                    getattr(block, "text", str(block)) for block in result.content
                )
                raise RuntimeError(f"Alpha Vantage MCP tool '{name}' failed: {error_text}")

            # Prefer structured content when the tool provides it; otherwise fall
            # back to parsing/joining the text content blocks.
            if result.structuredContent is not None:
                return result.structuredContent

            text_parts = [
                block.text for block in result.content if getattr(block, "type", None) == "text"
            ]
            return {"raw_text": "\n".join(text_parts)}

    # ------------------------------------------------------------------
    # Convenience wrappers for the most common data needs
    # ------------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Real-time (last available) quote for a symbol."""
        return await self.call_tool("GLOBAL_QUOTE", {"symbol": symbol})

    async def get_daily_history(
        self, symbol: str, outputsize: str = "compact"
    ) -> Dict[str, Any]:
        """Historical daily OHLCV time series ('compact' = last 100 days, 'full' = 20+ years)."""
        return await self.call_tool(
            "TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": outputsize}
        )

    async def get_intraday(
        self, symbol: str, interval: str = "5min", outputsize: str = "compact"
    ) -> Dict[str, Any]:
        """Intraday OHLCV time series (interval: 1min/5min/15min/30min/60min)."""
        return await self.call_tool(
            "TIME_SERIES_INTRADAY",
            {"symbol": symbol, "interval": interval, "outputsize": outputsize},
        )

    async def search_symbol(self, keywords: str) -> Dict[str, Any]:
        """Search for ticker symbols matching a company name/keyword."""
        return await self.call_tool("SYMBOL_SEARCH", {"keywords": keywords})


_client_instance: Optional[AlphaVantageMCPClient] = None


def get_alpha_vantage_mcp_client() -> AlphaVantageMCPClient:
    """Return a process-wide singleton client, built from app settings."""
    global _client_instance
    if _client_instance is None:
        from src.core.config import get_settings

        settings = get_settings()
        api_key = settings.external_apis.alpha_vantage_api_key
        if not api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is not configured; set it in the environment "
                "to use the Alpha Vantage MCP integration."
            )
        _client_instance = AlphaVantageMCPClient(api_key=api_key)
    return _client_instance
