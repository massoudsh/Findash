"""
TASK-006 — Unit tests for AssetService (TGJU fetch + cache logic)
Run: pytest tests/test_asset_service.py -v
"""
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.asset_service import AssetService, SYMBOL_MAP


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)       # cache miss by default
    r.setex = AsyncMock(return_value=True)
    return r


@pytest.fixture
def service(mock_redis):
    return AssetService(redis_client=mock_redis, db_session=None)


MOCK_TGJU_RESPONSE = {
    "data": [[
        "635000000",   # price (rial)
        "5000000",     # change
        "0.8",         # change percent
        "628000000",   # low
        "638000000",   # high
        "", "", "", "", "", "", "",
        "1403/03/14 12:30",  # timestamp
    ]]
}


class TestAssetServiceGetPrice:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, service, mock_redis):
        cached_data = json.dumps({"price_toman": 63_500.0, "source": "tgju.org"})
        mock_redis.get = AsyncMock(return_value=cached_data)

        result = await service.get_price("USD")
        assert result is not None
        assert result["price_toman"] == 63_500.0
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.asset_service.httpx.AsyncClient")
    async def test_cache_miss_fetches_tgju(self, mock_httpx, service, mock_redis):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TGJU_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.get        = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_client

        result = await service.get_price("USD")
        assert result is not None
        assert result["price_toman"] == 63_500_000.0   # 635000000 / 10
        assert result["source"] == "tgju.org"

    @pytest.mark.asyncio
    async def test_unknown_symbol_returns_none(self, service):
        result = await service.get_price("UNKNOWN_SYM")
        assert result is None

    @pytest.mark.asyncio
    @patch("src.services.asset_service.httpx.AsyncClient")
    async def test_tgju_error_returns_none(self, mock_httpx, service, mock_redis):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.get        = AsyncMock(side_effect=Exception("Connection error"))
        mock_httpx.return_value = mock_client

        result = await service.get_price("XAU18")
        assert result is None

    @pytest.mark.asyncio
    @patch("src.services.asset_service.httpx.AsyncClient")
    async def test_cache_stored_after_fetch(self, mock_httpx, service, mock_redis):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_TGJU_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_client.get        = AsyncMock(return_value=mock_response)
        mock_httpx.return_value = mock_client

        await service.get_price("USD")
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "asset:price:USD"
        assert call_args[0][1] == 60   # TTL


class TestSymbolMap:
    def test_all_seeds_in_symbol_map(self):
        from src.models.asset import ASSET_SEEDS
        for seed in ASSET_SEEDS:
            assert seed["symbol"] in SYMBOL_MAP

    def test_symbol_map_has_source_key(self):
        for symbol, meta in SYMBOL_MAP.items():
            assert "source_key" in meta, f"Missing source_key for {symbol}"
