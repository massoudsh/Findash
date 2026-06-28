"""
TASK-006 — Unit tests for /api/assets endpoints
Run: pytest tests/test_assets_api.py -v
Requires: pytest, pytest-asyncio, httpx
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.main_refactored import app
from src.models.asset import ASSET_SEEDS

client = TestClient(app)


# ─── Mock asset service ───────────────────────────────────────────────────────

MOCK_PRICE = {
    "price":              3_500_000.0,
    "price_toman":        3_500_000.0,
    "change_24h":         45_000.0,
    "change_percent_24h": 1.3,
    "high_24h":           3_520_000.0,
    "low_24h":            3_450_000.0,
    "updated_at":         "2024-06-01T12:00:00",
    "source":             "tgju.org",
}

MOCK_USD_PRICE = {
    **MOCK_PRICE,
    "price_toman": 63_500.0,
}


def _mock_service(symbol_prices: dict | None = None):
    """Return a mock AssetService."""
    svc = MagicMock()
    prices = symbol_prices or {}

    async def get_price(symbol):
        return prices.get(symbol, MOCK_PRICE)

    async def get_all_prices(category=None):
        seeds = ASSET_SEEDS if not category else [s for s in ASSET_SEEDS if s["category"] == category]
        return [{**s, **MOCK_PRICE} for s in seeds]

    async def get_usd_to_toman():
        return 63_500.0

    async def get_history(symbol, days=30):
        return [{"timestamp": "2024-01-01T00:00:00", "open": 3_400_000, "high": 3_520_000, "low": 3_350_000, "close": 3_500_000}]

    svc.get_price = get_price
    svc.get_all_prices = get_all_prices
    svc.get_usd_to_toman = get_usd_to_toman
    svc.get_history = get_history
    return svc


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestAssetSymbolMap:
    def test_all_seeds_have_required_fields(self):
        for seed in ASSET_SEEDS:
            assert "symbol"     in seed, f"Missing symbol in {seed}"
            assert "name_fa"    in seed, f"Missing name_fa in {seed}"
            assert "name_en"    in seed, f"Missing name_en in {seed}"
            assert "category"   in seed, f"Missing category in {seed}"
            assert "source_key" in seed, f"Missing source_key in {seed}"

    def test_symbol_uniqueness(self):
        symbols = [s["symbol"] for s in ASSET_SEEDS]
        assert len(symbols) == len(set(symbols)), "Duplicate symbols found"

    def test_known_symbols_present(self):
        symbols = {s["symbol"] for s in ASSET_SEEDS}
        for expected in ["XAU18", "USD", "BTC", "COIN_FULL", "XAG", "RE_TEHRAN"]:
            assert expected in symbols, f"Missing expected symbol: {expected}"

    def test_valid_categories(self):
        valid = {"gold", "silver", "currency", "real_estate", "crypto"}
        for seed in ASSET_SEEDS:
            assert seed["category"] in valid, f"Invalid category: {seed['category']}"


class TestAssetsListEndpoint:
    @patch("src.api.routes.assets.get_asset_service")
    def test_list_returns_200(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets")
        assert res.status_code == 200

    @patch("src.api.routes.assets.get_asset_service")
    def test_list_has_correct_shape(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets")
        data = res.json()
        assert "assets"       in data
        assert "usd_to_toman" in data
        assert "last_updated" in data
        assert isinstance(data["assets"], list)

    @patch("src.api.routes.assets.get_asset_service")
    def test_category_filter_gold(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets?category=gold")
        assert res.status_code == 200

    def test_invalid_category_still_returns_200(self):
        # service filters by category, unknown → empty list
        res = client.get("/api/assets?category=unknown")
        assert res.status_code in (200, 422)


class TestAssetDetailEndpoint:
    @patch("src.api.routes.assets.get_asset_service")
    def test_valid_symbol_returns_200(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets/XAU18")
        assert res.status_code == 200
        data = res.json()
        assert data["symbol"]     == "XAU18"
        assert data["name_fa"]    == "طلای ۱۸ عیار (هر گرم)"
        assert data["price_toman"] > 0

    def test_unknown_symbol_returns_404(self):
        res = client.get("/api/assets/UNKNOWN_XYZ")
        assert res.status_code == 404

    @patch("src.api.routes.assets.get_asset_service")
    def test_lowercase_symbol_normalized(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets/xau18")
        assert res.status_code == 200


class TestUsdRateEndpoint:
    @patch("src.api.routes.assets.get_asset_service")
    def test_usd_rate_returns_float(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.get("/api/assets/usd-rate")
        assert res.status_code == 200
        data = res.json()
        assert "usd_to_toman" in data
        assert isinstance(data["usd_to_toman"], (int, float))


class TestPortfolioEndpoint:
    @patch("src.api.routes.assets.get_asset_service")
    def test_add_valid_asset(self, mock_dep):
        mock_dep.return_value = _mock_service()
        res = client.post("/api/assets/portfolio", json={
            "symbol":    "XAU18",
            "quantity":  5.0,
            "buy_price": 3_200_000.0,
            "buy_date":  "2023-01-15",
        })
        assert res.status_code == 201
        data = res.json()
        assert data["symbol"]        == "XAU18"
        assert data["quantity"]      == 5.0
        assert data["current_value"] > 0
        assert "profit_loss" in data

    def test_add_unknown_symbol_returns_404(self):
        res = client.post("/api/assets/portfolio", json={
            "symbol":    "FAKE_SYM",
            "quantity":  1.0,
            "buy_price": 100.0,
            "buy_date":  "2023-01-01",
        })
        assert res.status_code == 404

    def test_missing_required_fields_returns_422(self):
        res = client.post("/api/assets/portfolio", json={"symbol": "XAU18"})
        assert res.status_code == 422
