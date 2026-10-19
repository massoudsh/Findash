import importlib.util
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


spec = importlib.util.spec_from_file_location(
    "iran_market_contract", Path(__file__).resolve().parents[1] / "src/api/endpoints/iran_market.py"
)
market = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market)


class IranMarketContractTests(unittest.TestCase):
    def setUp(self):
        market._cache.clear()
        market._cache_ts.clear()
        app = FastAPI()
        app.include_router(market.router)
        self.client = TestClient(app)

    def test_crypto_uses_nobitex_rls_keys_and_preserves_rial(self):
        with patch.object(market, "_tgju_price", AsyncMock(return_value=None)), patch.object(
            market, "_nobitex_stats", AsyncMock(return_value={
                "btc-rls": {"latest": "1100000", "dayOpen": "1000000"}
            })
        ):
            response = self.client.get("/api/iran-market/overview")
        self.assertEqual(response.status_code, 200)
        items = {item["symbol"]: item for item in response.json()["items"]}
        self.assertEqual(items["BTC-IRT"]["price"], 1100000)
        self.assertEqual(items["BTC-IRT"]["currency"], "IRR")
        self.assertEqual(items["BTC-IRT"]["change_pct"], 10)
        self.assertTrue(items["BTC-IRT"]["available"])

    def test_gold_and_currency_have_explicit_rial_units(self):
        with patch.object(market, "_tgju_price", AsyncMock(return_value={"price": 500000, "change_pct": 1})), patch.object(
            market, "_nobitex_stats", AsyncMock(return_value=None)
        ):
            data = self.client.get("/api/iran-market/overview").json()
        for item in data["items"]:
            self.assertEqual(item["currency"], "IRR")
            if item["category"] != "crypto":
                self.assertEqual(item["price"], 500000)
        self.assertGreater(data["cached_at"], 0)

    def test_provider_failure_returns_unavailable_not_sample_prices(self):
        with patch.object(market, "_tgju_price", AsyncMock(return_value=None)), patch.object(
            market, "_nobitex_stats", AsyncMock(return_value=None)
        ):
            data = self.client.get("/api/iran-market/overview").json()
        self.assertTrue(data["items"])
        self.assertTrue(all(not item["available"] and item["price"] is None for item in data["items"]))

    def test_cache_keeps_original_timestamp(self):
        with patch.object(market, "_tgju_price", AsyncMock(return_value=None)) as tgju, patch.object(
            market, "_nobitex_stats", AsyncMock(return_value=None)
        ):
            first = self.client.get("/api/iran-market/overview").json()
            second = self.client.get("/api/iran-market/overview").json()
        self.assertEqual(first["cached_at"], second["cached_at"])
        self.assertEqual(tgju.await_count, len(market.TGJU_INDICATORS))


if __name__ == "__main__":
    unittest.main()
