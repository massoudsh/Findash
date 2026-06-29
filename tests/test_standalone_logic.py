"""
Standalone unit tests — no DB/Redis/Kafka needed.
Tests pure business logic: asset helpers, alert logic, news parsing, portfolio math.
"""
import pytest
import json
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


# ── Asset / Iranian market helpers ──────────────────────────────────────────

class TestAssetPriceHelpers:
    """Tests for asset price calculation helpers (pure functions)."""

    def _pct_change(self, old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100

    def test_positive_change(self):
        assert round(self._pct_change(100, 110), 2) == 10.0

    def test_negative_change(self):
        assert round(self._pct_change(200, 180), 2) == -10.0

    def test_zero_old_price_is_safe(self):
        assert self._pct_change(0, 50) == 0.0

    def test_no_change(self):
        assert self._pct_change(150, 150) == 0.0

    def _format_irt(self, toman: float) -> str:
        return f"{toman:,.0f} تومان"

    def test_format_irt(self):
        assert self._format_irt(1_500_000) == "1,500,000 تومان"

    def test_format_irt_small(self):
        assert self._format_irt(0) == "0 تومان"


# ── Price alert logic ────────────────────────────────────────────────────────

class TestPriceAlertLogic:
    """Tests for price alert trigger logic (mirrors use-price-alerts.ts logic in Python)."""

    def _should_trigger(self, alert: dict, price: float) -> bool:
        if alert.get("triggered"):
            return False
        if alert["direction"] == "above":
            return price >= alert["targetPrice"]
        elif alert["direction"] == "below":
            return price <= alert["targetPrice"]
        return False

    def test_above_trigger_exact(self):
        alert = {"symbol": "AAPL", "targetPrice": 200.0, "direction": "above", "triggered": False}
        assert self._should_trigger(alert, 200.0) is True

    def test_above_trigger_exceeded(self):
        alert = {"symbol": "AAPL", "targetPrice": 200.0, "direction": "above", "triggered": False}
        assert self._should_trigger(alert, 205.0) is True

    def test_above_not_triggered(self):
        alert = {"symbol": "AAPL", "targetPrice": 200.0, "direction": "above", "triggered": False}
        assert self._should_trigger(alert, 199.99) is False

    def test_below_trigger(self):
        alert = {"symbol": "BTC-USD", "targetPrice": 60000, "direction": "below", "triggered": False}
        assert self._should_trigger(alert, 59999) is True

    def test_already_triggered_skipped(self):
        alert = {"symbol": "AAPL", "targetPrice": 200.0, "direction": "above", "triggered": True}
        assert self._should_trigger(alert, 250.0) is False


# ── Portfolio P&L math ────────────────────────────────────────────────────────

class TestPortfolioPnl:
    """Tests for trade tracker P&L calculations (mirrors trade-tracker.tsx logic in Python)."""

    def _calc_positions(self, trades: list, prices: dict) -> list:
        """Mirror of calcPositions in trade-tracker.tsx."""
        pos_map: dict = {}
        for t in trades:
            sym = t["symbol"]
            if sym not in pos_map:
                pos_map[sym] = {"qty": 0, "cost": 0.0}
            if t["side"] == "buy":
                pos_map[sym]["cost"] += t["price"] * t["quantity"]
                pos_map[sym]["qty"] += t["quantity"]
            else:
                pos_map[sym]["cost"] -= t["price"] * t["quantity"]
                pos_map[sym]["qty"] -= t["quantity"]

        result = []
        for sym, v in pos_map.items():
            if v["qty"] <= 0:
                continue
            avg_cost = v["cost"] / v["qty"]
            current_price = prices.get(sym, avg_cost)
            market_value = current_price * v["qty"]
            unrealized_pnl = market_value - v["cost"]
            pnl_pct = (unrealized_pnl / v["cost"]) * 100 if v["cost"] > 0 else 0
            result.append({
                "symbol": sym,
                "quantity": v["qty"],
                "avgCost": avg_cost,
                "marketValue": market_value,
                "unrealizedPnl": unrealized_pnl,
                "unrealizedPnlPct": pnl_pct,
            })
        return result

    def test_single_buy_profit(self):
        trades = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0}]
        positions = self._calc_positions(trades, {"AAPL": 160.0})
        assert len(positions) == 1
        pos = positions[0]
        assert pos["quantity"] == 10
        assert pos["avgCost"] == 150.0
        assert pos["unrealizedPnl"] == pytest.approx(100.0)
        assert pos["unrealizedPnlPct"] == pytest.approx(6.666, rel=1e-2)

    def test_single_buy_loss(self):
        trades = [{"symbol": "TSLA", "side": "buy", "quantity": 5, "price": 250.0}]
        positions = self._calc_positions(trades, {"TSLA": 200.0})
        pos = positions[0]
        assert pos["unrealizedPnl"] == pytest.approx(-250.0)

    def test_buy_then_partial_sell_removes_qty(self):
        trades = [
            {"symbol": "MSFT", "side": "buy", "quantity": 20, "price": 300.0},
            {"symbol": "MSFT", "side": "sell", "quantity": 5, "price": 310.0},
        ]
        positions = self._calc_positions(trades, {"MSFT": 320.0})
        assert len(positions) == 1
        assert positions[0]["quantity"] == 15

    def test_full_sell_removes_position(self):
        trades = [
            {"symbol": "NVDA", "side": "buy", "quantity": 10, "price": 500.0},
            {"symbol": "NVDA", "side": "sell", "quantity": 10, "price": 600.0},
        ]
        positions = self._calc_positions(trades, {"NVDA": 650.0})
        assert len(positions) == 0

    def test_multiple_symbols(self):
        trades = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "price": 150.0},
            {"symbol": "BTC-USD", "side": "buy", "quantity": 1, "price": 60000.0},
        ]
        positions = self._calc_positions(trades, {"AAPL": 155.0, "BTC-USD": 65000.0})
        assert len(positions) == 2
        syms = {p["symbol"] for p in positions}
        assert syms == {"AAPL", "BTC-USD"}


# ── News RSS parser ──────────────────────────────────────────────────────────

class TestNewsRSSParser:
    """Tests for the RSS parsing logic used in /api/news/route.ts (logic mirrored here)."""

    def _parse_rss_items(self, xml: str, source: str) -> list:
        """Simple RSS item extractor — mirrors parseRSS in route.ts."""
        import re
        items = []
        item_pattern = re.compile(r'<item[^>]*>(.*?)</item>', re.DOTALL)
        for m in item_pattern.finditer(xml):
            block = m.group(1)
            def get(tag):
                match = re.search(
                    rf'<{tag}[^>]*><!\[CDATA\[(.*?)\]\]></{tag}>|<{tag}[^>]*>(.*?)</{tag}>',
                    block, re.DOTALL
                )
                return (match.group(1) or match.group(2) or '').strip() if match else ''
            title = get('title')
            if not title:
                continue
            items.append({"title": title, "link": get('link'), "source": source})
        return items

    def _sample_rss(self):
        return """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title><![CDATA[قیمت طلا افزایش یافت]]></title>
      <link>https://example.com/1</link>
    </item>
    <item>
      <title>Dollar reaches new high</title>
      <link>https://example.com/2</link>
    </item>
  </channel>
</rss>"""

    def test_parse_two_items(self):
        items = self._parse_rss_items(self._sample_rss(), "TestSource")
        assert len(items) == 2

    def test_cdata_title(self):
        items = self._parse_rss_items(self._sample_rss(), "TestSource")
        assert items[0]["title"] == "قیمت طلا افزایش یافت"

    def test_plain_title(self):
        items = self._parse_rss_items(self._sample_rss(), "TestSource")
        assert items[1]["title"] == "Dollar reaches new high"

    def test_source_attached(self):
        items = self._parse_rss_items(self._sample_rss(), "TGJU")
        assert all(i["source"] == "TGJU" for i in items)

    def test_empty_feed_returns_empty(self):
        items = self._parse_rss_items("<rss><channel></channel></rss>", "X")
        assert items == []


# ── WebSocket message parsing ────────────────────────────────────────────────

class TestWebSocketMessageParsing:
    """Tests for WS tick message parsing (mirrors use-market-ws.ts logic)."""

    def _parse_ws_message(self, raw: str) -> dict | None:
        try:
            msg = json.loads(raw)
            if msg.get("type") == "tick" and isinstance(msg.get("data"), dict):
                return msg["data"]
            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def test_valid_tick(self):
        raw = json.dumps({"type": "tick", "data": {"symbol": "AAPL", "price": 180.5}})
        result = self._parse_ws_message(raw)
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 180.5

    def test_non_tick_returns_none(self):
        raw = json.dumps({"type": "heartbeat"})
        assert self._parse_ws_message(raw) is None

    def test_invalid_json_returns_none(self):
        assert self._parse_ws_message("not-json{") is None

    def test_missing_data_returns_none(self):
        raw = json.dumps({"type": "tick"})
        assert self._parse_ws_message(raw) is None

    def test_batch_message_type(self):
        raw = json.dumps({"type": "batch", "data": [{"symbol": "BTC", "price": 60000}]})
        msg = json.loads(raw)
        assert msg["type"] == "batch"
        assert isinstance(msg["data"], list)
        assert msg["data"][0]["symbol"] == "BTC"


# ── Utility / formatting ──────────────────────────────────────────────────────

class TestFormatters:
    """Currency and percentage formatting."""

    def _format_currency(self, value: float, prefix: str = "$") -> str:
        return f"{prefix}{value:,.2f}"

    def _format_pct(self, value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    def test_format_currency_basic(self):
        assert self._format_currency(1234.5) == "$1,234.50"

    def test_format_currency_large(self):
        assert self._format_currency(1_000_000) == "$1,000,000.00"

    def test_format_pct_positive(self):
        assert self._format_pct(5.25) == "+5.25%"

    def test_format_pct_negative(self):
        assert self._format_pct(-3.1) == "-3.10%"

    def test_format_pct_zero(self):
        assert self._format_pct(0) == "+0.00%"
