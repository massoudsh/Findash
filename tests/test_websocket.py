"""
Tests for WebSocket functionality (فین‌دَش)
Tests connection management, message handling, and real-time data.

بازنویسی: نسخه قبلی به `src.core.websocket` (ماژول حذف‌شده) و `src.main` (حذف‌شده،
جایگزین با `src.main_refactored`) اشاره می‌کرد. پیاده‌سازی واقعی WebSocket فعلی
`src/realtime/websockets.py` (`WebSocketManager`/`WebSocketConnection`، API مبتنی بر
dict، بدون کلاس `WebSocketMessage`/`MessageType`) و endpoint واقعی آن
`/api/ws/trading` در `src/api/endpoints/unified_websocket.py` است.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from fastapi import WebSocket
from fastapi.testclient import TestClient

from src.realtime.websockets import WebSocketManager, WebSocketConnection
from src.main_refactored import app


class TestWebSocketConnection:
    """Test WebSocketConnection dataclass"""

    @pytest.fixture
    def mock_websocket(self):
        mock = Mock(spec=WebSocket)
        mock.send_text = AsyncMock()
        mock.accept = AsyncMock()
        mock.close = AsyncMock()
        return mock

    def test_connection_fields(self, mock_websocket):
        connection = WebSocketConnection(
            websocket=mock_websocket,
            client_id="test-client-123",
            connected_at=datetime.utcnow(),
            subscriptions=set(),
        )
        assert connection.client_id == "test-client-123"
        assert connection.subscriptions == set()
        assert connection.last_ping is None


class TestWebSocketManager:
    """Test WebSocketManager (src.realtime.websockets)"""

    @pytest.fixture
    def manager(self):
        return WebSocketManager()

    @pytest.fixture
    def mock_websocket(self):
        mock = Mock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_text = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")

        assert "client-1" in manager.active_connections
        assert manager.stats["total_connections"] == 1
        assert manager.get_connection_count() == 1
        mock_websocket.accept.assert_called_once()
        # connect() sends a "connection_established" confirmation
        mock_websocket.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "market_data")

        await manager.disconnect("client-1")

        assert "client-1" not in manager.active_connections
        assert manager.get_connection_count() == 0
        assert "market_data" not in manager.subscriptions
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_unknown_client_is_noop(self, manager):
        # Should not raise
        await manager.disconnect("does-not-exist")

    @pytest.mark.asyncio
    async def test_subscribe_and_unsubscribe(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")

        subscribed = await manager.subscribe_client("client-1", "market_data")
        assert subscribed is True
        assert "market_data" in manager.active_connections["client-1"].subscriptions
        assert manager.get_channel_subscribers("market_data") == 1

        unsubscribed = await manager.unsubscribe_client("client-1", "market_data")
        assert unsubscribed is True
        assert "market_data" not in manager.active_connections["client-1"].subscriptions
        assert manager.get_channel_subscribers("market_data") == 0

    @pytest.mark.asyncio
    async def test_subscribe_unknown_client_returns_false(self, manager):
        result = await manager.subscribe_client("no-such-client", "market_data")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_client(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        mock_websocket.send_text.reset_mock()

        sent = await manager.send_to_client("client-1", {"type": "test", "data": {"x": 1}})

        assert sent is True
        mock_websocket.send_text.assert_called_once()
        payload = json.loads(mock_websocket.send_text.call_args[0][0])
        assert payload["type"] == "test"

    @pytest.mark.asyncio
    async def test_send_to_client_unknown_returns_false(self, manager):
        sent = await manager.send_to_client("no-such-client", {"type": "test"})
        assert sent is False

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.connect(mock_websocket, "client-2")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.broadcast_to_all({"type": "announcement"})

        assert sent_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.connect(mock_websocket, "client-2")
        await manager.subscribe_client("client-1", "market_data")
        await manager.subscribe_client("client-2", "market_data")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.broadcast_to_channel("market_data", {"type": "market_data", "symbol": "BTC-IRT"})

        assert sent_count == 2

    @pytest.mark.asyncio
    async def test_broadcast_to_channel_with_no_subscribers(self, manager):
        sent_count = await manager.broadcast_to_channel("empty_channel", {"type": "x"})
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")

        await manager.handle_message("client-1", json.dumps({"type": "subscribe", "channel": "market_data"}))

        assert "market_data" in manager.active_connections["client-1"].subscriptions

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "market_data")

        await manager.handle_message("client-1", json.dumps({"type": "unsubscribe", "channel": "market_data"}))

        assert "market_data" not in manager.active_connections["client-1"].subscriptions

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        assert manager.active_connections["client-1"].last_ping is None
        mock_websocket.send_text.reset_mock()

        await manager.handle_message("client-1", json.dumps({"type": "ping"}))

        assert manager.active_connections["client-1"].last_ping is not None
        mock_websocket.send_text.assert_called_once()
        payload = json.loads(mock_websocket.send_text.call_args[0][0])
        assert payload["type"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_get_subscriptions_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "market_data")
        mock_websocket.send_text.reset_mock()

        await manager.handle_message("client-1", json.dumps({"type": "get_subscriptions"}))

        payload = json.loads(mock_websocket.send_text.call_args[0][0])
        assert payload["type"] == "subscriptions"
        assert "market_data" in payload["subscriptions"]

    @pytest.mark.asyncio
    async def test_handle_invalid_json_message(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        mock_websocket.send_text.reset_mock()

        await manager.handle_message("client-1", "{ invalid json }")

        payload = json.loads(mock_websocket.send_text.call_args[0][0])
        assert payload["type"] == "error"
        assert "Invalid JSON" in payload["message"]

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        mock_websocket.send_text.reset_mock()

        await manager.handle_message("client-1", json.dumps({"type": "unknown_type"}))

        # Unknown types are just logged (no active_connections mutation, no exception)
        assert "client-1" in manager.active_connections

    @pytest.mark.asyncio
    async def test_send_market_data(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "market_data:BTC-IRT")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.send_market_data("BTC-IRT", {"price": 1000})

        assert sent_count == 1

    @pytest.mark.asyncio
    async def test_send_portfolio_update(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "portfolio:p1")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.send_portfolio_update("p1", {"value": 500})

        assert sent_count == 1

    @pytest.mark.asyncio
    async def test_send_alert(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "alerts:user123")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.send_alert("user123", {"message": "margin call"})

        assert sent_count == 1

    @pytest.mark.asyncio
    async def test_send_trade_update(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "trades:p1")
        mock_websocket.send_text.reset_mock()

        sent_count = await manager.send_trade_update("p1", {"side": "buy"})

        assert sent_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_price_update(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "prices:all")
        mock_websocket.send_text.reset_mock()

        await manager.broadcast_price_update({"symbol": "USD-IRR", "price": 60000})

        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_broadcast_sentiment_update(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.subscribe_client("client-1", "sentiment")
        mock_websocket.send_text.reset_mock()

        await manager.broadcast_sentiment_update({"symbol": "BTC-IRT", "score": 0.7})

        mock_websocket.send_text.assert_called()

    def test_get_stats(self, manager):
        stats = manager.get_stats()

        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "active_channels" in stats

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        # Force the connection to look stale
        manager.active_connections["client-1"].connected_at = datetime.utcfromtimestamp(0)

        removed = await manager.cleanup_stale_connections(timeout_minutes=1)

        assert removed == 1
        assert "client-1" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_all(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.connect(mock_websocket, "client-2")

        await manager.disconnect_all()

        assert manager.get_connection_count() == 0


class TestWebSocketTradingEndpoint:
    """
    Integration tests for the real `/api/ws/trading` endpoint (unified_websocket.py).

    از `client` fixture مشترک conftest.py استفاده نمی‌شود چون `websocket_manager`
    سراسری این router فقط در startup-event اپ (`main_refactored.py`) مقداردهی
    می‌شود؛ `TestClient` باید به‌صورت context manager (`with ... as client:`) استفاده
    شود تا lifespan واقعاً اجرا شود، وگرنه endpoint با کد 1008 بسته می‌شود.

    نکته: هنگام اتصال، دو پیام «connection_established» پشت سر هم دریافت
    می‌شود؛ اولی را `WebSocketManager.connect()` می‌فرستد (بدون
    `channels_available`) و دومی را خودِ endpoint (`unified_websocket.py`)
    با جزئیات کامل. تست‌ها هر دو را مصرف می‌کنند.
    """

    def test_websocket_connection_established(self):
        with TestClient(app) as client:
            with client.websocket_connect("/api/ws/trading") as websocket:
                first = websocket.receive_json()
                assert first["type"] == "connection_established"

                second = websocket.receive_json()
                assert second["type"] == "connection_established"
                assert "channels_available" in second

    def test_websocket_subscribe_flow(self):
        with TestClient(app) as client:
            with client.websocket_connect("/api/ws/trading") as websocket:
                websocket.receive_json()  # connection_established (manager)
                websocket.receive_json()  # connection_established (endpoint)

                websocket.send_json({"type": "subscribe", "channel": "market_data"})

                confirmation = websocket.receive_json()
                assert confirmation["type"] == "subscription_confirmed"
                assert confirmation["channel"] == "market_data"

    def test_websocket_ping_pong(self):
        with TestClient(app) as client:
            with client.websocket_connect("/api/ws/trading") as websocket:
                websocket.receive_json()  # connection_established (manager)
                websocket.receive_json()  # connection_established (endpoint)

                websocket.send_json({"type": "ping"})

                response = websocket.receive_json()
                assert response["type"] == "pong"
