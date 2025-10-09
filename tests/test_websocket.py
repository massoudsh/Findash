"""
Tests for WebSocket functionality in Quantum Trading Matrix™
Tests connection management, message handling, and real-time data
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from src.core.websocket import (
    WebSocketManager, WebSocketConnection, WebSocketMessage, MessageType,
    MarketData, TradeExecution, websocket_manager
)
from src.main import app


class TestWebSocketConnection:
    """Test WebSocket connection wrapper"""
    
    @pytest.fixture
    def mock_websocket(self):
        mock = Mock(spec=WebSocket)
        mock.send_text = AsyncMock()
        return mock
    
    @pytest.fixture
    def connection(self, mock_websocket):
        return WebSocketConnection(mock_websocket, "test-client-123")
    
    def test_connection_initialization(self, connection):
        """Test connection initialization"""
        assert connection.client_id == "test-client-123"
        assert connection.subscriptions == set()
        assert not connection.authenticated
        assert connection.user_id is None
        assert connection.metadata == {}
    
    @pytest.mark.asyncio
    async def test_send_message(self, connection):
        """Test sending message to client"""
        message = WebSocketMessage(MessageType.MARKET_DATA, {"symbol": "BTC-USD", "price": 45000})
        
        await connection.send_message(message)
        
        connection.websocket.send_text.assert_called_once()
        sent_data = json.loads(connection.websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "market_data"
        assert sent_data["client_id"] == "test-client-123"
    
    @pytest.mark.asyncio
    async def test_send_error(self, connection):
        """Test sending error message"""
        await connection.send_error("Test error", "TEST_ERROR")
        
        connection.websocket.send_text.assert_called_once()
        sent_data = json.loads(connection.websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert sent_data["data"]["code"] == "TEST_ERROR"
        assert sent_data["data"]["message"] == "Test error"
    
    def test_subscription_management(self, connection):
        """Test subscription and unsubscription"""
        # Subscribe
        connection.subscribe("market_data")
        assert "market_data" in connection.subscriptions
        assert connection.is_subscribed("market_data")
        
        # Unsubscribe
        connection.unsubscribe("market_data")
        assert "market_data" not in connection.subscriptions
        assert not connection.is_subscribed("market_data")


class TestWebSocketManager:
    """Test WebSocket manager"""
    
    @pytest.fixture
    def manager(self):
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        mock = Mock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_text = AsyncMock()
        mock.receive_text = AsyncMock()
        return mock
    
    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """Test WebSocket connection establishment"""
        client_id = await manager.connect(mock_websocket)
        
        assert client_id in manager.connections
        assert manager.stats["total_connections"] == 1
        assert manager.stats["active_connections"] == 1
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test WebSocket disconnection"""
        client_id = await manager.connect(mock_websocket)
        
        # Subscribe to a channel
        connection = manager.connections[client_id]
        connection.subscribe("test_channel")
        manager.channels["test_channel"] = {client_id}
        
        await manager.disconnect(client_id)
        
        assert client_id not in manager.connections
        assert manager.stats["active_connections"] == 0
        assert "test_channel" not in manager.channels  # Channel cleaned up
    
    @pytest.mark.asyncio
    async def test_authenticate_connection(self, manager, mock_websocket):
        """Test connection authentication"""
        client_id = await manager.connect(mock_websocket)
        
        user_data = {"user_id": "user123", "permissions": ["trade"]}
        await manager.authenticate_connection(client_id, "user123", user_data)
        
        connection = manager.connections[client_id]
        assert connection.authenticated
        assert connection.user_id == "user123"
        assert "user123" in manager.user_connections
        assert client_id in manager.user_connections["user123"]
    
    @pytest.mark.asyncio
    async def test_handle_subscription_message(self, manager, mock_websocket):
        """Test handling subscription message"""
        client_id = await manager.connect(mock_websocket)
        
        subscription_message = json.dumps({
            "type": "subscribe",
            "data": {"channel": "market_data"}
        })
        
        await manager.handle_message(client_id, subscription_message)
        
        connection = manager.connections[client_id]
        assert connection.is_subscribed("market_data")
        assert "market_data" in manager.channels
        assert client_id in manager.channels["market_data"]
    
    @pytest.mark.asyncio
    async def test_handle_unsubscription_message(self, manager, mock_websocket):
        """Test handling unsubscription message"""
        client_id = await manager.connect(mock_websocket)
        
        # First subscribe
        manager.connections[client_id].subscribe("market_data")
        manager.channels["market_data"] = {client_id}
        
        unsubscription_message = json.dumps({
            "type": "unsubscribe",
            "data": {"channel": "market_data"}
        })
        
        await manager.handle_message(client_id, unsubscription_message)
        
        connection = manager.connections[client_id]
        assert not connection.is_subscribed("market_data")
        assert "market_data" not in manager.channels  # Channel cleaned up
    
    @pytest.mark.asyncio
    async def test_handle_heartbeat_message(self, manager, mock_websocket):
        """Test handling heartbeat message"""
        client_id = await manager.connect(mock_websocket)
        
        heartbeat_message = json.dumps({
            "type": "heartbeat",
            "data": {}
        })
        
        # Mock the current time to test heartbeat update
        with patch('src.core.websocket.datetime') as mock_datetime:
            mock_now = datetime.now()
            mock_datetime.utcnow.return_value = mock_now
            
            await manager.handle_message(client_id, heartbeat_message)
            
            connection = manager.connections[client_id]
            assert connection.last_heartbeat == mock_now
    
    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager, mock_websocket):
        """Test broadcasting message to channel subscribers"""
        # Connect two clients
        client1_id = await manager.connect(mock_websocket)
        client2_id = await manager.connect(mock_websocket)
        
        # Subscribe both to same channel
        manager.connections[client1_id].subscribe("market_data")
        manager.connections[client2_id].subscribe("market_data")
        manager.channels["market_data"] = {client1_id, client2_id}
        
        message = WebSocketMessage(MessageType.MARKET_DATA, {"symbol": "BTC-USD"})
        await manager.broadcast_to_channel("market_data", message)
        
        # Both clients should receive the message
        assert mock_websocket.send_text.call_count == 2
        assert manager.stats["messages_sent"] == 2
    
    @pytest.mark.asyncio
    async def test_send_to_user(self, manager, mock_websocket):
        """Test sending message to specific user"""
        client_id = await manager.connect(mock_websocket)
        
        # Authenticate user
        await manager.authenticate_connection(client_id, "user123", {})
        
        message = WebSocketMessage(MessageType.SYSTEM_NOTIFICATION, {"text": "Hello user!"})
        sent = await manager.send_to_user("user123", message)
        
        assert sent
        mock_websocket.send_text.assert_called()
        assert manager.stats["messages_sent"] == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_market_data(self, manager, mock_websocket):
        """Test broadcasting market data"""
        client_id = await manager.connect(mock_websocket)
        
        # Subscribe to market data
        manager.connections[client_id].subscribe("market_data")
        manager.channels["market_data"] = {client_id}
        
        market_data = MarketData(
            symbol="BTC-USD",
            price=45000.0,
            change=500.0,
            change_percent=1.12,
            volume=100000
        )
        
        await manager.broadcast_market_data(market_data)
        
        # Check data is cached
        assert "BTC-USD" in manager.market_data_cache
        assert manager.market_data_cache["BTC-USD"] == market_data
        
        # Check message was sent
        mock_websocket.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_broadcast_trade_execution(self, manager, mock_websocket):
        """Test broadcasting trade execution"""
        client_id = await manager.connect(mock_websocket)
        
        # Authenticate user and subscribe to trades
        await manager.authenticate_connection(client_id, "user123", {})
        
        trade = TradeExecution(
            trade_id="trade123",
            symbol="BTC-USD",
            side="buy",
            quantity=0.1,
            price=45000.0,
            timestamp=datetime.utcnow().isoformat(),
            user_id="user123"
        )
        
        await manager.broadcast_trade_execution(trade)
        
        # User should receive the trade notification
        mock_websocket.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_send_risk_alert(self, manager, mock_websocket):
        """Test sending risk alert to user"""
        client_id = await manager.connect(mock_websocket)
        await manager.authenticate_connection(client_id, "user123", {})
        
        alert_data = {
            "type": "margin_call",
            "severity": "high",
            "message": "Margin call alert",
            "required_action": "Add funds or close positions"
        }
        
        await manager.send_risk_alert("user123", alert_data)
        
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "risk_alert"
        assert sent_data["data"] == alert_data
    
    def test_get_connection_stats(self, manager):
        """Test getting connection statistics"""
        stats = manager.get_connection_stats()
        
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "channels" in stats
        assert "authenticated_users" in stats
    
    @pytest.mark.asyncio
    async def test_invalid_json_message(self, manager, mock_websocket):
        """Test handling invalid JSON message"""
        client_id = await manager.connect(mock_websocket)
        
        invalid_json = "{ invalid json }"
        await manager.handle_message(client_id, invalid_json)
        
        # Should send error message
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "Invalid JSON format" in sent_data["data"]["message"]
    
    @pytest.mark.asyncio
    async def test_unknown_message_type(self, manager, mock_websocket):
        """Test handling unknown message type"""
        client_id = await manager.connect(mock_websocket)
        
        unknown_message = json.dumps({
            "type": "unknown_type",
            "data": {}
        })
        
        await manager.handle_message(client_id, unknown_message)
        
        # Should send error message
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "Unknown message type" in sent_data["data"]["message"]


class TestWebSocketMessages:
    """Test WebSocket message structures"""
    
    def test_websocket_message_creation(self):
        """Test WebSocket message creation"""
        data = {"symbol": "BTC-USD", "price": 45000}
        message = WebSocketMessage(MessageType.MARKET_DATA, data)
        
        assert message.type == MessageType.MARKET_DATA
        assert message.data == data
        assert message.timestamp is not None
        assert message.client_id is None
        
        # Test JSON serialization
        json_str = message.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "market_data"
        assert parsed["data"] == data
    
    def test_market_data_structure(self):
        """Test MarketData structure"""
        market_data = MarketData(
            symbol="BTC-USD",
            price=45000.0,
            change=500.0,
            change_percent=1.12,
            volume=100000,
            bid=44999.0,
            ask=45001.0,
            high_24h=46000.0,
            low_24h=44000.0
        )
        
        assert market_data.symbol == "BTC-USD"
        assert market_data.price == 45000.0
        assert market_data.change == 500.0
        assert market_data.bid == 44999.0
    
    def test_trade_execution_structure(self):
        """Test TradeExecution structure"""
        trade = TradeExecution(
            trade_id="trade123",
            symbol="BTC-USD",
            side="buy",
            quantity=0.1,
            price=45000.0,
            timestamp=datetime.utcnow().isoformat(),
            user_id="user123"
        )
        
        assert trade.trade_id == "trade123"
        assert trade.symbol == "BTC-USD"
        assert trade.side == "buy"
        assert trade.user_id == "user123"


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_websocket_stats_endpoint(self, client):
        """Test WebSocket stats endpoint"""
        response = client.get("/websocket/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_connections" in data
        assert "active_connections" in data
        assert "messages_sent" in data
    
    @pytest.mark.asyncio
    async def test_websocket_endpoint_connection(self):
        """Test WebSocket endpoint connection"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/trading") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "system_notification"
                assert "Connected to Quantum Trading Matrix" in data["data"]["message"]
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_flow(self):
        """Test complete subscription flow"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/trading") as websocket:
                # Receive welcome message
                welcome = websocket.receive_json()
                assert welcome["type"] == "system_notification"
                
                # Subscribe to market data
                websocket.send_json({
                    "type": "subscribe",
                    "data": {"channel": "market_data"}
                })
                
                # Should receive confirmation
                response = websocket.receive_json()
                assert response["type"] == "system_notification"
                assert "Subscribed to market_data" in response["data"]["message"]
    
    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat mechanism"""
        with TestClient(app) as client:
            with client.websocket_connect("/ws/trading") as websocket:
                # Skip welcome message
                websocket.receive_json()
                
                # Send heartbeat
                websocket.send_json({
                    "type": "heartbeat",
                    "data": {}
                })
                
                # Should receive heartbeat response
                response = websocket.receive_json()
                assert response["type"] == "heartbeat"
                assert "server_time" in response["data"]


@pytest.mark.asyncio
async def test_websocket_manager_background_tasks():
    """Test WebSocket manager background tasks"""
    manager = WebSocketManager()
    
    # Mock WebSocket for testing
    mock_websocket = Mock(spec=WebSocket)
    mock_websocket.accept = AsyncMock()
    mock_websocket.send_text = AsyncMock()
    
    # Connect a client to start background tasks
    client_id = await manager.connect(mock_websocket)
    
    # Background tasks should be running
    assert manager._heartbeat_task is not None
    assert manager._market_data_task is not None
    
    # Disconnect to stop background tasks
    await manager.disconnect(client_id)
    
    # Background tasks should be stopped
    assert manager._heartbeat_task is None
    assert manager._market_data_task is None


@pytest.mark.asyncio
async def test_websocket_error_handling():
    """Test WebSocket error handling"""
    manager = WebSocketManager()
    
    # Test with non-existent client
    result = await manager.send_to_user("non_existent_user", 
                                       WebSocketMessage(MessageType.SYSTEM_NOTIFICATION, {}))
    assert not result
    
    # Test disconnecting non-existent client
    await manager.disconnect("non_existent_client")  # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__]) 