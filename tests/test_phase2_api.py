"""
Phase 2: Backend APIs and data wiring — tests.
Covers: trading bots API, agent panels (M1/M4/M9), dashboard portfolio and account summary.
"""

import pytest
from fastapi.testclient import TestClient


class TestPhase2TradingBotsAPI:
    """Trading bots API: create, list, update, pause/start, risk and agent source config."""

    def test_list_trading_bots(self, client: TestClient):
        response = client.get("/api/trading-bots/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_trading_bot(self, client: TestClient):
        payload = {
            "name": "Test Bot",
            "strategy": "momentum",
            "executionMode": "paper",
            "symbols": ["ETHUSDT"],
            "agentSources": ["M4", "M9"],
            "risk": {"max_position_pct": 5.0, "stop_loss_pct": 2.0},
        }
        response = client.post("/api/trading-bots/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Bot"
        assert data["strategy"] == "momentum"
        assert data.get("execution_mode") == "paper"
        assert data.get("symbols") == ["ETHUSDT"]
        assert data.get("agent_sources") == ["M4", "M9"]


class TestPhase2AgentPanels:
    """Agent status/signals endpoints for M1, M4, M9, M11 panels."""

    def test_agent_panels(self, client: TestClient):
        response = client.get("/api/agents/panels")
        assert response.status_code == 200
        data = response.json()
        assert "data_collector" in data
        assert "strategy_signals" in data
        assert "sentiment" in data
        assert isinstance(data["data_collector"], list)
        assert isinstance(data["strategy_signals"], list)
        assert isinstance(data["sentiment"], list)


class TestPhase2DashboardSummary:
    """Dashboard portfolio and account summary from API."""

    def test_dashboard_summary(self, client: TestClient):
        response = client.get("/api/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert "account" in data
        assert "portfolios" in data
        assert "summary" in data
        assert "timestamp" in data
        assert "total_equity" in data["account"] or "total_portfolio_value" in data["summary"]
        assert isinstance(data["portfolios"], list)
