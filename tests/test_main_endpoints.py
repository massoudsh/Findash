import pytest
from unittest.mock import patch, MagicMock

# --- Test Authentication Endpoints ---
# NOTE: previously targeted a legacy `/auth/token` OAuth2-form route backed by
# the SQLAlchemy `User` model. That route no longer exists in the app; the
# real, current auth flow is the JSON-based `/api/auth/*` endpoints in
# `src/api/endpoints/professional_auth.py` (already exhaustively covered by
# tests/test_auth.py). These tests were updated to exercise the real routes.


@pytest.fixture(autouse=True)
def _reset_rate_limiter_state():
    """
    Avoid cross-test/cross-file bleed of the in-memory rate limiter fallback
    (auth_rate_limit is 5 requests/minute), mirroring tests/test_auth.py.
    """
    from src.core.rate_limiter import rate_limiter as _rl
    _rl._memory_requests.clear()
    _rl._memory_burst.clear()
    yield


class TestAuthEndpoints:
    """
    Tests for the /api/auth/login endpoint.
    """

    def test_login_success(self, client):
        """
        Test successful login with correct credentials.
        """
        response = client.post(
            "/api/auth/login",
            json={"email": "demo@octopus.trading", "password": "DemoUser2025!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure_wrong_password(self, client):
        """
        Test login failure with an incorrect password.
        """
        response = client.post(
            "/api/auth/login",
            json={"email": "demo@octopus.trading", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_login_failure_nonexistent_user(self, client):
        """
        Test login failure with an email that does not exist.
        """
        response = client.post(
            "/api/auth/login",
            json={"email": "nouser@example.com", "password": "somepassword"},
        )
        assert response.status_code == 401


# --- Helper for getting auth token ---

@pytest.fixture(scope="function")
def auth_headers(client):
    """
    Logs in a demo user and returns authentication headers.
    """
    response = client.post(
        "/api/auth/login",
        json={"email": "demo@octopus.trading", "password": "DemoUser2025!"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --- Test Strategy/Backtesting Endpoints ---
# NOTE: previously targeted a legacy `/strategies/backtest` + `/strategies/results/{id}`
# Celery task-id + AsyncResult polling API (`api.endpoints.strategies.run_backtest_task`)
# that doesn't exist anywhere in the codebase. The real, current equivalent is the
# synchronous, auth-gated `/api/backtesting/run` + `/api/backtesting/results/{id}`
# endpoints in `src/api/endpoints/backtesting.py`.

BACKTEST_PAYLOAD = {
    "strategy": "momentum",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}


class TestStrategyEndpoints:
    """
    Tests for /api/backtesting/* endpoints.
    """

    def test_backtest_unauthorized(self, client):
        """
        Test submitting a backtest without authentication.
        """
        response = client.post("/api/backtesting/run", json=BACKTEST_PAYLOAD)
        assert response.status_code == 401

    def test_backtest_submission_success(self, client, auth_headers):
        """
        Test successful submission of a backtest.
        """
        response = client.post("/api/backtesting/run", json=BACKTEST_PAYLOAD, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == BACKTEST_PAYLOAD["strategy"]
        assert data["symbol"] == BACKTEST_PAYLOAD["symbol"]
        assert "backtest_id" in data

    def test_backtest_submission_invalid_payload(self, client, auth_headers):
        """
        Test backtest submission with an invalid payload.
        """
        payload = {"strategy": "Incomplete"}  # Missing required fields
        response = client.post("/api/backtesting/run", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_get_results_unauthorized(self, client):
        """
        Test getting results without authentication.
        """
        response = client.get("/api/backtesting/results/some-task-id")
        assert response.status_code == 401

    def test_get_results_success(self, client, auth_headers):
        """
        Test getting results for a backtest that was actually run.
        """
        create_response = client.post("/api/backtesting/run", json=BACKTEST_PAYLOAD, headers=auth_headers)
        backtest_id = create_response.json()["backtest_id"]

        response = client.get(f"/api/backtesting/results/{backtest_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["backtest_id"] == backtest_id

    def test_get_results_task_not_found(self, client, auth_headers):
        """
        Test getting results for a backtest id that does not exist.
        """
        response = client.get("/api/backtesting/results/unknown-task", headers=auth_headers)
        assert response.status_code == 404
