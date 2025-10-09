import pytest
from unittest.mock import patch, MagicMock

# --- Test Authentication Endpoints ---

class TestAuthEndpoints:
    """
    Tests for the /auth/token endpoint.
    """

    def test_login_success(self, client, test_user):
        """
        Test successful login with correct credentials.
        """
        response = client.post(
            "/auth/token",
            data={"username": "testuser", "password": "testpassword"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure_wrong_password(self, client, test_user):
        """
        Test login failure with an incorrect password.
        """
        response = client.post(
            "/auth/token",
            data={"username": "testuser", "password": "wrongpassword"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Incorrect username or password"}

    def test_login_failure_nonexistent_user(self, client):
        """
        Test login failure with a username that does not exist.
        """
        response = client.post(
            "/auth/token",
            data={"username": "nouser", "password": "somepassword"},
        )
        assert response.status_code == 401
        assert response.json() == {"detail": "Incorrect username or password"}


# --- Helper for getting auth token ---

@pytest.fixture(scope="function")
def auth_headers(client, test_user):
    """
    Logs in the test user and returns authentication headers.
    """
    response = client.post(
        "/auth/token",
        data={"username": "testuser", "password": "testpassword"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --- Test Strategy Endpoints ---

class TestStrategyEndpoints:
    """
    Tests for /strategies/* endpoints.
    """

    def test_backtest_unauthorized(self, client):
        """
        Test submitting a backtest without authentication.
        """
        response = client.post("/strategies/backtest", json={})
        assert response.status_code == 401

    @patch("api.endpoints.strategies.run_backtest_task")
    def test_backtest_submission_success(self, mock_run_backtest, client, auth_headers):
        """
        Test successful submission of a backtest task.
        """
        mock_task = MagicMock()
        mock_task.id = "test-task-id-123"
        mock_run_backtest.delay.return_value = mock_task

        payload = {
            "strategy_name": "Test Strategy",
            "symbol": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "parameters": {"param1": 10, "param2": "value"},
        }
        response = client.post("/strategies/backtest", json=payload, headers=auth_headers)

        assert response.status_code == 200
        assert response.json() == {"task_id": "test-task-id-123"}
        mock_run_backtest.delay.assert_called_once_with(
            payload["strategy_name"],
            payload["symbol"],
            payload["start_date"],
            payload["end_date"],
            payload["parameters"],
        )

    def test_backtest_submission_invalid_payload(self, client, auth_headers):
        """
        Test backtest submission with an invalid payload.
        """
        payload = {"strategy_name": "Incomplete"}  # Missing fields
        response = client.post("/strategies/backtest", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_get_results_unauthorized(self, client):
        """
        Test getting results without authentication.
        """
        response = client.get("/strategies/results/some-task-id")
        assert response.status_code == 401

    @patch("api.endpoints.strategies.AsyncResult")
    def test_get_results_pending(self, mock_async_result, client, auth_headers):
        """
        Test getting results for a pending task.
        """
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_async_result.return_value = mock_result

        response = client.get("/strategies/results/pending-task", headers=auth_headers)
        assert response.status_code == 202
        assert response.json() == {"status": "PENDING", "result": None}

    @patch("api.endpoints.strategies.AsyncResult")
    def test_get_results_success(self, mock_async_result, client, auth_headers):
        """
        Test getting results for a successful task.
        """
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"pnl": 1234.56, "sharpe_ratio": 1.5}
        mock_async_result.return_value = mock_result

        response = client.get("/strategies/results/success-task", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {
            "status": "SUCCESS",
            "result": {"pnl": 1234.56, "sharpe_ratio": 1.5},
        }

    @patch("api.endpoints.strategies.AsyncResult")
    def test_get_results_failure(self, mock_async_result, client, auth_headers):
        """
        Test getting results for a failed task.
        """
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.result = "Something went wrong"
        mock_async_result.return_value = mock_result

        response = client.get("/strategies/results/failed-task", headers=auth_headers)
        assert response.status_code == 200 # The endpoint itself succeeds
        assert response.json() == {
            "status": "FAILURE",
            "result": "Something went wrong",
        }
    
    @patch("api.endpoints.strategies.AsyncResult")
    def test_get_results_task_not_found(self, mock_async_result, client, auth_headers):
        """
        Test getting results for a task that does not exist.
        Celery's AsyncResult returns a PENDING state for unknown tasks.
        We might want to adjust this behavior in the future, but for now, we test the current reality.
        """
        mock_result = MagicMock()
        mock_result.state = 'PENDING' # This is what AsyncResult does for unknown tasks
        mock_result.result = None
        mock_async_result.return_value = mock_result

        response = client.get("/strategies/results/unknown-task", headers=auth_headers)

        assert response.status_code == 202
        assert response.json() == {"status": "PENDING", "result": None} 