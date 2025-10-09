"""
Test suite for authentication system
Tests JWT authentication, password hashing, rate limiting, and security features
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
import json

from src.main_refactored import app
from src.core.security import (
    verify_token,
    create_refresh_token,
    verify_password,
    hash_password,
    get_password_hash
)
from src.core.config import get_settings

settings = get_settings()

# Test client
client = TestClient(app)


class TestPasswordSecurity:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test password is properly hashed"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        # Hash should be different from original password
        assert hashed != password
        # Hash should be consistent
        assert verify_password(password, hashed)
        # Wrong password should fail
        assert not verify_password("wrong_password", hashed)
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes"""
        password = "test_password_123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        # Different hashes but both should verify
        assert hash1 != hash2
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestJWTTokens:
    """Test JWT token creation and verification"""
    
    def test_create_access_token(self):
        """Test access token creation"""
        user_data = {"sub": "user123", "email": "test@example.com"}
        token = create_access_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long
    
    def test_verify_valid_token(self):
        """Test valid token verification"""
        user_data = {"sub": "user123", "email": "test@example.com"}
        token = create_access_token(user_data)
        
        payload = verify_token(token, "access")
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"
    
    def test_verify_invalid_token(self):
        """Test invalid token verification"""
        invalid_token = "invalid.token.here"
        
        payload = verify_token(invalid_token, "access")
        assert payload is None
    
    def test_verify_expired_token(self):
        """Test expired token verification"""
        user_data = {"sub": "user123", "email": "test@example.com"}
        
        # Create token that expires immediately
        expired_delta = timedelta(seconds=-1)
        token = create_access_token(user_data, expired_delta)
        
        payload = verify_token(token, "access")
        assert payload is None
    
    def test_refresh_token_creation(self):
        """Test refresh token creation"""
        user_data = {"sub": "user123"}
        token = create_refresh_token(user_data)
        
        payload = verify_token(token, "refresh")
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"


class TestAuthenticationEndpoints:
    """Test authentication API endpoints"""
    
    def test_login_success(self):
        """Test successful login"""
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_email(self):
        """Test login with invalid email"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "demo123"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]
    
    def test_login_invalid_password(self):
        """Test login with invalid password"""
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "wrong_password"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]
    
    def test_login_invalid_email_format(self):
        """Test login with invalid email format"""
        login_data = {
            "email": "invalid-email-format",
            "password": "demo123"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_register_success(self):
        """Test successful user registration"""
        register_data = {
            "email": "newuser@example.com",
            "password": "newpassword123",
            "confirm_password": "newpassword123",
            "first_name": "New",
            "last_name": "User"
        }
        
        response = client.post("/api/auth/register", json=register_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "User created successfully"
        assert "user_id" in data
        assert data["email"] == "newuser@example.com"
    
    def test_register_existing_email(self):
        """Test registration with existing email"""
        register_data = {
            "email": "demo@quantumtrading.com",  # Already exists
            "password": "newpassword123",
            "confirm_password": "newpassword123",
            "first_name": "Demo",
            "last_name": "User"
        }
        
        response = client.post("/api/auth/register", json=register_data)
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    def test_register_password_mismatch(self):
        """Test registration with mismatched passwords"""
        register_data = {
            "email": "newuser2@example.com",
            "password": "password123",
            "confirm_password": "different_password",
            "first_name": "New",
            "last_name": "User"
        }
        
        response = client.post("/api/auth/register", json=register_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_refresh_token_success(self):
        """Test successful token refresh"""
        # First login to get refresh token
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        login_response = client.post("/api/auth/login", json=login_data)
        login_data_response = login_response.json()
        refresh_token = login_data_response["refresh_token"]
        
        # Now refresh the token
        refresh_data = {"refresh_token": refresh_token}
        response = client.post("/api/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_refresh_token_invalid(self):
        """Test refresh with invalid token"""
        refresh_data = {"refresh_token": "invalid.refresh.token"}
        response = client.post("/api/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
    
    def test_get_current_user_profile(self):
        """Test getting current user profile"""
        # Login and get access token
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Get user profile
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "demo@quantumtrading.com"
        assert data["first_name"] == "Demo"
        assert data["last_name"] == "User"
        assert data["is_active"] is True
    
    def test_get_profile_without_auth(self):
        """Test getting profile without authentication"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 403  # No authorization header
    
    def test_get_profile_invalid_token(self):
        """Test getting profile with invalid token"""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = client.get("/api/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_logout(self):
        """Test user logout"""
        # Login first
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Logout
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/auth/logout", headers=headers)
        
        assert response.status_code == 200
        assert response.json()["message"] == "Successfully logged out"


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiting_enforcement(self):
        """Test that rate limiting is enforced"""
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        # Make requests up to the limit
        responses = []
        for _ in range(105):  # Exceed the limit of 100 per minute
            response = client.post("/api/auth/login", json=login_data)
            responses.append(response)
        
        # Check that some requests are rate limited
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0
        
        # Check rate limit response format
        if rate_limited_responses:
            error_response = rate_limited_responses[0].json()
            assert error_response["error"] == "rate_limit_exceeded"
            assert "Rate limit exceeded" in error_response["message"]


class TestSecurityHeaders:
    """Test security headers middleware"""
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses"""
        response = client.get("/health")
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers


class TestAPIValidation:
    """Test API input validation"""
    
    def test_email_validation(self):
        """Test email format validation"""
        invalid_emails = [
            "not-an-email",
            "@domain.com",
            "user@",
            "user..name@domain.com",
            ""
        ]
        
        for invalid_email in invalid_emails:
            login_data = {
                "email": invalid_email,
                "password": "password123"
            }
            
            response = client.post("/api/auth/login", json=login_data)
            assert response.status_code == 422  # Validation error
    
    def test_password_length_validation(self):
        """Test password length validation"""
        # Password too short
        register_data = {
            "email": "test@example.com",
            "password": "short",  # Less than 8 characters
            "confirm_password": "short",
            "first_name": "Test",
            "last_name": "User"
        }
        
        response = client.post("/api/auth/register", json=register_data)
        assert response.status_code == 422
    
    def test_required_fields_validation(self):
        """Test that required fields are validated"""
        # Missing required fields
        incomplete_data = {
            "email": "test@example.com"
            # Missing password, first_name, last_name, confirm_password
        }
        
        response = client.post("/api/auth/register", json=incomplete_data)
        assert response.status_code == 422


class TestAPIKeyManagement:
    """Test API key creation and management"""
    
    def test_create_api_key(self):
        """Test API key creation"""
        # Login first
        login_data = {
            "email": "demo@quantumtrading.com",
            "password": "demo123"
        }
        
        login_response = client.post("/api/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Create API key
        api_key_data = {
            "name": "Test API Key",
            "description": "API key for testing",
            "expires_in_days": 30
        }
        
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.post("/api/auth/api-keys", json=api_key_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["name"] == "Test API Key"
        assert data["description"] == "API key for testing"
        assert data["api_key"].startswith("qtm_")  # Our API key prefix


@pytest.fixture
def authenticated_client():
    """Fixture that provides an authenticated test client"""
    login_data = {
        "email": "demo@quantumtrading.com",
        "password": "demo123"
    }
    
    response = client.post("/api/auth/login", json=login_data)
    access_token = response.json()["access_token"]
    
    # Create a client with auth headers
    class AuthenticatedTestClient:
        def __init__(self, token):
            self.token = token
            self.headers = {"Authorization": f"Bearer {token}"}
        
        def get(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return client.get(url, **kwargs)
        
        def post(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return client.post(url, **kwargs)
        
        def put(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return client.put(url, **kwargs)
        
        def delete(self, url, **kwargs):
            kwargs.setdefault("headers", {}).update(self.headers)
            return client.delete(url, **kwargs)
    
    return AuthenticatedTestClient(access_token)


class TestIntegration:
    """Integration tests for the complete authentication flow"""
    
    def test_complete_auth_flow(self):
        """Test complete authentication flow from registration to API access"""
        # 1. Register new user
        register_data = {
            "email": "integration@example.com",
            "password": "integration123",
            "confirm_password": "integration123",
            "first_name": "Integration",
            "last_name": "Test"
        }
        
        register_response = client.post("/api/auth/register", json=register_data)
        assert register_response.status_code == 200
        
        # 2. Login with new user
        login_data = {
            "email": "integration@example.com",
            "password": "integration123"
        }
        
        login_response = client.post("/api/auth/login", json=login_data)
        assert login_response.status_code == 200
        
        tokens = login_response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        
        # 3. Access protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        profile_response = client.get("/api/auth/me", headers=headers)
        assert profile_response.status_code == 200
        
        profile = profile_response.json()
        assert profile["email"] == "integration@example.com"
        
        # 4. Refresh token
        refresh_data = {"refresh_token": refresh_token}
        refresh_response = client.post("/api/auth/refresh", json=refresh_data)
        assert refresh_response.status_code == 200
        
        new_tokens = refresh_response.json()
        new_access_token = new_tokens["access_token"]
        
        # 5. Use new token
        new_headers = {"Authorization": f"Bearer {new_access_token}"}
        new_profile_response = client.get("/api/auth/me", headers=new_headers)
        assert new_profile_response.status_code == 200
        
        # 6. Logout
        logout_response = client.post("/api/auth/logout", headers=new_headers)
        assert logout_response.status_code == 200 