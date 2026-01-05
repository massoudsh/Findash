"""
DEPRECATED: This file has been integrated into professional_auth.py

This module is kept for backward compatibility only.
All functionality is now available in:
- src.api.endpoints.professional_auth

Please update imports to use:
    from src.api.endpoints.professional_auth import router as auth_router
    
This file will be removed in a future version.

Authentication endpoints for Quantum Trading Matrix™
Provides login, logout, token refresh, and user management
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from src.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    api_key_manager,
    get_current_active_user,
    rate_limit_dependency
)
from src.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["Authentication (Deprecated)"])


# Pydantic models
class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserRegister(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    confirm_password: str = Field(..., min_length=8, max_length=128)
    
    def validate_passwords_match(self):
        """Validate that passwords match"""
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    def validate_passwords_match(self):
        """Validate that new passwords match"""
        if self.new_password != self.confirm_new_password:
            raise ValueError("New passwords do not match")
        return self


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)
    
    def validate_passwords_match(self):
        """Validate that passwords match"""
        if self.new_password != self.confirm_new_password:
            raise ValueError("Passwords do not match")
        return self


class APIKeyRequest(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response"""
    key_id: str
    api_key: str
    name: str
    description: Optional[str]
    created_at: str
    expires_at: Optional[str]


class UserProfile(BaseModel):
    """User profile response"""
    id: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str]


# Mock user database (replace with real database in production)
mock_users = {
    "demo@quantumtrading.com": {
        "id": "demo-user-123",
        "email": "demo@quantumtrading.com",
        "password_hash": hash_password("demo123"),
        "first_name": "Demo",
        "last_name": "User",
        "is_active": True,
        "is_verified": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None
    }
}


@router.post("/login", response_model=TokenResponse)
async def login(
    user_data: UserLogin,
    request: Request,
    _: bool = Depends(rate_limit_dependency)
):
    """
    DEPRECATED: Use /api/auth/credentials or /api/auth/login instead
    Delegates to professional_auth endpoints
    """
    # Import here to avoid circular dependencies
    from src.api.endpoints.professional_auth import authenticate_credentials, UserCredentials
    
    # Convert to professional_auth format
    credentials = UserCredentials(email=user_data.email, password=user_data.password)
    response = await authenticate_credentials(credentials, request)
    
    # Convert response format
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=response.message
        )
    
    return TokenResponse(
        access_token=response.access_token or response.token,
        refresh_token=response.refresh_token or "",
        expires_in=response.expires_in or (settings.auth.jwt_access_token_expire_minutes * 60)
    )


@router.post("/register", response_model=dict)
async def register(
    user_data: UserRegister,
    request: Request,
    _: bool = Depends(rate_limit_dependency)
):
    """
    DEPRECATED: Use /api/auth/register instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import register_user, UserRegistration
    
    registration = UserRegistration(
        email=user_data.email,
        password=user_data.password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        confirm_password=user_data.confirm_password
    )
    response = await register_user(registration, request, _)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.message
        )
    
    return {
        "message": response.message,
        "email": user_data.email
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    _: bool = Depends(rate_limit_dependency)
):
    """
    DEPRECATED: Use /api/auth/refresh instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import refresh_token_endpoint as professional_refresh
    
    response = await professional_refresh({"refresh_token": refresh_data.refresh_token}, _)
    
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=response.message
        )
    
    return TokenResponse(
        access_token=response.access_token or "",
        refresh_token=response.refresh_token or "",
        expires_in=response.expires_in or (settings.auth.jwt_access_token_expire_minutes * 60)
    )


@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_active_user)
):
    """
    DEPRECATED: Use /api/auth/logout instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import logout as professional_logout
    return await professional_logout(current_user)


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_active_user)
):
    """
    DEPRECATED: Use /api/auth/profile instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import get_user_profile, UserProfile as ProfessionalUserProfile
    
    profile = await get_user_profile(current_user)
    
    # Convert to legacy format
    return UserProfile(
        id=profile.id,
        email=profile.email,
        first_name=profile.first_name,
        last_name=profile.last_name,
        is_active=profile.is_active,
        is_verified=True,  # Professional auth doesn't track verification separately
        created_at=profile.created_at,
        last_login=profile.last_login
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    DEPRECATED: Use /api/auth/change-password instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import change_password as professional_change_password
    
    password_dict = {
        "current_password": password_data.current_password,
        "new_password": password_data.new_password,
        "confirm_password": password_data.confirm_new_password
    }
    
    return await professional_change_password(password_dict, current_user)


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    DEPRECATED: Use /api/auth/api-keys instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import create_api_key as professional_create_api_key
    
    result = await professional_create_api_key({
        "name": api_key_data.name,
        "description": api_key_data.description
    }, current_user)
    
    return APIKeyResponse(
        key_id=result["key_id"],
        api_key=result["api_key"],
        name=result["name"],
        description=api_key_data.description,
        created_at=result["created_at"],
        expires_at=None
    )


@router.post("/password-reset-request")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    _: bool = Depends(rate_limit_dependency)
):
    """
    DEPRECATED: Use /api/auth/password-reset-request instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import request_password_reset as professional_reset_request
    
    return await professional_reset_request({"email": reset_data.email}, _)


@router.post("/password-reset-confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    _: bool = Depends(rate_limit_dependency)
):
    """
    DEPRECATED: Use /api/auth/password-reset-confirm instead
    Delegates to professional_auth endpoint
    """
    from src.api.endpoints.professional_auth import confirm_password_reset as professional_reset_confirm
    
    reset_dict = {
        "token": reset_data.token,
        "new_password": reset_data.new_password,
        "confirm_password": reset_data.confirm_new_password
    }
    
    return await professional_reset_confirm(reset_dict, _) 