"""
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
router = APIRouter(prefix="/auth", tags=["Authentication"])


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
    User login endpoint
    
    Returns JWT access and refresh tokens for authenticated users
    """
    # Find user (replace with database query in production)
    user = mock_users.get(user_data.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if user is active
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled"
        )
    
    # Create tokens
    token_data = {
        "sub": user["id"],
        "email": user["email"],
        "scopes": ["read", "write"]  # Add appropriate scopes
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": user["id"]})
    
    # Update last login (in production, update database)
    user["last_login"] = "2024-01-01T00:00:00Z"
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.auth.jwt_access_token_expire_minutes * 60
    )


@router.post("/register", response_model=dict)
async def register(
    user_data: UserRegister,
    request: Request,
    _: bool = Depends(rate_limit_dependency)
):
    """
    User registration endpoint
    
    Creates a new user account
    """
    # Validate passwords match
    user_data.validate_passwords_match()
    
    # Check if user already exists
    if user_data.email in mock_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user (in production, save to database)
    user_id = f"user-{len(mock_users) + 1}"
    mock_users[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "is_active": True,
        "is_verified": False,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None
    }
    
    return {
        "message": "User created successfully",
        "user_id": user_id,
        "email": user_data.email
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    _: bool = Depends(rate_limit_dependency)
):
    """
    Token refresh endpoint
    
    Issues new access token using refresh token
    """
    # Verify refresh token
    payload = verify_token(refresh_data.refresh_token, "refresh")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Find user (replace with database query)
    user = None
    for u in mock_users.values():
        if u["id"] == user_id:
            user = u
            break
    
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled"
        )
    
    # Create new access token
    token_data = {
        "sub": user["id"],
        "email": user["email"],
        "scopes": ["read", "write"]
    }
    
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token({"sub": user["id"]})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.auth.jwt_access_token_expire_minutes * 60
    )


@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_active_user)
):
    """
    User logout endpoint
    
    In a full implementation, this would invalidate the token
    """
    # In production, add token to blacklist or invalidate in database
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get current user profile
    """
    # Find user details (replace with database query)
    user = None
    for u in mock_users.values():
        if u["id"] == current_user["id"]:
            user = u
            break
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserProfile(
        id=user["id"],
        email=user["email"],
        first_name=user["first_name"],
        last_name=user["last_name"],
        is_active=user["is_active"],
        is_verified=user["is_verified"],
        created_at=user["created_at"],
        last_login=user["last_login"]
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Change user password
    """
    # Validate passwords match
    password_data.validate_passwords_match()
    
    # Find user
    user = None
    for u in mock_users.values():
        if u["id"] == current_user["id"]:
            user = u
            break
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify current password
    if not verify_password(password_data.current_password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password (in production, update database)
    user["password_hash"] = hash_password(password_data.new_password)
    
    return {"message": "Password changed successfully"}


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Create API key for user
    """
    # Generate API key
    api_key = api_key_manager.generate_api_key(current_user["id"], api_key_data.name)
    
    # In production, save to database with expiration
    key_id = f"key-{current_user['id']}-{len(api_key_data.name)}"
    
    return APIKeyResponse(
        key_id=key_id,
        api_key=api_key,
        name=api_key_data.name,
        description=api_key_data.description,
        created_at="2024-01-01T00:00:00Z",
        expires_at=None  # Calculate based on expires_in_days
    )


@router.post("/password-reset-request")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    _: bool = Depends(rate_limit_dependency)
):
    """
    Request password reset
    
    Sends password reset email (if user exists)
    """
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/password-reset-confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    _: bool = Depends(rate_limit_dependency)
):
    """
    Confirm password reset with token
    """
    # Validate passwords match
    reset_data.validate_passwords_match()
    
    # Verify reset token (implement token verification logic)
    # In production, verify token from database and update password
    
    return {"message": "Password reset successfully"} 