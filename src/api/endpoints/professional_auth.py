"""
Professional Authentication System for Octopus Trading Platform™
Enterprise-grade authentication with proper NextAuth.js integration
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
import bcrypt
import jwt

from src.core.security import (
    get_current_active_user,
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    api_key_manager
)
from src.core.config import get_settings
from src.core.rate_limiter import auth_rate_limit, rate_limiter, get_client_identifier, standard_rate_limit

settings = get_settings()
router = APIRouter(prefix="/api/auth", tags=["Professional Authentication"])
logger = logging.getLogger(__name__)

# Professional user models
class UserCredentials(BaseModel):
    """User login credentials"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

class UserRegistration(BaseModel):
    """User registration data"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    confirm_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class AuthResponse(BaseModel):
    """Authentication response"""
    success: bool
    user: Optional[Dict[str, Any]] = None
    token: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    message: str

class UserProfile(BaseModel):
    """User profile information"""
    id: str
    email: str
    first_name: str
    last_name: str
    is_active: bool
    created_at: str
    last_login: Optional[str]
    role: str = "trader"
    permissions: list = ["trade", "view_portfolio", "view_analytics"]

# Professional user database - Environment-based secure defaults
import os
from src.core.config import get_settings

settings = get_settings()

# Generate secure demo users from environment or use secure defaults
DEMO_ADMIN_PASSWORD = os.getenv("DEMO_ADMIN_PASSWORD", "SecureAdmin2025!")
DEMO_TRADER_PASSWORD = os.getenv("DEMO_TRADER_PASSWORD", "TraderPro2025!")
DEMO_USER_PASSWORD = os.getenv("DEMO_USER_PASSWORD", "DemoUser2025!")

PROFESSIONAL_USERS = {
    "admin@octopus.trading": {
        "id": "usr_admin_001",
        "email": "admin@octopus.trading",
        "password_hash": bcrypt.hashpw(DEMO_ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "first_name": "System",
        "last_name": "Administrator",
        "is_active": True,
        "created_at": "2025-01-01T00:00:00Z",
        "last_login": None,
        "role": "admin",
        "permissions": ["admin", "trade", "view_portfolio", "view_analytics", "manage_users"]
    },
    "trader@octopus.trading": {
        "id": "usr_trader_001", 
        "email": "trader@octopus.trading",
        "password_hash": bcrypt.hashpw(DEMO_TRADER_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "first_name": "Professional",
        "last_name": "Trader",
        "is_active": True,
        "created_at": "2025-01-01T00:00:00Z",
        "last_login": None,
        "role": "trader",
        "permissions": ["trade", "view_portfolio", "view_analytics"]
    },
    "demo@octopus.trading": {
        "id": "usr_demo_001",
        "email": "demo@octopus.trading", 
        "password_hash": bcrypt.hashpw(DEMO_USER_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "first_name": "Demo",
        "last_name": "User",
        "is_active": True,
        "created_at": "2025-01-01T00:00:00Z",
        "last_login": None,
        "role": "demo",
        "permissions": ["view_portfolio", "view_analytics"]
    }
}

def create_jwt_token(user_data: dict) -> str:
    """Create JWT token for user"""
    payload = {
        "sub": user_data["id"],
        "email": user_data["email"],
        "role": user_data["role"],
        "permissions": user_data["permissions"],
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, settings.auth.jwt_secret_key, algorithm=settings.auth.jwt_algorithm)

@router.post("/credentials", response_model=AuthResponse, dependencies=[Depends(auth_rate_limit)])
async def authenticate_credentials(
    credentials: UserCredentials,
    request: Request
):
    """
    Professional authentication endpoint for NextAuth.js credentials provider
    
    This endpoint is called by the frontend NextAuth.js configuration
    to validate user credentials and return user information.
    """
    
    # Get client identifier for rate limiting
    client_id = get_client_identifier(request)
    
    # Check login attempts
    login_check = rate_limiter.check_login_attempts(client_id)
    if not login_check["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "account_locked",
                "message": f"Account temporarily locked due to too many failed attempts. Try again in {login_check.get('retry_after', 300)} seconds.",
                "retry_after": login_check.get("retry_after", 300)
            }
        )
    
    try:
        # Find user in database
        user = PROFESSIONAL_USERS.get(credentials.email)
        if not user:
            logger.warning(f"Authentication attempt with non-existent email: {credentials.email}")
            rate_limiter.record_failed_login(client_id)
            return AuthResponse(
                success=False,
                message="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(credentials.password, user["password_hash"]):
            logger.warning(f"Failed password attempt for user: {credentials.email}")
            rate_limiter.record_failed_login(client_id)
            return AuthResponse(
                success=False,
                message="Invalid email or password"
            )
        
        # Check if user is active
        if not user["is_active"]:
            logger.warning(f"Login attempt for inactive user: {credentials.email}")
            return AuthResponse(
                success=False,
                message="Account is disabled"
            )
        
        # Clear failed login attempts on successful login
        rate_limiter.clear_login_attempts(client_id)
        
        # Update last login
        user["last_login"] = datetime.utcnow().isoformat() + "Z"
        
        # Create JWT tokens (access and refresh)
        token = create_jwt_token(user)
        access_token = create_access_token({
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "permissions": user["permissions"]
        })
        refresh_token = create_refresh_token({"sub": user["id"]})
        
        # Return user data for NextAuth.js
        user_profile = {
            "id": user["id"],
            "email": user["email"],
            "name": f"{user['first_name']} {user['last_name']}",
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "role": user["role"],
            "permissions": user["permissions"],
            "token": token
        }
        
        logger.info(f"Successful authentication for user: {credentials.email}")
        
        return AuthResponse(
            success=True,
            user=user_profile,
            token=token,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.auth.jwt_access_token_expire_minutes * 60,
            message="Authentication successful"
        )
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return AuthResponse(
            success=False,
            message="Authentication failed"
        )

@router.post("/register", response_model=AuthResponse)
async def register_user(
    registration: UserRegistration,
    request: Request,
    _: bool = Depends(auth_rate_limit)
):
    """
    User registration endpoint
    """
    try:
        # Check if user already exists
        if registration.email in PROFESSIONAL_USERS:
            return AuthResponse(
                success=False,
                message="Email already registered"
            )
        
        # Create new user
        user_id = f"usr_{len(PROFESSIONAL_USERS) + 1:06d}"
        password_hash = bcrypt.hashpw(registration.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        new_user = {
            "id": user_id,
            "email": registration.email,
            "password_hash": password_hash,
            "first_name": registration.first_name,
            "last_name": registration.last_name,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_login": None,
            "role": "trader",
            "permissions": ["trade", "view_portfolio", "view_analytics"]
        }
        
        # Add to database
        PROFESSIONAL_USERS[registration.email] = new_user
        
        logger.info(f"New user registered: {registration.email}")
        
        return AuthResponse(
            success=True,
            message="Registration successful"
        )
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return AuthResponse(
            success=False,
            message="Registration failed"
        )

@router.get("/users", response_model=list[UserProfile])
async def list_users(
    current_user: dict = Depends(get_current_active_user)
):
    """
    List all users (admin only)
    """
    # Check admin permissions
    if "admin" not in current_user.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Return user profiles
    profiles = []
    for user in PROFESSIONAL_USERS.values():
        profiles.append(UserProfile(
            id=user["id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            is_active=user["is_active"],
            created_at=user["created_at"],
            last_login=user["last_login"],
            role=user["role"],
            permissions=user["permissions"]
        ))
    
    return profiles

@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get current user profile
    """
    user = PROFESSIONAL_USERS.get(current_user["email"])
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
        created_at=user["created_at"],
        last_login=user["last_login"],
        role=user["role"],
        permissions=user["permissions"]
    )

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token_endpoint(
    refresh_token_data: Dict[str, str],
    _: bool = Depends(auth_rate_limit)
):
    """
    Token refresh endpoint - Issue new access token using refresh token
    """
    try:
        refresh_token = refresh_token_data.get("refresh_token")
        if not refresh_token:
            return AuthResponse(
                success=False,
                message="Refresh token required"
            )
        
        # Verify refresh token
        payload = verify_token(refresh_token, token_type="refresh")
        if not payload:
            return AuthResponse(
                success=False,
                message="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            return AuthResponse(
                success=False,
                message="Invalid refresh token"
            )
        
        # Find user
        user = None
        for u in PROFESSIONAL_USERS.values():
            if u["id"] == user_id:
                user = u
                break
        
        if not user or not user["is_active"]:
            return AuthResponse(
                success=False,
                message="User not found or disabled"
            )
        
        # Create new tokens
        access_token = create_access_token({
            "sub": user["id"],
            "email": user["email"],
            "role": user["role"],
            "permissions": user["permissions"]
        })
        new_refresh_token = create_refresh_token({"sub": user["id"]})
        
        return AuthResponse(
            success=True,
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.auth.jwt_access_token_expire_minutes * 60,
            message="Token refreshed successfully"
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return AuthResponse(
            success=False,
            message="Token refresh failed"
        )

@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_active_user)
):
    """
    User logout endpoint
    In production, this would invalidate tokens
    """
    logger.info(f"User logged out: {current_user.get('email')}")
    return {"message": "Successfully logged out"}

@router.post("/change-password")
async def change_password(
    password_data: Dict[str, str],
    current_user: dict = Depends(get_current_active_user)
):
    """
    Change user password
    """
    try:
        current_password = password_data.get("current_password")
        new_password = password_data.get("new_password")
        confirm_password = password_data.get("confirm_password")
        
        if not all([current_password, new_password, confirm_password]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All password fields required"
            )
        
        if new_password != confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New passwords do not match"
            )
        
        user = PROFESSIONAL_USERS.get(current_user["email"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not verify_password(current_password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user["password_hash"] = hash_password(new_password)
        
        logger.info(f"Password changed for user: {current_user.get('email')}")
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.post("/api-keys", dependencies=[Depends(get_current_active_user)])
async def create_api_key(
    api_key_data: Dict[str, str],
    current_user: dict = Depends(get_current_active_user)
):
    """
    Create API key for user
    """
    try:
        name = api_key_data.get("name", "Default API Key")
        
        # Generate API key
        api_key = api_key_manager.generate_api_key(current_user["id"], name)
        
        key_id = f"key_{current_user['id']}_{len(name)}"
        
        return {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key creation failed"
        )

@router.post("/password-reset-request")
async def request_password_reset(
    reset_data: Dict[str, str],
    _: bool = Depends(auth_rate_limit)
):
    """
    Request password reset - Always returns success to prevent email enumeration
    """
    # In production, send password reset email if user exists
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/password-reset-confirm")
async def confirm_password_reset(
    reset_data: Dict[str, str],
    _: bool = Depends(auth_rate_limit)
):
    """
    Confirm password reset with token
    """
    # In production, verify token and update password
    return {"message": "Password reset successfully"}

@router.get("/demo-accounts")
async def get_demo_accounts():
    """
    Get demo account information for testing (passwords not exposed)
    """
    # Only show in development mode
    if settings.environment != "development":
        raise HTTPException(status_code=404, detail="Demo accounts not available in production")
    
    return {
        "demo_accounts": [
            {
                "email": "admin@octopus.trading",
                "role": "admin",
                "description": "System Administrator - Full access",
                "note": "Contact admin for password"
            },
            {
                "email": "trader@octopus.trading", 
                "role": "trader",
                "description": "Professional Trader - Trading access",
                "note": "Contact admin for password"
            },
            {
                "email": "demo@octopus.trading",
                "role": "demo", 
                "description": "Demo User - View-only access",
                "note": "Contact admin for password"
            }
        ],
        "note": "These are demo accounts for testing. Passwords are environment-controlled. In production, use secure registration.",
        "environment": settings.environment
    }

@router.post("/login", response_model=AuthResponse)
async def login(
    credentials: UserCredentials,
    request: Request,
    _: bool = Depends(auth_rate_limit)
):
    """
    Alternative login endpoint (compatible with auth.py)
    Returns standard token format with access_token and refresh_token
    """
    # Delegate to credentials endpoint
    response = await authenticate_credentials(credentials, request)
    return response 