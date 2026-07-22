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
from sqlalchemy.orm import Session

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
from src.database.postgres_connection import get_db
from src.database.models import User
from src.database import crud as db_crud

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
    user_id: Optional[str] = None
    email: Optional[str] = None

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

# Professional user database - backed by the real `users` table in PostgreSQL
# (src/database/models.py User, via SQLAlchemy). Environment-based secure
# defaults are used only to seed the fixed demo accounts on first use.
import os
from src.core.config import get_settings

settings = get_settings()

# Fixed demo accounts, seeded into the real `users` table (not an in-memory
# store) the first time any of them is touched. Passwords come from env vars
# so they can be overridden per-deployment; secure hard-coded fallbacks are
# used only when the env var isn't set (e.g. local/dev).
_DEMO_USERS_SEED = [
    {
        "email": "admin@octopus.trading", "username": "admin",
        "password_env": "DEMO_ADMIN_PASSWORD", "default_password": "SecureAdmin2025!",
        "first_name": "System", "last_name": "Administrator",
        "role": "admin", "permissions": ["admin", "trade", "view_portfolio", "view_analytics", "manage_users"],
    },
    {
        "email": "trader@octopus.trading", "username": "trader",
        "password_env": "DEMO_TRADER_PASSWORD", "default_password": "TraderPro2025!",
        "first_name": "Professional", "last_name": "Trader",
        "role": "trader", "permissions": ["trade", "view_portfolio", "view_analytics"],
    },
    {
        "email": "demo@octopus.trading", "username": "demo",
        "password_env": "DEMO_USER_PASSWORD", "default_password": "DemoUser2025!",
        "first_name": "Demo", "last_name": "User",
        "role": "demo", "permissions": ["view_portfolio", "view_analytics"],
    },
]


def _ensure_demo_users(db: Session) -> None:
    """Idempotently seed the fixed demo accounts into the real `users` table
    if they don't exist yet. Cheap no-op after the first call per database."""
    created = False
    for seed in _DEMO_USERS_SEED:
        if db_crud.get_user_by_email(db, seed["email"]):
            continue
        password = os.getenv(seed["password_env"], seed["default_password"])
        db.add(User(
            username=seed["username"],
            email=seed["email"],
            password_hash=hash_password(password),
            first_name=seed["first_name"],
            last_name=seed["last_name"],
            role=seed["role"],
            permissions=seed["permissions"],
            is_active=True,
        ))
        created = True
    if created:
        db.commit()


def _derive_unique_username(db: Session, email: str) -> str:
    """Derive a unique `username` (required, unique column) from an email's
    local part, since the registration form only collects email/name."""
    base = (email.split("@")[0] or "user")[:50]
    username = base
    suffix = 1
    while db_crud.get_user_by_username(db, username):
        suffix += 1
        username = f"{base}{suffix}"[:64]
    return username


def _to_profile(user: User) -> "UserProfile":
    return UserProfile(
        id=str(user.id),
        email=user.email,
        first_name=user.first_name or "",
        last_name=user.last_name or "",
        is_active=bool(user.is_active),
        created_at=user.created_at.isoformat() if user.created_at else "",
        last_login=user.last_login.isoformat() if user.last_login else None,
        role=user.role or "trader",
        permissions=user.permissions or ["trade", "view_portfolio", "view_analytics"],
    )

@router.post("/credentials", response_model=AuthResponse, dependencies=[Depends(auth_rate_limit)])
async def authenticate_credentials(
    credentials: UserCredentials,
    request: Request,
    db: Session = Depends(get_db)
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
        # Find user in the real `users` table (seeding demo accounts on first use)
        _ensure_demo_users(db)
        user = db_crud.get_user_by_email(db, credentials.email)
        if not user:
            logger.warning(f"Authentication attempt with non-existent email: {credentials.email}")
            rate_limiter.record_failed_login(client_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Verify password
        if not verify_password(credentials.password, user.password_hash):
            logger.warning(f"Failed password attempt for user: {credentials.email}")
            rate_limiter.record_failed_login(client_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Check if user is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {credentials.email}")
            return AuthResponse(
                success=False,
                message="Account is disabled"
            )

        # Clear failed login attempts on successful login
        rate_limiter.clear_login_attempts(client_id)

        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()

        permissions = user.permissions or []
        user_id = str(user.id)

        # Create JWT tokens (access and refresh)
        access_token = create_access_token({
            "sub": user_id,
            "email": user.email,
            "role": user.role,
            "permissions": permissions
        })
        refresh_token = create_refresh_token({"sub": user_id})

        # Return user data for NextAuth.js
        user_profile = {
            "id": user_id,
            "email": user.email,
            "name": f"{user.first_name or ''} {user.last_name or ''}".strip(),
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role,
            "permissions": permissions,
            "token": access_token
        }

        logger.info(f"Successful authentication for user: {credentials.email}")
        
        return AuthResponse(
            success=True,
            user=user_profile,
            token=access_token,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.auth.jwt_access_token_expire_minutes * 60,
            message="Authentication successful"
        )

    except HTTPException:
        raise
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
    db: Session = Depends(get_db),
    _: bool = Depends(auth_rate_limit)
):
    """
    User registration endpoint
    """
    try:
        _ensure_demo_users(db)

        # Check if user already exists
        if db_crud.get_user_by_email(db, registration.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user in the real `users` table
        new_user = User(
            username=_derive_unique_username(db, registration.email),
            email=registration.email,
            password_hash=hash_password(registration.password),
            first_name=registration.first_name,
            last_name=registration.last_name,
            role="trader",
            permissions=["trade", "view_portfolio", "view_analytics"],
            is_active=True,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        logger.info(f"New user registered: {registration.email}")

        return AuthResponse(
            success=True,
            message="User created successfully",
            user_id=str(new_user.id),
            email=new_user.email
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return AuthResponse(
            success=False,
            message="Registration failed"
        )

@router.get("/users", response_model=list[UserProfile])
async def list_users(
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only)
    """
    # Check admin permissions
    if "admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    # Return user profiles
    return [_to_profile(user) for user in db.query(User).all()]

@router.get("/profile", response_model=UserProfile)
@router.get("/me", response_model=UserProfile)
async def get_user_profile(
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current user profile
    """
    user = db_crud.get_user_by_email(db, current_user.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return _to_profile(user)

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token_endpoint(
    refresh_token_data: Dict[str, str],
    db: Session = Depends(get_db),
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        try:
            user = db_crud.get_user_by_id(db, int(payload.user_id)) if payload.user_id else None
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

        if not user or not user.is_active:
            return AuthResponse(
                success=False,
                message="User not found or disabled"
            )

        permissions = user.permissions or []
        user_id = str(user.id)

        # Create new tokens
        access_token = create_access_token({
            "sub": user_id,
            "email": user.email,
            "role": user.role,
            "permissions": permissions
        })
        new_refresh_token = create_refresh_token({"sub": user_id})

        return AuthResponse(
            success=True,
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.auth.jwt_access_token_expire_minutes * 60,
            message="Token refreshed successfully"
        )

    except HTTPException:
        raise
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
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out"}

@router.post("/change-password")
async def change_password(
    password_data: Dict[str, str],
    current_user: dict = Depends(get_current_active_user),
    db: Session = Depends(get_db)
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

        user = db_crud.get_user_by_email(db, current_user.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify current password
        if not verify_password(current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        # Update password
        user.password_hash = hash_password(new_password)
        db.commit()

        logger.info(f"Password changed for user: {current_user.email}")
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
    api_key_data: Dict[str, Any],
    current_user: dict = Depends(get_current_active_user)
):
    """
    Create API key for user
    """
    try:
        name = api_key_data.get("name", "Default API Key")
        description = api_key_data.get("description", "")

        # Generate API key
        api_key = api_key_manager.generate_api_key(current_user.user_id, name)

        key_id = f"key_{current_user.user_id}_{len(name)}"

        return {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
            "description": description,
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
    db: Session = Depends(get_db),
    _: bool = Depends(auth_rate_limit)
):
    """
    Alternative login endpoint (compatible with auth.py)
    Returns standard token format with access_token and refresh_token
    """
    # Delegate to credentials endpoint
    response = await authenticate_credentials(credentials, request, db)
    return response 