"""
Security & Access Control API Endpoints
API key management, session tracking, and trading permissions
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.postgres_connection import get_db
from src.database.models import APIKey, UserSession, TradingPermission
from src.core.config import get_settings
from src.core.security import get_current_active_user, TokenData, hash_password
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/security", tags=["Security & Access Control"])

# Pydantic models
class APIKey(BaseModel):
    id: str
    name: str
    key: Optional[str] = None  # Only returned on creation
    key_preview: str
    permissions: List[str]
    created_at: str
    last_used: str
    status: str  # 'active', 'revoked', 'expired'
    ip_whitelist: Optional[List[str]] = None

class Session(BaseModel):
    id: str
    device: str
    browser: str
    location: str
    ip_address: str
    login_time: str
    last_activity: str
    status: str  # 'active', 'expired', 'revoked'
    current: bool

class TradingPermission(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    restrictions: Optional[Dict[str, Any]] = None

class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(..., description="List of permissions: read, trade, withdraw, admin")
    ip_whitelist: Optional[List[str]] = None

class UpdatePermissionRequest(BaseModel):
    enabled: bool
    restrictions: Optional[Dict[str, Any]] = None

@router.get("/api-keys", response_model=List[APIKey])
async def get_api_keys(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Get all API keys for the current user
    """
    try:
        # In production, fetch from database
        api_keys = [
            APIKey(
                id='1',
                name='Trading Bot API',
                key_preview='sk_live_***xyz789',
                permissions=['read', 'trade', 'withdraw'],
                created_at=(datetime.utcnow() - timedelta(days=30)).isoformat(),
                last_used=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
                status='active',
                ip_whitelist=['192.168.1.100', '10.0.0.5']
            ),
            APIKey(
                id='2',
                name='Read-Only Analytics',
                key_preview='sk_live_***uvw012',
                permissions=['read'],
                created_at=(datetime.utcnow() - timedelta(days=60)).isoformat(),
                last_used=(datetime.utcnow() - timedelta(days=1)).isoformat(),
                status='active'
            ),
            APIKey(
                id='3',
                name='Legacy Integration',
                key_preview='sk_live_***rst345',
                permissions=['read', 'trade'],
                created_at=(datetime.utcnow() - timedelta(days=180)).isoformat(),
                last_used=(datetime.utcnow() - timedelta(days=90)).isoformat(),
                status='expired'
            )
        ]
        
        if status:
            api_keys = [k for k in api_keys if k.status == status]
        
        return api_keys
    except Exception as e:
        logger.error(f"Error fetching API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api-keys", response_model=APIKey)
async def create_api_key(
    request: CreateAPIKeyRequest,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Create a new API key
    """
    try:
        # Validate permissions
        valid_permissions = ['read', 'trade', 'withdraw', 'admin']
        for perm in request.permissions:
            if perm not in valid_permissions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid permission: {perm}. Valid permissions: {valid_permissions}"
                )
        
        # Generate API key
        key_id = f"key_{datetime.utcnow().timestamp()}"
        api_key = f"sk_live_{secrets.token_urlsafe(16)}"
        key_preview = f"sk_live_***{api_key[-6:]}"
        
        # In production, store in database with hashed key
        new_key = APIKey(
            id=key_id,
            name=request.name,
            key=api_key,  # Only returned on creation
            key_preview=key_preview,
            permissions=request.permissions,
            created_at=datetime.utcnow().isoformat(),
            last_used=datetime.utcnow().isoformat(),
            status='active',
            ip_whitelist=request.ip_whitelist
        )
        
        return new_key
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Revoke an API key
    """
    try:
        # In production, update status in database
        return {
            "success": True,
            "message": f"API key {key_id} revoked successfully"
        }
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[Session])
async def get_sessions(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Get active sessions for the current user
    """
    try:
        # In production, fetch from database
        sessions = [
            Session(
                id='1',
                device='MacBook Pro',
                browser='Chrome 120.0',
                location='San Francisco, CA',
                ip_address='192.168.1.100',
                login_time=(datetime.utcnow() - timedelta(hours=1)).isoformat(),
                last_activity=(datetime.utcnow() - timedelta(minutes=1)).isoformat(),
                status='active',
                current=True
            ),
            Session(
                id='2',
                device='iPhone 15 Pro',
                browser='Safari Mobile',
                location='San Francisco, CA',
                ip_address='192.168.1.101',
                login_time=(datetime.utcnow() - timedelta(days=1)).isoformat(),
                last_activity=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
                status='active',
                current=False
            ),
            Session(
                id='3',
                device='Windows PC',
                browser='Firefox 121.0',
                location='New York, NY',
                ip_address='203.0.113.45',
                login_time=(datetime.utcnow() - timedelta(days=7)).isoformat(),
                last_activity=(datetime.utcnow() - timedelta(days=6)).isoformat(),
                status='expired',
                current=False
            )
        ]
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        return sessions
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Revoke a specific session
    """
    try:
        # In production, update session status in database
        return {
            "success": True,
            "message": f"Session {session_id} revoked successfully"
        }
    except Exception as e:
        logger.error(f"Error revoking session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/revoke-all")
async def revoke_all_sessions(
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Revoke all sessions except the current one
    """
    try:
        # In production, update all sessions except current
        return {
            "success": True,
            "message": "All other sessions revoked successfully"
        }
    except Exception as e:
        logger.error(f"Error revoking all sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/permissions", response_model=List[TradingPermission])
async def get_trading_permissions(
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Get trading permissions and restrictions
    """
    try:
        # In production, fetch from database
        permissions = [
            TradingPermission(
                id='1',
                name='Day Trading',
                description='Allow day trading with position limits',
                enabled=True,
                restrictions={
                    'max_order_size': 10000,
                    'max_positions': 10,
                    'trading_hours': '09:30-16:00 EST'
                }
            ),
            TradingPermission(
                id='2',
                name='Options Trading',
                description='Enable options trading capabilities',
                enabled=True,
                restrictions={
                    'max_order_size': 5000,
                    'allowed_symbols': ['SPY', 'QQQ', 'AAPL', 'TSLA']
                }
            ),
            TradingPermission(
                id='3',
                name='Margin Trading',
                description='Allow margin trading with leverage',
                enabled=False,
                restrictions={
                    'max_order_size': 50000,
                    'max_positions': 20
                }
            ),
            TradingPermission(
                id='4',
                name='Crypto Trading',
                description='Enable cryptocurrency trading',
                enabled=True,
                restrictions={
                    'max_order_size': 25000,
                    'allowed_symbols': ['BTC-USD', 'ETH-USD', 'SOL-USD']
                }
            )
        ]
        return permissions
    except Exception as e:
        logger.error(f"Error fetching trading permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/permissions/{permission_id}")
async def update_trading_permission(
    permission_id: str,
    request: UpdatePermissionRequest,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_active_user)
):
    """
    Update a trading permission
    """
    try:
        # In production, update in database
        return {
            "success": True,
            "message": f"Permission {permission_id} updated successfully",
            "permission": {
                "id": permission_id,
                "enabled": request.enabled,
                "restrictions": request.restrictions
            }
        }
    except Exception as e:
        logger.error(f"Error updating trading permission: {e}")
        raise HTTPException(status_code=500, detail=str(e))

