"""
پنل مدیریت — بک‌اند واقعی (issue #12)

مدیریت کاربران واقعی (نقش/فعال-غیرفعال) + audit log واقعی، به‌جای داده mock
قبلی در `frontend-nextjs/src/app/admin/page.tsx`. دسترسی محدود به role=admin.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.core.security import require_admin, TokenData
from src.database.postgres_connection import get_db
from src.database.models import User, Portfolio, Trade, AuditLog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["Admin Panel"])

VALID_ROLES = {"admin", "trader", "demo"}


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class AdminUserOut(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    role: str
    is_active: bool
    risk_tolerance: str
    permissions: List[str]
    last_login: Optional[str]
    created_at: Optional[str]
    total_trades: int
    portfolio_value: float


class UpdateUserRequest(BaseModel):
    role: Optional[str] = Field(None, description="admin | trader | demo")
    is_active: Optional[bool] = None


class AuditLogOut(BaseModel):
    id: int
    timestamp: str
    actor: str
    action: str
    target_type: Optional[str]
    target_id: Optional[str]
    detail: Optional[dict]
    ip_address: Optional[str]


def _log_action(db: Session, actor_id: int, action: str, target_type: str, target_id: str,
                 detail: dict, ip: Optional[str]) -> None:
    db.add(AuditLog(
        actor_user_id=actor_id, action=action, target_type=target_type,
        target_id=str(target_id), detail=detail, ip_address=ip,
    ))
    db.commit()


# ─────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────

@router.get("/users", response_model=List[AdminUserOut], summary="لیست کاربران (فقط ادمین)")
async def list_users(
    db: Session = Depends(get_db),
    _: TokenData = Depends(require_admin),
):
    # تعداد معامله و ارزش پرتفوی واقعی هر کاربر (نه mock)
    trade_counts = dict(
        db.query(Portfolio.user_id, func.count(Trade.id))
        .join(Trade, Trade.portfolio_id == Portfolio.id)
        .group_by(Portfolio.user_id)
        .all()
    )
    portfolio_values = dict(
        db.query(Portfolio.user_id, func.coalesce(func.sum(Portfolio.total_value), 0))
        .group_by(Portfolio.user_id)
        .all()
    )

    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        AdminUserOut(
            id=u.id,
            name=f"{u.first_name or ''} {u.last_name or ''}".strip() or u.username,
            email=u.email,
            phone=u.phone,
            role=u.role or "trader",
            is_active=bool(u.is_active),
            risk_tolerance=u.risk_tolerance or "moderate",
            permissions=u.permissions or [],
            last_login=u.last_login.isoformat() if u.last_login else None,
            created_at=u.created_at.isoformat() if u.created_at else None,
            total_trades=int(trade_counts.get(u.id, 0)),
            portfolio_value=float(portfolio_values.get(u.id, 0) or 0),
        )
        for u in users
    ]


@router.patch("/users/{user_id}", response_model=AdminUserOut, summary="تغییر نقش/وضعیت کاربر (فقط ادمین)")
async def update_user(
    user_id: int,
    body: UpdateUserRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: TokenData = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "کاربر پیدا نشد")

    if user_id == int(admin.user_id) and body.is_active is False:
        raise HTTPException(400, "نمی‌توانید حساب خودتان را غیرفعال کنید")

    changes = {}
    if body.role is not None:
        if body.role not in VALID_ROLES:
            raise HTTPException(400, f"نقش نامعتبر — یکی از {sorted(VALID_ROLES)}")
        changes["role"] = {"from": user.role, "to": body.role}
        user.role = body.role
    if body.is_active is not None:
        changes["is_active"] = {"from": user.is_active, "to": body.is_active}
        user.is_active = body.is_active

    if changes:
        db.commit()
        db.refresh(user)
        _log_action(
            db, int(admin.user_id), "user.updated", "user", str(user.id),
            changes, request.client.host if request.client else None,
        )

    trades = db.query(func.count(Trade.id)).join(Portfolio, Trade.portfolio_id == Portfolio.id).filter(Portfolio.user_id == user.id).scalar() or 0
    pf_value = db.query(func.coalesce(func.sum(Portfolio.total_value), 0)).filter(Portfolio.user_id == user.id).scalar() or 0

    return AdminUserOut(
        id=user.id,
        name=f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username,
        email=user.email,
        phone=user.phone,
        role=user.role or "trader",
        is_active=bool(user.is_active),
        risk_tolerance=user.risk_tolerance or "moderate",
        permissions=user.permissions or [],
        last_login=user.last_login.isoformat() if user.last_login else None,
        created_at=user.created_at.isoformat() if user.created_at else None,
        total_trades=int(trades),
        portfolio_value=float(pf_value),
    )


# ─────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────

@router.get("/audit-log", response_model=List[AuditLogOut], summary="لاگ ممیزی واقعی (فقط ادمین)")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: TokenData = Depends(require_admin),
):
    rows = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        AuditLogOut(
            id=r.id,
            timestamp=r.created_at.isoformat() if r.created_at else "",
            actor=(r.actor.email if r.actor else "system"),
            action=r.action,
            target_type=r.target_type,
            target_id=r.target_id,
            detail=r.detail,
            ip_address=r.ip_address,
        )
        for r in rows
    ]
