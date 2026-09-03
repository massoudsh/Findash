"""
موتور سیاست ریسک (issue #22)

هر کاربر یک سیاست ریسک دارد (حداکثر افت روزانه مجاز + حداکثر تمرکز روی یک دارایی).
ارزیابی واقعی روی داده‌های واقعی پرتفوی (`PortfolioSnapshot.daily_pnl_percent` و
`Position.market_value`) انجام می‌شود — نه شبیه‌سازی. اگر نقض رخ دهد:
  - همیشه یک `RiskPolicyBreach` ثبت و یک اعلان درون‌برنامه‌ای ارسال می‌شود.
  - اگر `action_on_breach == 'stop_bots'`، تمام ربات‌های فعال کاربر متوقف می‌شوند
    (از همان persistence JSON که `trading_bots.py` استفاده می‌کند).
ارزیابی دوره‌ای همه‌ی کاربران در `src/notifications/tasks.py` (Celery beat) انجام
می‌شود؛ `POST /me/evaluate` هم برای بررسی فوری پرتفوی خود کاربر در دسترس است.
"""

import logging
from datetime import datetime, date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user, require_admin, TokenData
from src.database.postgres_connection import get_db
from src.database.models import RiskPolicy, RiskPolicyBreach, Portfolio, PortfolioSnapshot, Position
from src.services.notifications import create_in_app_notification
from src.core.persian_utils import to_persian_digits

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/risk-policy", tags=["Risk Policy"])

VALID_ACTIONS = {"alert", "stop_bots"}
RULE_LABELS_FA = {
    "max_daily_drawdown": "افت روزانه",
    "max_position_concentration": "تمرکز روی یک دارایی",
}


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class RiskPolicyOut(BaseModel):
    max_daily_drawdown_pct: float
    max_position_concentration_pct: float
    action_on_breach: str
    enabled: bool
    updated_at: Optional[str] = None


class RiskPolicyUpdate(BaseModel):
    max_daily_drawdown_pct: Optional[float] = Field(None, gt=0, le=100)
    max_position_concentration_pct: Optional[float] = Field(None, gt=0, le=100)
    action_on_breach: Optional[str] = None
    enabled: Optional[bool] = None


class RiskPolicyBreachOut(BaseModel):
    id: int
    rule: str
    value: float
    threshold: float
    action_taken: str
    created_at: Optional[str] = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_or_create_policy(db: Session, user_id: int) -> RiskPolicy:
    policy = db.query(RiskPolicy).filter(RiskPolicy.user_id == user_id).first()
    if not policy:
        policy = RiskPolicy(user_id=user_id)
        db.add(policy)
        db.commit()
        db.refresh(policy)
    return policy


def _stop_user_bots(user_id: int) -> int:
    """ربات‌های فعال کاربر را متوقف می‌کند. برمی‌گرداند: تعداد ربات متوقف‌شده."""
    try:
        from src.api.bots_persistence import load_bots, save_bots
    except ImportError:
        return 0
    db_bots = load_bots()
    stopped = 0
    uid = str(user_id)
    for bot in db_bots.values():
        if bot.get("user_id") == uid and bot.get("status") == "active":
            bot["status"] = "stopped"
            bot["updated_at"] = datetime.utcnow().isoformat()
            stopped += 1
    if stopped:
        save_bots(db_bots)
    return stopped


def _record_breach_once(db: Session, policy: RiskPolicy, rule: str, value: float, threshold: float) -> bool:
    """اگر امروز قبلاً همین نوع نقض برای این کاربر ثبت نشده باشد، ثبت می‌کند (جلوگیری از اسپم)."""
    start_of_day = datetime.combine(date.today(), datetime.min.time())
    existing = db.query(RiskPolicyBreach).filter(
        RiskPolicyBreach.user_id == policy.user_id,
        RiskPolicyBreach.rule == rule,
        RiskPolicyBreach.created_at >= start_of_day,
    ).first()
    if existing:
        return False

    db.add(RiskPolicyBreach(
        user_id=policy.user_id, rule=rule, value=value, threshold=threshold,
        action_taken=policy.action_on_breach,
    ))
    rule_fa = RULE_LABELS_FA.get(rule, rule)
    create_in_app_notification(
        db, policy.user_id, "risk",
        "نقض سیاست ریسک",
        to_persian_digits(f"{rule_fa} به {value:.2f}٪ رسید (آستانه: {threshold:.2f}٪)"),
    )
    if policy.action_on_breach == "stop_bots":
        n = _stop_user_bots(policy.user_id)
        if n:
            create_in_app_notification(
                db, policy.user_id, "risk", "ربات‌های معاملاتی متوقف شدند",
                f"به‌دلیل نقض سیاست ریسک، {n} ربات فعال متوقف شد.",
            )
    db.commit()
    return True


def evaluate_risk_policy_for_user(db: Session, user_id: int) -> int:
    """پرتفوی‌های فعال یک کاربر را با سیاست ریسک او مقایسه می‌کند. برمی‌گرداند: تعداد نقض جدید ثبت‌شده."""
    policy = db.query(RiskPolicy).filter(RiskPolicy.user_id == user_id, RiskPolicy.enabled == True).first()
    if not policy:
        return 0

    breach_count = 0
    portfolios = db.query(Portfolio).filter(Portfolio.user_id == user_id, Portfolio.is_active == True).all()
    for pf in portfolios:
        snap = (
            db.query(PortfolioSnapshot)
            .filter(PortfolioSnapshot.portfolio_id == pf.id)
            .order_by(PortfolioSnapshot.snapshot_date.desc())
            .first()
        )
        if snap and snap.daily_pnl_percent is not None:
            dd = float(snap.daily_pnl_percent)
            if dd <= -float(policy.max_daily_drawdown_pct):
                if _record_breach_once(db, policy, "max_daily_drawdown", abs(dd), float(policy.max_daily_drawdown_pct)):
                    breach_count += 1

        total_value = float(pf.total_value or 0)
        if total_value > 0:
            positions = db.query(Position).filter(Position.portfolio_id == pf.id, Position.is_active == True).all()
            for pos in positions:
                pct = float(pos.market_value or 0) / total_value * 100
                if pct > float(policy.max_position_concentration_pct):
                    if _record_breach_once(db, policy, "max_position_concentration", pct, float(policy.max_position_concentration_pct)):
                        breach_count += 1

    return breach_count


def evaluate_all_risk_policies(db: Session) -> int:
    """برای Celery beat — همه‌ی کاربرانی که سیاست ریسک فعال دارند."""
    total = 0
    user_ids = [row[0] for row in db.query(RiskPolicy.user_id).filter(RiskPolicy.enabled == True).all()]
    for uid in user_ids:
        total += evaluate_risk_policy_for_user(db, uid)
    return total


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.get("/me", response_model=RiskPolicyOut, summary="سیاست ریسک کاربر جاری")
async def get_my_policy(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    policy = _get_or_create_policy(db, int(current_user.user_id))
    return _to_out(policy)


@router.put("/me", response_model=RiskPolicyOut, summary="به‌روزرسانی سیاست ریسک کاربر جاری")
async def update_my_policy(
    body: RiskPolicyUpdate,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if body.action_on_breach is not None and body.action_on_breach not in VALID_ACTIONS:
        raise HTTPException(400, f"action_on_breach باید یکی از {sorted(VALID_ACTIONS)} باشد")

    policy = _get_or_create_policy(db, int(current_user.user_id))
    if body.max_daily_drawdown_pct is not None:
        policy.max_daily_drawdown_pct = body.max_daily_drawdown_pct
    if body.max_position_concentration_pct is not None:
        policy.max_position_concentration_pct = body.max_position_concentration_pct
    if body.action_on_breach is not None:
        policy.action_on_breach = body.action_on_breach
    if body.enabled is not None:
        policy.enabled = body.enabled
    db.commit()
    db.refresh(policy)
    return _to_out(policy)


@router.get("/breaches", response_model=List[RiskPolicyBreachOut], summary="تاریخچه نقض سیاست ریسک کاربر جاری")
async def list_breaches(
    limit: int = 50,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(RiskPolicyBreach)
        .filter(RiskPolicyBreach.user_id == int(current_user.user_id))
        .order_by(RiskPolicyBreach.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [
        RiskPolicyBreachOut(
            id=r.id, rule=r.rule, value=float(r.value), threshold=float(r.threshold),
            action_taken=r.action_taken, created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]


@router.post("/me/evaluate", summary="بررسی فوری سیاست ریسک پرتفوی خودم")
async def evaluate_my_policy(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    count = evaluate_risk_policy_for_user(db, int(current_user.user_id))
    return {"new_breaches": count}


@router.post("/evaluate-all", summary="اجرای دستی ارزیابی سیاست ریسک همه کاربران (فقط ادمین)")
async def evaluate_all(
    db: Session = Depends(get_db),
    _: TokenData = Depends(require_admin),
):
    count = evaluate_all_risk_policies(db)
    return {"new_breaches": count}


def _to_out(p: RiskPolicy) -> RiskPolicyOut:
    return RiskPolicyOut(
        max_daily_drawdown_pct=float(p.max_daily_drawdown_pct),
        max_position_concentration_pct=float(p.max_position_concentration_pct),
        action_on_breach=p.action_on_breach, enabled=p.enabled,
        updated_at=p.updated_at.isoformat() if p.updated_at else None,
    )
