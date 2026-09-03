"""
هشدار قیمت با ارسال Push/SMS/درون‌برنامه‌ای (issue #19)

کاربر یک قانون هشدار (نماد + جهت + قیمت هدف) ثبت می‌کند. ارزیابی واقعی قیمت‌های
لحظه‌ای (از همان منبع `iran_market.get_overview`) به‌صورت دوره‌ای در یک Celery task
(`src/notifications/tasks.py`) انجام می‌شود؛ `POST /evaluate` هم برای اجرای دستی/تست
(فقط ادمین) در دسترس است. کانال ارسال واقعی از `src/services/notifications.py`
استفاده می‌کند (که خودش اگر SMS_API_KEY/VAPID تنظیم نشده باشد، فقط لاگ می‌کند).
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user, require_admin, TokenData
from src.database.postgres_connection import get_db
from src.database.models import PriceAlertRule, PushSubscription, User
from src.services.notifications import create_in_app_notification, send_sms_text, send_web_push
from src.core.persian_utils import to_persian_digits

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alerts", tags=["Price Alerts"])

VALID_CHANNELS = {"in_app", "push", "sms"}
VALID_DIRECTIONS = {"above", "below"}


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class PriceAlertRuleCreate(BaseModel):
    symbol: str = Field(..., description="نماد از /api/iran-market/overview مثل BTC-IRT یا USD-IRR")
    direction: str = Field(..., description="above | below")
    target_price: float = Field(..., gt=0)
    channels: List[str] = Field(default_factory=lambda: ["in_app"])
    note: Optional[str] = Field(None, max_length=255)

    @field_validator("direction")
    @classmethod
    def _dir(cls, v: str) -> str:
        if v not in VALID_DIRECTIONS:
            raise ValueError(f"direction باید یکی از {sorted(VALID_DIRECTIONS)} باشد")
        return v

    @field_validator("channels")
    @classmethod
    def _chan(cls, v: List[str]) -> List[str]:
        bad = set(v) - VALID_CHANNELS
        if bad:
            raise ValueError(f"کانال نامعتبر: {bad} — مجاز: {sorted(VALID_CHANNELS)}")
        return v or ["in_app"]


class PriceAlertRuleOut(BaseModel):
    id: int
    symbol: str
    direction: str
    target_price: float
    channels: List[str]
    note: Optional[str] = None
    is_active: bool
    triggered_at: Optional[str] = None
    created_at: Optional[str] = None


class PushSubscribeRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth: str


# ─────────────────────────────────────────────
# CRUD قانون هشدار
# ─────────────────────────────────────────────

@router.post("/rules", response_model=PriceAlertRuleOut, summary="ایجاد قانون هشدار قیمت")
async def create_rule(
    body: PriceAlertRuleCreate,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    rule = PriceAlertRule(
        user_id=int(current_user.user_id),
        symbol=body.symbol.upper(),
        direction=body.direction,
        target_price=body.target_price,
        channels=body.channels,
        note=body.note,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return _to_out(rule)


@router.get("/rules", response_model=List[PriceAlertRuleOut], summary="لیست قوانین هشدار کاربر جاری")
async def list_rules(
    include_triggered: bool = False,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    q = db.query(PriceAlertRule).filter(PriceAlertRule.user_id == int(current_user.user_id))
    if not include_triggered:
        q = q.filter(PriceAlertRule.is_active == True)
    rules = q.order_by(PriceAlertRule.created_at.desc()).all()
    return [_to_out(r) for r in rules]


@router.delete("/rules/{rule_id}", summary="حذف/لغو قانون هشدار")
async def delete_rule(
    rule_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    rule = db.query(PriceAlertRule).filter(
        PriceAlertRule.id == rule_id, PriceAlertRule.user_id == int(current_user.user_id)
    ).first()
    if not rule:
        raise HTTPException(404, "قانون هشدار پیدا نشد")
    db.delete(rule)
    db.commit()
    return {"success": True}


# ─────────────────────────────────────────────
# Web Push subscription
# ─────────────────────────────────────────────

@router.post("/push-subscribe", summary="ثبت اشتراک Web Push مرورگر")
async def push_subscribe(
    body: PushSubscribeRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    existing = db.query(PushSubscription).filter(PushSubscription.endpoint == body.endpoint).first()
    if existing:
        existing.user_id = int(current_user.user_id)
        existing.p256dh_key = body.p256dh
        existing.auth_key = body.auth
    else:
        db.add(PushSubscription(
            user_id=int(current_user.user_id), endpoint=body.endpoint,
            p256dh_key=body.p256dh, auth_key=body.auth,
        ))
    db.commit()
    return {"success": True}


@router.delete("/push-subscribe", summary="لغو اشتراک Web Push")
async def push_unsubscribe(
    endpoint: str,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    sub = db.query(PushSubscription).filter(
        PushSubscription.endpoint == endpoint, PushSubscription.user_id == int(current_user.user_id)
    ).first()
    if sub:
        db.delete(sub)
        db.commit()
    return {"success": True}


# ─────────────────────────────────────────────
# ارزیابی (قابل import برای Celery task + trigger دستی ادمین)
# ─────────────────────────────────────────────

async def evaluate_price_alerts(db: Session) -> int:
    """قیمت‌های لحظه‌ای فعلی را با قوانین فعال مقایسه و برای موارد رسیده اعلان ارسال می‌کند.

    برمی‌گرداند: تعداد قوانینی که در این اجرا trigger شدند.
    """
    from src.api.endpoints.iran_market import get_overview

    overview = await get_overview()
    prices = {
        item["symbol"]: item["price"]
        for item in overview.get("items", [])
        if item.get("price") is not None
    }

    rules = db.query(PriceAlertRule).filter(PriceAlertRule.is_active == True).all()
    triggered_count = 0

    for rule in rules:
        price = prices.get(rule.symbol)
        if price is None:
            continue
        target = float(rule.target_price)
        hit = (rule.direction == "above" and price >= target) or (rule.direction == "below" and price <= target)
        if not hit:
            continue

        rule.is_active = False
        rule.triggered_at = datetime.utcnow()

        direction_fa = "بالای" if rule.direction == "above" else "زیر"
        title = f"هشدار قیمت {rule.symbol}"
        body = to_persian_digits(
            f"قیمت {rule.symbol} به {price:,.0f} رسید (هدف: {direction_fa} {target:,.0f})"
        )

        channels = rule.channels or ["in_app"]
        if "in_app" in channels:
            create_in_app_notification(db, rule.user_id, "price_alert", title, body)
        if "sms" in channels:
            user = db.query(User).filter(User.id == rule.user_id).first()
            if user and user.phone:
                await send_sms_text(user.phone, f"{title}: {body}")
        if "push" in channels:
            subs = db.query(PushSubscription).filter(PushSubscription.user_id == rule.user_id).all()
            for sub in subs:
                await send_web_push(sub, title, body)

        triggered_count += 1

    db.commit()
    return triggered_count


@router.post("/evaluate", summary="اجرای دستی ارزیابی هشدارها (فقط ادمین — برای تست/عملیات)")
async def trigger_evaluate(
    db: Session = Depends(get_db),
    _: TokenData = Depends(require_admin),
):
    count = await evaluate_price_alerts(db)
    return {"triggered": count}


def _to_out(r: PriceAlertRule) -> PriceAlertRuleOut:
    return PriceAlertRuleOut(
        id=r.id, symbol=r.symbol, direction=r.direction, target_price=float(r.target_price),
        channels=r.channels or [], note=r.note, is_active=r.is_active,
        triggered_at=r.triggered_at.isoformat() if r.triggered_at else None,
        created_at=r.created_at.isoformat() if r.created_at else None,
    )
