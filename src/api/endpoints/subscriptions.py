"""
مدیریت اشتراک کاربران (issue #13)
درخواست پرداخت اشتراک، وضعیت فعلی، و entitlement check برای gating در بقیه بخش‌های برنامه.

جریان:
  1. GET  /api/subscriptions/plans        → لیست پلن‌های فعال (seed خودکار در startup)
  2. POST /api/subscriptions/subscribe    → ایجاد سفارش زرین‌پال با purpose=subscription
  3. GET  /api/subscriptions/me           → وضعیت اشتراک کاربر جاری
  فعال‌سازی واقعی اشتراک در callback زرین‌پال (payment_zarinpal.py) انجام می‌شود،
  نه اینجا — چون تا verify نشدن پرداخت نباید اشتراک فعال شود.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user, TokenData
from src.database.postgres_connection import get_db
from src.database.models import SubscriptionPlan, UserSubscription
from src.api.endpoints.payment_zarinpal import create_order, order_redirect_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subscriptions", tags=["Subscriptions"])

# پلن‌های پیش‌فرض — همان قیمت‌گذاری که در frontend/payment/checkout استفاده می‌شود.
DEFAULT_PLANS = [
    {"code": "basic", "name_fa": "پایه",     "price_toman": 99000,  "duration_days": 30,
     "features": ["تحلیل تکنیکال", "اخبار بازار", "هشدار قیمت"]},
    {"code": "pro",   "name_fa": "حرفه‌ای",  "price_toman": 249000, "duration_days": 30,
     "features": ["همه امکانات پایه", "هوش مصنوعی معاملاتی", "تحلیل آپشن", "اسکن ریل‌تایم"]},
    {"code": "elite", "name_fa": "الیت",     "price_toman": 499000, "duration_days": 30,
     "features": ["همه امکانات حرفه‌ای", "API اختصاصی", "گزارش هفتگی AI", "پشتیبانی اولویت‌دار"]},
]


def seed_default_plans(db: Session) -> None:
    """اگر هیچ پلنی در DB نباشد، پلن‌های پیش‌فرض را ایجاد می‌کند (idempotent)."""
    existing = {p.code for p in db.query(SubscriptionPlan.code).all()}
    for p in DEFAULT_PLANS:
        if p["code"] not in existing:
            db.add(SubscriptionPlan(**p))
    db.commit()


class PlanOut(BaseModel):
    code: str
    name_fa: str
    price_toman: int
    duration_days: int
    features: Optional[list] = None


class SubscribeRequest(BaseModel):
    plan_code: str
    callback_url: Optional[str] = None


class SubscribeResponse(BaseModel):
    authority: str
    redirect_url: str
    order_id: int


class SubscriptionStatusOut(BaseModel):
    active: bool
    plan_code: Optional[str] = None
    plan_name: Optional[str] = None
    end_at: Optional[str] = None
    auto_renew: bool = False


@router.get("/plans", response_model=list[PlanOut], summary="لیست پلن‌های فعال")
async def list_plans(db: Session = Depends(get_db)):
    plans = db.query(SubscriptionPlan).filter(SubscriptionPlan.is_active == True).all()
    if not plans:
        seed_default_plans(db)
        plans = db.query(SubscriptionPlan).filter(SubscriptionPlan.is_active == True).all()
    return [
        PlanOut(code=p.code, name_fa=p.name_fa, price_toman=p.price_toman,
                duration_days=p.duration_days, features=p.features)
        for p in plans
    ]


@router.post("/subscribe", response_model=SubscribeResponse, summary="ایجاد درخواست پرداخت اشتراک")
async def subscribe(
    body: SubscribeRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """قیمت همیشه از روی رکورد SubscriptionPlan در دیتابیس محاسبه می‌شود،
    نه از ورودی کاربر — تا از دستکاری قیمت جلوگیری شود."""
    plan = db.query(SubscriptionPlan).filter(
        SubscriptionPlan.code == body.plan_code, SubscriptionPlan.is_active == True
    ).first()
    if not plan:
        seed_default_plans(db)
        plan = db.query(SubscriptionPlan).filter(
            SubscriptionPlan.code == body.plan_code, SubscriptionPlan.is_active == True
        ).first()
    if not plan:
        raise HTTPException(404, "پلن یافت نشد")

    order = await create_order(
        db, current_user.user_id, plan.price_toman,
        description=f"اشتراک — پلن {plan.name_fa}",
        purpose="subscription", purpose_ref=plan.code,
        callback_url=body.callback_url,
    )

    return SubscribeResponse(
        authority=order.authority,
        redirect_url=order_redirect_url(order),
        order_id=order.id,
    )


@router.get("/me", response_model=SubscriptionStatusOut, summary="وضعیت اشتراک کاربر جاری")
async def my_subscription(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    sub = (
        db.query(UserSubscription)
        .filter(UserSubscription.user_id == int(current_user.user_id), UserSubscription.status == "active")
        .order_by(UserSubscription.end_at.desc())
        .first()
    )
    if not sub or not sub.end_at or sub.end_at <= datetime.utcnow():
        return SubscriptionStatusOut(active=False)

    return SubscriptionStatusOut(
        active=True,
        plan_code=sub.plan.code if sub.plan else None,
        plan_name=sub.plan.name_fa if sub.plan else None,
        end_at=sub.end_at.isoformat(),
        auto_renew=sub.auto_renew,
    )


def has_active_subscription(db: Session, user_id: int) -> bool:
    """Helper قابل import برای gating در بقیه endpoint ها (issue #14)."""
    sub = (
        db.query(UserSubscription)
        .filter(UserSubscription.user_id == user_id, UserSubscription.status == "active")
        .order_by(UserSubscription.end_at.desc())
        .first()
    )
    return bool(sub and sub.end_at and sub.end_at > datetime.utcnow())


async def require_active_subscription(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> TokenData:
    """وابستگی FastAPI برای گیت کردن ویژگی‌های اشتراکی (issue #14).

    قبلاً هیچ auth/premium gating‌ای وجود نداشت — ویژگی‌های «فقط برای مشترکین»
    (مثل ربات معاملاتی خودکار، طبق فیچرلیست پلن pro در DEFAULT_PLANS) بدون
    هیچ محدودیتی برای همه (حتی کاربر مهمان) در دسترس بود. ادمین از این محدودیت
    معاف است تا بتواند بدون خرید اشتراک روی پنل تست کند.
    """
    if "admin" in current_user.roles:
        return current_user
    if not has_active_subscription(db, int(current_user.user_id)):
        raise HTTPException(
            status_code=402,
            detail="این ویژگی نیازمند اشتراک فعال است — از /api/subscriptions/plans یک پلن انتخاب کنید",
        )
    return current_user
