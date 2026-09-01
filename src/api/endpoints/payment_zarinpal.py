"""
ZarinPal Payment Gateway Integration
درگاه پرداخت زرین‌پال

Flow:
  1. POST /api/payment/zarinpal/create   → authority + redirect_url
  2. کاربر به درگاه زرین‌پال redirect می‌شود
  3. GET  /api/payment/zarinpal/callback  → verify + update DB
"""

import logging
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, BigInteger, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.security import get_current_active_user, TokenData
from src.database.postgres_connection import get_db, Base

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/payment/zarinpal", tags=["ZarinPal Payment"])

# ─────────────────────────────────────────────
# DB Model
# ─────────────────────────────────────────────

class PaymentOrder(Base):
    __tablename__ = "payment_orders"
    __table_args__ = {"extend_existing": True}

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id     = Column(String(128), nullable=False, index=True)
    gateway     = Column(String(32),  nullable=False, default="zarinpal")
    authority   = Column(String(128), nullable=False, unique=True)
    amount_rial = Column(BigInteger,  nullable=False)
    amount_toman= Column(BigInteger,  nullable=False)
    description = Column(Text,        nullable=True)
    # مقصد پرداخت: general (پرداخت یک‌باره قدیمی) | wallet_topup | subscription
    purpose     = Column(String(32),  nullable=False, default="general")
    purpose_ref = Column(String(64),  nullable=True)   # مثلاً کد پلن اشتراک
    status      = Column(String(16),  nullable=False, default="pending")   # pending|paid|failed|expired
    ref_id      = Column(String(64),  nullable=True)
    card_pan    = Column(String(20),  nullable=True)
    error_code  = Column(String(32),  nullable=True)
    callback_payload = Column(JSONB,  nullable=True)
    verify_payload   = Column(JSONB,  nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    paid_at     = Column(DateTime(timezone=True), nullable=True)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class CreatePaymentRequest(BaseModel):
    amount_toman: int = Field(..., ge=1000, description="مبلغ به تومان (حداقل ۱۰۰۰ تومان)")
    description: str  = Field(..., max_length=255, description="توضیح تراکنش")
    callback_url: Optional[str] = Field(None, description="آدرس بازگشت (اختیاری — پیش‌فرض از config)")
    purpose: str = Field("general", description="general | wallet_topup — این endpoint عمومی است، برای اشتراک از /api/subscriptions/subscribe استفاده کنید")

    @validator("purpose")
    def _validate_purpose(cls, v):
        # 'subscription' عمداً اینجا مجاز نیست: مبلغ آن باید سمت سرور از روی
        # پلن محاسبه شود (نه از بدنه درخواست کلاینت) — به /api/subscriptions/subscribe مراجعه کنید.
        if v not in ("general", "wallet_topup"):
            raise ValueError("purpose باید general یا wallet_topup باشد")
        return v


class CreatePaymentResponse(BaseModel):
    authority:    str
    redirect_url: str
    order_id:     int


class PaymentStatusResponse(BaseModel):
    order_id:    int
    status:      str
    amount_toman: int
    ref_id:      Optional[str]
    card_pan:    Optional[str]
    paid_at:     Optional[str]
    description: Optional[str]


# ─────────────────────────────────────────────
# ZarinPal Error Codes
# ─────────────────────────────────────────────

ZARINPAL_ERRORS: dict[int, str] = {
    -9:  "اطلاعات ورودی نادرست",
    -10: "آی‌پی یا merchant نامعتبر",
    -11: "merchant غیرفعال",
    -12: "تلاش بیش از حد",
    -15: "درگاه پرداخت معلق",
    -21: "اطلاعات پرداخت وجود ندارد",
    -22: "تراکنش ناموفق",
    -33: "مبلغ تراکنش با مبلغ پرداخت‌شده برابر نیست",
    -34: "Limit تراکنش تجاوز شد",
    -40: "دسترسی به متد مجاز نیست",
    -41: "اطلاعات AdditionalData نامعتبر",
    -42: "عمر شناسه پرداخت منقضی شده",
    -54: "تراکنش آرشیو شده",
    100: "موفق",
    101: "قبلاً تأیید شده",
}


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _merchant_id() -> str:
    mid = getattr(settings, "zarinpal_merchant_id", None) or \
          getattr(getattr(settings, "payment", None), "zarinpal_merchant_id", None)
    if not mid:
        raise HTTPException(503, "ZARINPAL_MERCHANT_ID not configured")
    return mid


def _callback_url(override: Optional[str]) -> str:
    if override:
        return override
    base = getattr(settings, "app_base_url", None) or \
           getattr(getattr(settings, "app", None), "base_url", None) or \
           "http://localhost:3003"
    return f"{base}/payment/callback/zarinpal"


async def _zarinpal_create(amount_toman: int, callback_url: str, description: str) -> str:
    """Calls ZarinPal request API → returns authority string."""
    payload = {
        "merchant_id": _merchant_id(),
        "amount": amount_toman * 10,        # تبدیل به ریال
        "callback_url": callback_url,
        "description": description,
    }
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.post(
            "https://api.zarinpal.com/pg/v4/payment/request.json",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
    data = resp.json()
    code = (data.get("data") or {}).get("code", -999)
    if code != 100:
        msg = ZARINPAL_ERRORS.get(code, f"کد خطا: {code}")
        raise HTTPException(502, f"ZarinPal create failed: {msg}")
    return data["data"]["authority"]


async def _zarinpal_verify(authority: str, amount_toman: int) -> dict:
    """Calls ZarinPal verify API — MUST be called even on callback failure."""
    payload = {
        "merchant_id": _merchant_id(),
        "amount": amount_toman * 10,
        "authority": authority,
    }
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.post(
            "https://api.zarinpal.com/pg/v4/payment/verify.json",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
    data = resp.json()
    code = (data.get("data") or {}).get("code", -999)
    if code == 100:
        return {
            "success": True,
            "ref_id":   str(data["data"]["ref_id"]),
            "card_pan": data["data"].get("card_pan"),
            "raw":      data,
        }
    if code == 101:
        # Already verified — still success (replay protection)
        return {
            "success":          True,
            "ref_id":           str(data["data"].get("ref_id", "")),
            "card_pan":         None,
            "already_verified": True,
            "raw":              data,
        }
    msg = ZARINPAL_ERRORS.get(code, f"کد خطا: {code}")
    return {"success": False, "code": code, "message": msg, "raw": data}


async def create_order(
    db: Session,
    user_id: str,
    amount_toman: int,
    description: str,
    purpose: str,
    purpose_ref: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> "PaymentOrder":
    """ایجاد سفارش زرین‌پال قابل reuse برای هر جریان دیگر (کیف پول، اشتراک، ...).

    مبلغ همیشه باید سمت سرور (نه از بدنه درخواست کاربر) تعیین شود.
    """
    cb_url = _callback_url(callback_url)
    authority = await _zarinpal_create(amount_toman, cb_url, description)
    order = PaymentOrder(
        user_id=str(user_id), gateway="zarinpal", authority=authority,
        amount_rial=amount_toman * 10, amount_toman=amount_toman,
        description=description, purpose=purpose, purpose_ref=purpose_ref,
        status="pending",
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def order_redirect_url(order: "PaymentOrder") -> str:
    return f"https://www.zarinpal.com/pg/StartPay/{order.authority}"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post("/create", response_model=CreatePaymentResponse, summary="ایجاد درخواست پرداخت")
async def create_payment(
    body: CreatePaymentRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    مرحله ۱: ایجاد سفارش پرداخت و دریافت لینک درگاه.

    - مبلغ به **تومان** ارسال می‌شود.
    - authority و redirect_url را به frontend برمی‌گرداند.
    - پیش از ریدایرکت، سفارش در DB ثبت می‌شود.
    """
    order = await create_order(
        db, current_user.user_id, body.amount_toman, body.description,
        purpose=body.purpose, callback_url=body.callback_url,
    )

    return CreatePaymentResponse(
        authority    = order.authority,
        redirect_url = order_redirect_url(order),
        order_id     = order.id,
    )


@router.get("/callback", summary="بازگشت از درگاه زرین‌پال")
async def zarinpal_callback(
    Status:    str = Query(...),
    Authority: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    مرحله ۳ + ۴: callback از زرین‌پال + verify خودکار.

    - هرگز فقط به Status اعتماد نکنید — verify همیشه انجام می‌شود.
    - کاربر را به /payment/success یا /payment/failed ریدایرکت می‌کند.
    """
    base_url = getattr(settings, "app_base_url", None) or \
               getattr(getattr(settings, "app", None), "base_url", None) or \
               "http://localhost:3003"

    order: Optional[PaymentOrder] = db.query(PaymentOrder).filter(
        PaymentOrder.authority == Authority
    ).first()

    if not order:
        logger.error(f"ZarinPal callback: order not found for authority={Authority}")
        return RedirectResponse(f"{base_url}/payment/failed?reason=not_found")

    order.callback_payload = {"Status": Status, "Authority": Authority}

    if Status != "OK":
        order.status     = "failed"
        order.error_code = "user_cancelled"
        db.commit()
        return RedirectResponse(f"{base_url}/payment/failed?id={order.id}&reason=cancelled")

    # ── verify ────────────────────────────────────────────
    result = await _zarinpal_verify(Authority, order.amount_toman)
    order.verify_payload = result.get("raw")

    if not result["success"]:
        order.status     = "failed"
        order.error_code = str(result.get("code", "verify_failed"))
        db.commit()
        return RedirectResponse(
            f"{base_url}/payment/failed?id={order.id}&reason={order.error_code}"
        )

    order.status   = "paid"
    order.ref_id   = result["ref_id"]
    order.card_pan = result.get("card_pan")
    order.paid_at  = datetime.utcnow()
    db.commit()

    try:
        _dispatch_payment_success(db, order)
    except Exception as e:
        # پرداخت قبلاً موفق ثبت شده؛ خطای dispatch نباید کاربر را به صفحه شکست بفرستد،
        # اما باید لاگ شود تا با پیگیری دستی جبران شود.
        logger.error(f"Payment success but dispatch failed for order={order.id}: {e}")

    return RedirectResponse(
        f"{base_url}/payment/success?id={order.id}&ref={result['ref_id']}"
    )


def _dispatch_payment_success(db: Session, order: "PaymentOrder") -> None:
    """اثر جانبی پرداخت موفق را بر اساس purpose سفارش اعمال می‌کند.

    - wallet_topup: افزایش موجودی کیف پول ریالی کاربر + ثبت تراکنش
    - subscription: فعال‌سازی/تمدید اشتراک کاربر بر اساس purpose_ref (کد پلن)
    - general: بدون اثر جانبی (پرداخت آزاد قدیمی)
    """
    if order.purpose == "wallet_topup":
        from src.database.models import WalletBalance, WalletTransaction
        import uuid

        user_id_int = int(order.user_id)
        balance = (
            db.query(WalletBalance)
            .filter(WalletBalance.user_id == user_id_int, WalletBalance.currency == "IRT")
            .first()
        )
        if not balance:
            balance = WalletBalance(user_id=user_id_int, currency="IRT", balance=0, available=0)
            db.add(balance)
            db.flush()

        balance.balance = (balance.balance or 0) + order.amount_toman
        balance.available = (balance.available or 0) + order.amount_toman

        db.add(WalletTransaction(
            id=str(uuid.uuid4()),
            user_id=user_id_int,
            type="deposit",
            amount=order.amount_toman,
            currency="IRT",
            status="completed",
            method="card",
            description=f"شارژ کیف پول از طریق زرین‌پال — سفارش #{order.id}",
            reference=order.ref_id,
        ))
        db.commit()

    elif order.purpose == "subscription":
        from src.database.models import SubscriptionPlan, UserSubscription
        from datetime import timedelta

        plan = db.query(SubscriptionPlan).filter(
            SubscriptionPlan.code == order.purpose_ref, SubscriptionPlan.is_active == True
        ).first()
        if not plan:
            logger.error(f"Subscription plan not found for code={order.purpose_ref} (order={order.id})")
            return

        user_id_int = int(order.user_id)
        sub = (
            db.query(UserSubscription)
            .filter(UserSubscription.user_id == user_id_int, UserSubscription.status == "active")
            .order_by(UserSubscription.end_at.desc())
            .first()
        )
        now = datetime.utcnow()
        if sub and sub.end_at and sub.end_at > now:
            # تمدید: مدت پلن جدید را به انتهای اشتراک فعلی اضافه می‌کند
            sub.end_at = sub.end_at + timedelta(days=plan.duration_days)
            sub.plan_id = plan.id
            sub.last_payment_order_id = order.id
        else:
            sub = UserSubscription(
                user_id=user_id_int,
                plan_id=plan.id,
                status="active",
                end_at=now + timedelta(days=plan.duration_days),
                last_payment_order_id=order.id,
            )
            db.add(sub)
        db.commit()


@router.get("/status/{order_id}", response_model=PaymentStatusResponse, summary="وضعیت سفارش پرداخت")
async def payment_status(
    order_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """وضعیت یک سفارش پرداخت را برمی‌گرداند."""
    user_id = str(current_user.user_id)
    order: Optional[PaymentOrder] = db.query(PaymentOrder).filter(
        PaymentOrder.id      == order_id,
        PaymentOrder.user_id == user_id,
    ).first()

    if not order:
        raise HTTPException(404, "سفارش پیدا نشد")

    return PaymentStatusResponse(
        order_id     = order.id,
        status       = order.status,
        amount_toman = order.amount_toman,
        ref_id       = order.ref_id,
        card_pan     = order.card_pan,
        paid_at      = order.paid_at.isoformat() if order.paid_at else None,
        description  = order.description,
    )


@router.get("/history", summary="تاریخچه پرداخت‌های کاربر")
async def payment_history(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """لیست تمام پرداخت‌های کاربر جاری."""
    user_id = str(current_user.user_id)
    orders = (
        db.query(PaymentOrder)
        .filter(PaymentOrder.user_id == user_id)
        .order_by(PaymentOrder.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id":          o.id,
            "status":      o.status,
            "amount_toman":o.amount_toman,
            "ref_id":      o.ref_id,
            "description": o.description,
            "created_at":  o.created_at.isoformat() if o.created_at else None,
            "paid_at":     o.paid_at.isoformat() if o.paid_at else None,
        }
        for o in orders
    ]
