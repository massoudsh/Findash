"""
TASK-012 — احراز هویت با شماره موبایل + OTP
SMS-based authentication for Iran market users.
"""

import logging
import random
import string
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

from src.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/auth", tags=["OTP Authentication"])
logger = logging.getLogger(__name__)

# ─── In-memory OTP store (production: use Redis) ──────────────────────────────
# { phone: {"code": "123456", "expires_at": datetime, "attempts": int} }
_OTP_STORE: dict = {}
_RATE_STORE: dict = {}  # { phone: [timestamp, ...] }

OTP_TTL_MINUTES = 5
OTP_MAX_ATTEMPTS = 3
RATE_LIMIT_COUNT = 3      # max sends per window
RATE_LIMIT_WINDOW = 600   # 10 minutes in seconds


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def _normalize_phone(phone: str) -> str:
    """09121234567 → +989121234567"""
    phone = phone.strip().replace(" ", "").replace("-", "")
    if phone.startswith("09"):
        return "+98" + phone[1:]
    if phone.startswith("9") and len(phone) == 10:
        return "+98" + phone
    return phone


def _is_rate_limited(phone: str) -> bool:
    now = datetime.utcnow().timestamp()
    history = _RATE_STORE.get(phone, [])
    # Remove old timestamps outside window
    history = [t for t in history if now - t < RATE_LIMIT_WINDOW]
    _RATE_STORE[phone] = history
    return len(history) >= RATE_LIMIT_COUNT


def _record_send(phone: str) -> None:
    now = datetime.utcnow().timestamp()
    history = _RATE_STORE.get(phone, [])
    history.append(now)
    _RATE_STORE[phone] = history


async def _send_sms(phone: str, code: str) -> bool:
    """Send OTP via SMS provider.
    Configure SMS_PROVIDER and SMS_API_KEY in .env to enable real SMS.
    """
    provider = getattr(settings, "SMS_PROVIDER", None) or ""
    api_key = getattr(settings, "SMS_API_KEY", None) or ""

    if provider.lower() == "kavenegar" and api_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://api.kavenegar.com/v1/{api_key}/verify/lookup.json",
                    data={"receptor": phone, "token": code, "template": "findash-otp"},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"KaveNegar SMS failed: {e}")
            return False

    # Development fallback: log the OTP
    logger.info(f"[DEV] OTP for {phone}: {code}")
    return True


# ─── Schemas ──────────────────────────────────────────────────────────────────

class SendOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15, description="شماره موبایل ایران (مثال: 09121234567)")

    @validator("phone")
    def validate_iran_phone(cls, v):
        normalized = _normalize_phone(v)
        if not (normalized.startswith("+989") and len(normalized) == 13):
            raise ValueError("شماره موبایل معتبر ایران وارد کنید (مثال: 09121234567)")
        return v


class VerifyOTPRequest(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")
    device_id: Optional[str] = Field(None, max_length=128)


class OTPResponse(BaseModel):
    success: bool
    message: str
    expires_in_seconds: Optional[int] = None


class OTPVerifyResponse(BaseModel):
    success: bool
    message: str
    access_token: Optional[str] = None
    token_type: str = "bearer"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/send-otp", response_model=OTPResponse, summary="ارسال کد OTP")
async def send_otp(request: Request, body: SendOTPRequest):
    """
    شماره موبایل را دریافت کرده و کد ۶ رقمی OTP از طریق SMS ارسال می‌کند.
    محدودیت: حداکثر ۳ بار در ۱۰ دقیقه.
    """
    phone = _normalize_phone(body.phone)

    if _is_rate_limited(phone):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="تعداد درخواست‌ها از حد مجاز گذشته. ۱۰ دقیقه صبر کنید.",
        )

    code = _generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_TTL_MINUTES)

    _OTP_STORE[phone] = {
        "code": code,
        "expires_at": expires_at,
        "attempts": 0,
    }
    _record_send(phone)

    sent = await _send_sms(phone, code)
    if not sent:
        logger.error(f"Failed to send OTP SMS to {phone}")
        # Still return success to avoid phone enumeration

    logger.info(f"OTP sent to {phone} (expires: {expires_at.isoformat()})")

    return OTPResponse(
        success=True,
        message=f"کد تأیید به شماره شما ارسال شد",
        expires_in_seconds=OTP_TTL_MINUTES * 60,
    )


@router.post("/verify-otp", response_model=OTPVerifyResponse, summary="تأیید کد OTP")
async def verify_otp(body: VerifyOTPRequest):
    """
    کد OTP را تأیید کرده و در صورت موفقیت، JWT صادر می‌کند.
    بعد از ۳ تلاش اشتباه، شماره قفل می‌شود.
    """
    from src.core.security import create_access_token

    phone = _normalize_phone(body.phone)
    record = _OTP_STORE.get(phone)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="کد OTP معتبر نیست یا منقضی شده. ابتدا درخواست کد جدید دهید.",
        )

    # Check expiry
    if datetime.utcnow() > record["expires_at"]:
        del _OTP_STORE[phone]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="کد OTP منقضی شده است. کد جدید درخواست دهید.",
        )

    # Check attempts
    if record["attempts"] >= OTP_MAX_ATTEMPTS:
        del _OTP_STORE[phone]
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="تعداد تلاش‌های مجاز تمام شد. کد جدید درخواست دهید.",
        )

    record["attempts"] += 1

    if record["code"] != body.code:
        remaining = OTP_MAX_ATTEMPTS - record["attempts"]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"کد اشتباه است. {remaining} تلاش باقی دارید.",
        )

    # Success — issue JWT
    del _OTP_STORE[phone]

    token_data = {
        "sub": phone,
        "phone": phone,
        "auth_method": "otp",
        "device_id": body.device_id or "",
    }
    try:
        access_token = create_access_token(token_data)
    except Exception:
        # Fallback: minimal JWT if security module unavailable
        import jwt as pyjwt
        access_token = pyjwt.encode(
            {**token_data, "exp": datetime.utcnow() + timedelta(hours=24)},
            getattr(settings, "JWT_SECRET_KEY", "dev-key"),
            algorithm="HS256",
        )

    logger.info(f"OTP verified successfully for {phone}")

    return OTPVerifyResponse(
        success=True,
        message="ورود موفق",
        access_token=access_token,
    )
