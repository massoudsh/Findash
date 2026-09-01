"""
سرویس مشترک اعلان‌رسانی — SMS (KaveNegar)، Web Push (VAPID) و اعلان درون‌برنامه‌ای.

استفاده‌کنندگان: هشدار قیمت (#19)، یادآوری انقضای اشتراک (#13)،
تغییر وضعیت KYC (#20)، و رویدادهای پنل ادمین (#12).

اگر SMS_API_KEY یا VAPID_PRIVATE_KEY تنظیم نشده باشد، ارسال واقعی رخ نمی‌دهد
و فقط در لاگ ثبت می‌شود (همان الگوی stub-fallback که در بقیه پروژه استفاده شده) —
این طوری فراخوانی این توابع در سندباکس/محیط dev هرگز کرش نمی‌کند.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.database.models import Notification, PushSubscription

logger = logging.getLogger(__name__)
settings = get_settings()


async def send_sms_text(phone: str, message: str) -> bool:
    """ارسال پیامک متنی آزاد (نه OTP) از طریق KaveNegar Send API.

    OTP از template جدای verify/lookup در otp_auth.py استفاده می‌کند؛
    این تابع برای هشدار قیمت/یادآوری اشتراک است که متن آزاد نیاز دارد.
    """
    provider = (settings.sms.provider or "").lower()
    api_key = settings.sms.api_key or ""
    sender = settings.sms.sender_line or ""

    if provider != "kavenegar" or not api_key:
        logger.info(f"[SMS dev-fallback] to={phone}: {message}")
        return False

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.kavenegar.com/v1/{api_key}/sms/send.json",
                data={"receptor": phone, "sender": sender, "message": message},
            )
            if resp.status_code != 200:
                logger.error(f"KaveNegar send failed: {resp.status_code} {resp.text}")
            return resp.status_code == 200
    except Exception as e:
        logger.error(f"KaveNegar send exception: {e}")
        return False


async def send_web_push(subscription: PushSubscription, title: str, body: str) -> bool:
    """ارسال Web Push به یک اشتراک مرورگر مشخص با کلیدهای VAPID."""
    if not settings.webpush.vapid_private_key:
        logger.info(f"[WebPush dev-fallback] to={subscription.endpoint[:40]}...: {title} — {body}")
        return False

    try:
        from pywebpush import webpush, WebPushException
        import json

        webpush(
            subscription_info={
                "endpoint": subscription.endpoint,
                "keys": {"p256dh": subscription.p256dh_key, "auth": subscription.auth_key},
            },
            data=json.dumps({"title": title, "body": body}),
            vapid_private_key=settings.webpush.vapid_private_key,
            vapid_claims={"sub": f"mailto:{settings.webpush.vapid_admin_email}"},
        )
        return True
    except ImportError:
        logger.warning("pywebpush نصب نیست — Web Push ارسال نشد (pip install pywebpush)")
        return False
    except Exception as e:
        logger.error(f"Web Push failed: {e}")
        return False


def create_in_app_notification(
    db: Session, user_id: int, category: str, title: str, body: Optional[str] = None
) -> Notification:
    """ثبت اعلان درون‌برنامه‌ای در دیتابیس (منبع صفحه /notifications)."""
    n = Notification(user_id=user_id, category=category, title=title, body=body)
    db.add(n)
    db.commit()
    db.refresh(n)
    return n
