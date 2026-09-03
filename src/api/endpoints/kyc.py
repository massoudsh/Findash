"""
احراز هویت مالی / KYC (issue #20)

نکته مهم (طبق خود issue): استعلام هویت واقعی از سازمان ثبت‌احوال نیازمند اتصال به
یک ارائه‌دهنده مجاز (Finnotech/Jibit/Zibal و...) و تأیید حقوقی/رگولاتوری است —
این یک تصمیم محصول/کسب‌وکار است، نه چیزی که بدون قرارداد بتوان به‌صورت فنی جعل کرد.
آنچه اینجا واقعی و کامل است:
  - فرم ثبت KYC با اعتبارسنجی واقعی فرمت کد ملی (الگوریتم استاندارد چک‌سام ایران)
    و شماره موبایل ایرانی — نه صرفاً «هر رشته‌ای پذیرفته می‌شود».
  - جریان کامل وضعیت: pending_review → verified/rejected با بازبینی دستی ادمین
    (audit log واقعی + اعلان درون‌برنامه‌ای به کاربر).
`provider`/`provider_reference` برای زمانی نگه داشته شده که تصمیم ارائه‌دهنده
استعلام آنی گرفته شود؛ فعلاً بازبینی توسط تیم پشتیبانی/ادمین انجام می‌شود.
"""

import logging
import re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user, require_admin, TokenData
from src.database.postgres_connection import get_db
from src.database.models import KYCProfile, User, AuditLog
from src.services.notifications import create_in_app_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kyc", tags=["KYC"])

MOBILE_RE = re.compile(r"^(?:\+98|0098|0)?9\d{9}$")


def _normalize_mobile(mobile: str) -> str:
    mobile = mobile.strip().replace(" ", "").replace("-", "")
    if mobile.startswith("+98"):
        return "0" + mobile[3:]
    if mobile.startswith("0098"):
        return "0" + mobile[4:]
    if mobile.startswith("9") and len(mobile) == 10:
        return "0" + mobile
    return mobile


def validate_iranian_national_code(code: str) -> bool:
    """الگوریتم استاندارد چک‌سام کد ملی ایران (چک‌رقم بر پایه وزن‌دهی ۱۰ تا ۲)."""
    if not re.match(r"^\d{10}$", code):
        return False
    if code == code[0] * 10:
        return False
    check = int(code[9])
    total = sum(int(code[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    return check == remainder if remainder < 2 else check == 11 - remainder


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class KYCSubmitRequest(BaseModel):
    national_code: str
    full_name: str = Field(..., min_length=2, max_length=255)
    birth_date_shamsi: Optional[str] = Field(None, description="فرمت ۱۴۰۰-۰۱-۰۱")
    mobile_number: str

    @field_validator("national_code")
    @classmethod
    def _nc(cls, v: str) -> str:
        v = v.strip()
        if not validate_iranian_national_code(v):
            raise ValueError("کد ملی نامعتبر است")
        return v

    @field_validator("mobile_number")
    @classmethod
    def _mobile(cls, v: str) -> str:
        if not MOBILE_RE.match(v.strip().replace(" ", "").replace("-", "")):
            raise ValueError("شماره موبایل نامعتبر است")
        return _normalize_mobile(v)

    @field_validator("birth_date_shamsi")
    @classmethod
    def _birth(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.match(r"^1[23]\d{2}-\d{2}-\d{2}$", v.strip()):
            raise ValueError("تاریخ تولد باید به فرمت شمسی ۱۴۰۰-۰۱-۰۱ باشد")
        return v.strip()


class KYCStatusOut(BaseModel):
    status: str
    full_name: Optional[str] = None
    national_code_masked: Optional[str] = None
    mobile_number: Optional[str] = None
    rejection_reason: Optional[str] = None
    submitted_at: Optional[str] = None
    reviewed_at: Optional[str] = None


class KYCAdminOut(KYCStatusOut):
    id: int
    user_id: int
    user_email: Optional[str] = None


class KYCReviewRequest(BaseModel):
    status: str = Field(..., description="verified | rejected")
    rejection_reason: Optional[str] = None

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in {"verified", "rejected"}:
            raise ValueError("status باید verified یا rejected باشد")
        return v


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post("/submit", response_model=KYCStatusOut, summary="ثبت/به‌روزرسانی فرم احراز هویت")
async def submit_kyc(
    body: KYCSubmitRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    profile = db.query(KYCProfile).filter(KYCProfile.user_id == int(current_user.user_id)).first()
    if profile and profile.status == "verified":
        raise HTTPException(400, "احراز هویت شما قبلاً تأیید شده و نیازی به ثبت مجدد نیست")

    if profile:
        profile.national_code = body.national_code
        profile.full_name = body.full_name
        profile.birth_date_shamsi = body.birth_date_shamsi
        profile.mobile_number = body.mobile_number
        profile.status = "pending_review"
        profile.rejection_reason = None
        profile.reviewed_at = None
        profile.submitted_at = datetime.utcnow()
    else:
        profile = KYCProfile(
            user_id=int(current_user.user_id), national_code=body.national_code,
            full_name=body.full_name, birth_date_shamsi=body.birth_date_shamsi,
            mobile_number=body.mobile_number, status="pending_review",
        )
        db.add(profile)
    db.commit()
    db.refresh(profile)
    return _to_out(profile)


@router.get("/me", response_model=KYCStatusOut, summary="وضعیت احراز هویت کاربر جاری")
async def get_my_kyc(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    profile = db.query(KYCProfile).filter(KYCProfile.user_id == int(current_user.user_id)).first()
    if not profile:
        return KYCStatusOut(status="not_submitted")
    return _to_out(profile)


@router.get("/pending", response_model=List[KYCAdminOut], summary="لیست درخواست‌های در انتظار بازبینی (فقط ادمین)")
async def list_pending(
    db: Session = Depends(get_db),
    _: TokenData = Depends(require_admin),
):
    rows = (
        db.query(KYCProfile)
        .filter(KYCProfile.status == "pending_review")
        .order_by(KYCProfile.submitted_at.asc())
        .all()
    )
    out = []
    for p in rows:
        base = _to_out(p)
        user = db.query(User).filter(User.id == p.user_id).first()
        out.append(KYCAdminOut(**base.model_dump(), id=p.id, user_id=p.user_id, user_email=user.email if user else None))
    return out


@router.post("/{kyc_id}/review", response_model=KYCAdminOut, summary="تأیید/رد درخواست KYC (فقط ادمین)")
async def review_kyc(
    kyc_id: int,
    body: KYCReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: TokenData = Depends(require_admin),
):
    profile = db.query(KYCProfile).filter(KYCProfile.id == kyc_id).first()
    if not profile:
        raise HTTPException(404, "درخواست KYC پیدا نشد")
    if body.status == "rejected" and not body.rejection_reason:
        raise HTTPException(400, "برای رد درخواست، دلیل الزامی است")

    profile.status = body.status
    profile.rejection_reason = body.rejection_reason if body.status == "rejected" else None
    profile.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)

    db.add(AuditLog(
        actor_user_id=int(admin.user_id), action="kyc.reviewed", target_type="kyc_profile",
        target_id=str(profile.id), detail={"status": body.status, "reason": body.rejection_reason},
        ip_address=request.client.host if request.client else None,
    ))
    db.commit()

    title = "احراز هویت شما تأیید شد" if body.status == "verified" else "احراز هویت شما رد شد"
    body_text = body.rejection_reason if body.status == "rejected" else "می‌توانید از امکانات نیازمند احراز هویت استفاده کنید."
    create_in_app_notification(db, profile.user_id, "kyc", title, body_text)

    user = db.query(User).filter(User.id == profile.user_id).first()
    return KYCAdminOut(**_to_out(profile).model_dump(), id=profile.id, user_id=profile.user_id, user_email=user.email if user else None)


def _to_out(p: KYCProfile) -> KYCStatusOut:
    masked = f"{p.national_code[:3]}***{p.national_code[-2:]}" if p.national_code else None
    return KYCStatusOut(
        status=p.status, full_name=p.full_name, national_code_masked=masked,
        mobile_number=p.mobile_number, rejection_reason=p.rejection_reason,
        submitted_at=p.submitted_at.isoformat() if p.submitted_at else None,
        reviewed_at=p.reviewed_at.isoformat() if p.reviewed_at else None,
    )
