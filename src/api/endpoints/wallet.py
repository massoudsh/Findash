"""
Wallet & Funding API Endpoints — کیف پول ریالی (issue #21)

موجودی و تراکنش‌ها واقعی و از دیتابیس هستند. شارژ کیف پول از طریق درگاه زرین‌پال
(همان زیرساخت create_order در payment_zarinpal.py) و با purpose=wallet_topup انجام
می‌شود؛ اعتبار واقعی موجودی فقط پس از verify شدن پرداخت در callback اعمال می‌گردد
(به src/api/endpoints/payment_zarinpal.py::_dispatch_payment_success مراجعه کنید).

برداشت (withdraw) به شماره شبا: چون این پروژه به درگاه Payout بانکی متصل نیست،
درخواست برداشت با وضعیت pending ثبت و موجودی available بلافاصله قفل (locked) می‌شود؛
واریز واقعی نیازمند اتصال به یک ارائه‌دهنده Payout (مثلاً جیبیت/زرین‌پال Payout) و
تسویه دستی/عملیاتی توسط ادمین است — این خارج از حوزه کد و نیازمند قرارداد تجاری است.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from src.database.postgres_connection import get_db
from src.database import models as db_models
from src.core.security import get_current_active_user, TokenData
from src.api.endpoints.payment_zarinpal import create_order, order_redirect_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/wallet", tags=["Wallet & Funding"])

CURRENCY = "IRT"  # این کیف پول فقط ریال/تومان ایران را پشتیبانی می‌کند

SHEBA_RE = re.compile(r"^IR\d{24}$")


def _validate_sheba(sheba: str) -> bool:
    """اعتبارسنجی شماره شبا با الگوریتم استاندارد IBAN (mod-97)."""
    if not SHEBA_RE.match(sheba):
        return False
    rearranged = sheba[4:] + sheba[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class WalletBalanceOut(BaseModel):
    currency: str
    balance: float
    available: float
    locked: float
    pending: float


class TransactionOut(BaseModel):
    id: str
    type: str
    amount: float
    currency: str
    status: str
    method: str
    timestamp: str
    description: Optional[str] = None
    reference: Optional[str] = None
    fees: Optional[float] = None


class BankAccountOut(BaseModel):
    id: int
    name: str
    sheba_masked: str
    bank_name: str
    verified: bool
    last_used: Optional[str] = None


class DepositRequest(BaseModel):
    amount_toman: int = Field(..., ge=10000, description="مبلغ شارژ به تومان (حداقل ۱۰,۰۰۰ تومان)")
    callback_url: Optional[str] = None


class WithdrawRequest(BaseModel):
    amount_toman: int = Field(..., ge=10000, description="مبلغ برداشت به تومان")
    bank_account_id: int


class LinkBankAccountRequest(BaseModel):
    name: str = Field(..., max_length=128)
    sheba_number: str = Field(..., description="شماره شبا با فرمت IRxxxxxxxxxxxxxxxxxxxxxxxx")
    bank_name: str = Field(..., max_length=128)

    @field_validator("sheba_number")
    @classmethod
    def _validate(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "")
        if not _validate_sheba(v):
            raise ValueError("شماره شبا نامعتبر است")
        return v


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_or_create_balance(db: Session, user_id: int) -> db_models.WalletBalance:
    bal = (
        db.query(db_models.WalletBalance)
        .filter(db_models.WalletBalance.user_id == user_id, db_models.WalletBalance.currency == CURRENCY)
        .first()
    )
    if not bal:
        bal = db_models.WalletBalance(user_id=user_id, currency=CURRENCY, balance=0, available=0, locked=0, pending=0)
        db.add(bal)
        db.commit()
        db.refresh(bal)
    return bal


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.get("/balances", response_model=List[WalletBalanceOut], summary="موجودی کیف پول")
async def get_wallet_balances(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    bal = _get_or_create_balance(db, int(current_user.user_id))
    return [WalletBalanceOut(
        currency=bal.currency, balance=float(bal.balance or 0), available=float(bal.available or 0),
        locked=float(bal.locked or 0), pending=float(bal.pending or 0),
    )]


@router.get("/transactions", response_model=List[TransactionOut], summary="تاریخچه تراکنش‌های کیف پول")
async def get_transactions(
    limit: int = 50,
    offset: int = 0,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    txs = (
        db.query(db_models.WalletTransaction)
        .filter(db_models.WalletTransaction.user_id == int(current_user.user_id))
        .order_by(db_models.WalletTransaction.timestamp.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )
    return [
        TransactionOut(
            id=t.id, type=t.type, amount=float(t.amount), currency=t.currency, status=t.status,
            method=t.method, timestamp=t.timestamp.isoformat() if t.timestamp else "",
            description=t.description, reference=t.reference,
            fees=float(t.fees) if t.fees is not None else None,
        )
        for t in txs
    ]


@router.get("/bank-accounts", response_model=List[BankAccountOut], summary="لیست حساب‌های بانکی متصل")
async def get_bank_accounts(
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    accounts = db.query(db_models.BankAccount).filter(
        db_models.BankAccount.user_id == int(current_user.user_id)
    ).all()
    return [
        BankAccountOut(
            id=a.id, name=a.name, sheba_masked=f"IR...{a.sheba_number[-4:]}",
            bank_name=a.bank_name, verified=a.verified,
            last_used=a.last_used.isoformat() if a.last_used else None,
        )
        for a in accounts
    ]


@router.post("/bank-accounts/link", response_model=BankAccountOut, summary="افزودن حساب بانکی (شبا)")
async def link_bank_account(
    body: LinkBankAccountRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    account = db_models.BankAccount(
        user_id=int(current_user.user_id), name=body.name, sheba_number=body.sheba_number,
        bank_name=body.bank_name, type="sheba", verified=False,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return BankAccountOut(
        id=account.id, name=account.name, sheba_masked=f"IR...{account.sheba_number[-4:]}",
        bank_name=account.bank_name, verified=account.verified, last_used=None,
    )


@router.delete("/bank-accounts/{account_id}", summary="حذف حساب بانکی")
async def unlink_bank_account(
    account_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    account = db.query(db_models.BankAccount).filter(
        db_models.BankAccount.id == account_id, db_models.BankAccount.user_id == int(current_user.user_id)
    ).first()
    if not account:
        raise HTTPException(404, "حساب بانکی پیدا نشد")
    db.delete(account)
    db.commit()
    return {"success": True}


@router.post("/deposit", summary="شارژ کیف پول از طریق زرین‌پال")
async def create_deposit(
    body: DepositRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """سفارش پرداخت زرین‌پال با purpose=wallet_topup ایجاد می‌کند.
    موجودی فقط پس از verify موفق در callback افزایش می‌یابد (نه اینجا)."""
    order = await create_order(
        db, current_user.user_id, body.amount_toman,
        description="شارژ کیف پول", purpose="wallet_topup",
        callback_url=body.callback_url,
    )
    return {
        "authority": order.authority,
        "redirect_url": order_redirect_url(order),
        "order_id": order.id,
    }


@router.post("/withdraw", summary="درخواست برداشت به شماره شبا")
async def create_withdrawal(
    body: WithdrawRequest,
    current_user: TokenData = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """موجودی available را قفل کرده و یک تراکنش withdrawal با وضعیت pending ثبت می‌کند.
    واریز واقعی به شبا نیازمند اتصال به درگاه Payout و تأیید عملیاتی/دستی است."""
    account = db.query(db_models.BankAccount).filter(
        db_models.BankAccount.id == body.bank_account_id,
        db_models.BankAccount.user_id == int(current_user.user_id),
    ).first()
    if not account:
        raise HTTPException(404, "حساب بانکی پیدا نشد")

    bal = _get_or_create_balance(db, int(current_user.user_id))
    if float(bal.available or 0) < body.amount_toman:
        raise HTTPException(400, "موجودی کافی نیست")

    bal.available = float(bal.available) - body.amount_toman
    bal.locked = float(bal.locked or 0) + body.amount_toman

    tx = db_models.WalletTransaction(
        id=str(uuid.uuid4()), user_id=int(current_user.user_id), type="withdrawal",
        amount=body.amount_toman, currency=CURRENCY, status="pending", method="sheba",
        description=f"درخواست برداشت به {account.bank_name}", bank_account_id=account.id,
    )
    db.add(tx)
    account.last_used = datetime.utcnow()
    db.commit()

    return {
        "success": True,
        "transaction_id": tx.id,
        "message": "درخواست برداشت ثبت شد و پس از بررسی عملیاتی تسویه می‌شود.",
    }
