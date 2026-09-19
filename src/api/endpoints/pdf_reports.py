"""گزارش PDF فارسی — endpoint (issue #18).

خروجی مستقیم `application/pdf` است. اگر پرتفویی با `portfolio_id` مشخص
درخواست شود، فقط وقتی برگردانده می‌شود که متعلق به همان کاربر باشد؛ در غیر
این صورت ۴۰۴ (نه ۴۰۳) تا وجود/عددم وجود پرتفوی دیگران لو نرود.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user, TokenData
from src.database.postgres_connection import get_db
from src.database.models import Portfolio, Position, Trade, RiskMetrics
from src.core.persian_utils import to_jalali_str
from src.services.pdf_reports import build_portfolio_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["Reports"])

_PDF_HEADERS = {
    "Content-Disposition": 'inline; filename="portfolio-report.pdf"',
    "Cache-Control": "no-store",
}


def _resolve_portfolio(db: Session, user_id: int, portfolio_id: Optional[int]) -> Portfolio:
    query = db.query(Portfolio).filter(Portfolio.user_id == user_id)
    if portfolio_id is not None:
        query = query.filter(Portfolio.id == portfolio_id)
    else:
        query = query.filter(Portfolio.is_active.is_(True))
    portfolio = query.order_by(Portfolio.created_at.desc()).first()
    if not portfolio:
        raise HTTPException(404, "پرتفویی برای این کاربر پیدا نشد")
    return portfolio


@router.get("/portfolio.pdf", summary="گزارش PDF فارسی پرتفوی")
async def portfolio_pdf(
    portfolio_id: Optional[int] = Query(None, description="شناسه پرتفوی (پیش‌فرض: فعال‌ترین پرتفوی کاربر)"),
    include_trades: bool = Query(True, description="درج آخرین معاملات"),
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_active_user),
):
    user_id = int(current_user.user_id)
    portfolio = _resolve_portfolio(db, user_id, portfolio_id)

    positions = (
        db.query(Position)
        .filter(Position.portfolio_id == portfolio.id, Position.is_active.is_(True))
        .all()
    )
    trades = []
    if include_trades:
        trades = (
            db.query(Trade)
            .filter(Trade.portfolio_id == portfolio.id)
            .order_by(Trade.trade_date.desc())
            .limit(20)
            .all()
        )

    risk_row = db.query(RiskMetrics).filter(RiskMetrics.portfolio_id == portfolio.id).first()
    risk = None
    if risk_row:
        risk = {
            "var_1d": risk_row.value_at_risk_1d,
            "var_1w": risk_row.value_at_risk_1w,
            "sharpe_ratio": risk_row.sharpe_ratio,
        }

    # بینش‌های ساده و قابل اتکا (بدون فراخوانی LLM — گزارش باید سریع و
    # قطعی باشد؛ تحلیل‌های تولیدی جای دیگری زندگی می‌کنند).
    insights = []
    if positions:
        top = max(positions, key=lambda p: float(p.market_value or 0))
        try:
            total = float(portfolio.total_value or 0)
            share = (float(top.market_value or 0) / total * 100) if total else 0.0
            if share >= 30:
                insights.append(
                    f"تمرکز پرتفوی بالاست: {top.symbol} حدود {share:.1f}٪ از ارزش کل را تشکیل می‌دهد."
                )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    try:
        initial = float(portfolio.initial_cash or 0)
        if initial:
            pnl_pct = (float(portfolio.total_value or 0) - initial) / initial * 100
            insights.append(
                f"بازده کل از زمان شروع: {pnl_pct:.2f}٪ نسبت به سرمایه اولیه."
            )
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    try:
        pdf_bytes = build_portfolio_report(
            portfolio={
                "name": portfolio.name,
                "total_value": portfolio.total_value,
                "current_cash": portfolio.current_cash,
                "initial_cash": portfolio.initial_cash,
            },
            positions=[
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "average_price": p.average_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in positions
            ],
            trades=[
                {
                    "trade_date": t.trade_date,
                    "symbol": t.symbol,
                    "trade_type": t.trade_type,
                    "quantity": t.quantity,
                    "price": t.price,
                    "total_amount": t.total_amount,
                }
                for t in trades
            ],
            risk=risk,
            insights=insights,
        )
    except RuntimeError as exc:
        # فونت فارسی روی این سرور نصب نیست — خطای زیرساختی، نه خطای کاربر
        logger.error("PDF generation unavailable: %s", exc)
        raise HTTPException(503, "تولید PDF در این سرور در دسترس نیست (فونت فارسی نصب نیست)")

    return Response(content=pdf_bytes, media_type="application/pdf", headers=_PDF_HEADERS)


@router.get("/portfolio/preview", summary="متادیتای گزارش PDF (برای نمایش در UI)")
async def portfolio_preview(
    portfolio_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: TokenData = Depends(get_current_active_user),
):
    """اطلاعات سبک برای UI — بدون ساخت خود PDF."""
    user_id = int(current_user.user_id)
    portfolio = _resolve_portfolio(db, user_id, portfolio_id)
    n_positions = (
        db.query(Position)
        .filter(Position.portfolio_id == portfolio.id, Position.is_active.is_(True))
        .count()
    )
    return {
        "portfolio_id": portfolio.id,
        "name": portfolio.name,
        "positions_count": n_positions,
        "generated_for": to_jalali_str(),
        "download_url": f"/api/reports/portfolio.pdf?portfolio_id={portfolio.id}",
    }
