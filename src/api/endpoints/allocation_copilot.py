"""
AI Personal Asset Allocation Copilot
-------------------------------------
Educational, rule-based analysis of a household's existing asset composition.

Deliberately does NOT recommend buying/selling any asset and issues no price
or timing signals. It only explains, in plain language, how concentrated or
diversified the portfolio the user already holds is (Herfindahl-Hirschman
Index) and surfaces category-level concentration risk — so it stays a trust
layer / financial-literacy tool rather than an advisory or signal service.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/copilot", tags=["Asset Allocation Copilot"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HoldingIn(BaseModel):
    code: str
    name: str
    type: str  # gold | silver | currency | crypto | stock | bond | real_estate | cash
    value: float = Field(ge=0, description="Current market value of the holding (any single currency)")


class AllocationAnalysisRequest(BaseModel):
    holdings: List[HoldingIn] = Field(default_factory=list)


class CategoryBreakdown(BaseModel):
    type: str
    value: float
    pct: float


class AllocationAnalysisResponse(BaseModel):
    total_value: float
    category_breakdown: List[CategoryBreakdown]
    top_holding_pct: float
    hhi: float  # 0..10000, standard Herfindahl-Hirschman Index scale
    concentration_level: str  # "کم" | "متوسط" | "بالا"
    diversification_score: int  # 0..100, higher = more diversified
    insights: List[str]
    disclaimer: str


DISCLAIMER = (
    "این تحلیل صرفاً برای افزایش آگاهی نسبت به ترکیب دارایی فعلی شماست و "
    "به‌هیچ‌وجه توصیه‌ی خرید، فروش یا نگه‌داری هیچ دارایی خاصی نیست."
)

CATEGORY_LABEL_FA = {
    "gold": "طلا",
    "silver": "نقره",
    "currency": "ارز",
    "crypto": "کریپتو",
    "stock": "سهام",
    "bond": "اوراق",
    "real_estate": "مسکن",
    "cash": "نقدی",
}


def _concentration_level(hhi: float) -> str:
    # Thresholds mirror common competition-economics conventions (0-10000 scale)
    if hhi < 1500:
        return "کم"
    if hhi <= 2500:
        return "متوسط"
    return "بالا"


def _build_insights(breakdown: List[CategoryBreakdown], top_holding_pct: float, hhi: float) -> List[str]:
    insights: List[str] = []

    if not breakdown:
        return ["هنوز دارایی‌ای ثبت نشده — با افزودن دارایی‌های خود می‌توانید ترکیب پرتفوی‌تان را ببینید."]

    dominant = max(breakdown, key=lambda c: c.pct)
    if dominant.pct >= 60:
        insights.append(
            f"حدود {dominant.pct:.0f}٪ ارزش پرتفوی شما در یک دسته ({CATEGORY_LABEL_FA.get(dominant.type, dominant.type)}) "
            "متمرکز است — این یعنی نوسان قیمت همان دسته، اثر بزرگی روی کل دارایی شما دارد."
        )
    elif len(breakdown) == 1:
        insights.append(
            f"همه دارایی ثبت‌شده شما در دسته «{CATEGORY_LABEL_FA.get(dominant.type, dominant.type)}» است."
        )

    if top_holding_pct >= 40:
        insights.append(
            f"بزرگ‌ترین دارایی منفرد شما حدود {top_holding_pct:.0f}٪ از کل ارزش پرتفوی را تشکیل می‌دهد."
        )

    if len(breakdown) >= 4 and hhi < 2000:
        insights.append("دارایی‌های شما بین چند دسته مختلف پخش شده که معمولاً ریسک تمرکز را کاهش می‌دهد.")

    if len(breakdown) <= 2:
        insights.append(
            "پرتفوی شما تنها شامل "
            f"{len(breakdown)} دسته دارایی است؛ افزایش تعداد دسته‌ها می‌تواند اثر نوسان یک بازار خاص را کم کند."
        )

    return insights


@router.post("/allocation-analysis", response_model=AllocationAnalysisResponse)
def analyze_allocation(payload: AllocationAnalysisRequest) -> AllocationAnalysisResponse:
    """
    Rule-based concentration/diversification read-out of the user's existing
    holdings. No price predictions, no buy/sell signals.
    """
    holdings = [h for h in payload.holdings if h.value > 0]
    total_value = sum(h.value for h in holdings)

    if total_value <= 0:
        return AllocationAnalysisResponse(
            total_value=0,
            category_breakdown=[],
            top_holding_pct=0,
            hhi=0,
            concentration_level="کم",
            diversification_score=0,
            insights=_build_insights([], 0, 0),
            disclaimer=DISCLAIMER,
        )

    by_category: dict = {}
    for h in holdings:
        by_category[h.type] = by_category.get(h.type, 0.0) + h.value

    breakdown = sorted(
        (
            CategoryBreakdown(type=t, value=v, pct=round(v / total_value * 100, 2))
            for t, v in by_category.items()
        ),
        key=lambda c: c.pct,
        reverse=True,
    )

    # HHI on category shares (0-10000 scale: sum of squared percentage shares)
    hhi = round(sum((c.pct) ** 2 for c in breakdown), 1)
    concentration_level = _concentration_level(hhi)
    diversification_score = max(0, min(100, round(100 - hhi / 100)))

    top_holding_pct = round(max((h.value for h in holdings), default=0) / total_value * 100, 2)

    insights = _build_insights(breakdown, top_holding_pct, hhi)

    return AllocationAnalysisResponse(
        total_value=total_value,
        category_breakdown=breakdown,
        top_holding_pct=top_holding_pct,
        hhi=hhi,
        concentration_level=concentration_level,
        diversification_score=diversification_score,
        insights=insights,
        disclaimer=DISCLAIMER,
    )
