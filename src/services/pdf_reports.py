"""تولید گزارش PDF فارسی از پرتفوی و تحلیل (issue #18).

نکته‌ی اصلی: متن فارسی در PDF باید دو مرحله پیش‌پردازش شود، وگرنه حروف جدا و
از چپ به راست نمایش داده می‌شوند:
  ۱. `arabic_reshaper` حروف را به شکل متصل (presentation forms) تبدیل می‌کند.
  ۲. `python-bidi` ترتیب حروف را راست‌به‌چپ می‌کند.
ترکیب این دو با یک فونت TTF فارسی که گلیف‌های عربی داشته باشد الزامی است؛
فونت‌های داخلی reportlab (Helvetica و…) هیچ گلیف فارسی ندارند و خروجی
مربع‌های خالی می‌شود.
"""

import io
from typing import List, Optional

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.core.persian_utils import to_jalali_str, to_persian_digits

FONT_NAME = "Vazirmatn"
_FONT_PATHS = [
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf",
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Bold.ttf",
]
_BOLD_PATHS = [
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Bold.ttf",
    "/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf",
]

ACCENT = colors.HexColor("#3B82F6")
DANGER = colors.HexColor("#EF4444")
SUCCESS = colors.HexColor("#22C55E")
MUTED = colors.HexColor("#6B7280")

_fonts_ready: Optional[bool] = None


def register_fonts() -> bool:
    """فونت فارسی را یک‌بار در reportlab ثبت می‌کند. اگر فونت نبود False."""
    global _fonts_ready
    if _fonts_ready is not None:
        return _fonts_ready

    import os
    for regular in _FONT_PATHS:
        if not os.path.exists(regular):
            continue
        pdfmetrics.registerFont(TTFont(FONT_NAME, regular))
        for bold in _BOLD_PATHS:
            if os.path.exists(bold):
                pdfmetrics.registerFont(TTFont(f"{FONT_NAME}-Bold", bold))
                break
        _fonts_ready = True
        return True

    _fonts_ready = False
    return False


def fa(text) -> str:
    """متن فارسی را برای نمایش درست در PDF آماده می‌کند (شکل‌دهی + RTL).

    اعداد و متن لاتین باید از این تابع عبور کنند تا ترتیب کلی جمله حفظ شود،
    ولی خودِ ارقام تغییر نمی‌کنند (فارسی‌سازی ارقام جداگانه با to_persian_digits).
    """
    if text is None:
        return ""
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "faTitle", parent=base["Title"], fontName=FONT_NAME, fontSize=20,
            textColor=ACCENT, alignment=TA_CENTER, spaceAfter=4 * mm,
        ),
        "subtitle": ParagraphStyle(
            "faSub", parent=base["Normal"], fontName=FONT_NAME, fontSize=10,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=8 * mm,
        ),
        "h2": ParagraphStyle(
            "faH2", parent=base["Heading2"], fontName=FONT_NAME, fontSize=13,
            textColor=colors.HexColor("#111827"), alignment=TA_RIGHT,
            spaceBefore=6 * mm, spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            "faBody", parent=base["Normal"], fontName=FONT_NAME, fontSize=9.5,
            alignment=TA_RIGHT, leading=15,
        ),
    }


def _fmt_money(value) -> str:
    """عدد را با جداکننده هزارگان و ارقام فارسی برمی‌گرداند."""
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return to_persian_digits("0")
    return to_persian_digits(f"{n:,.0f}")


def _fmt_pct(value) -> str:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return to_persian_digits("0.00") + "٪"
    return to_persian_digits(f"{n:.2f}") + "٪"


def _rtl_table(data: List[List[str]], col_widths: List[float]) -> Table:
    """جدولی با هدر آبی و سلول‌های راست‌چین فارسی."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    return table


def build_portfolio_report(
    portfolio: dict,
    positions: List[dict],
    trades: Optional[List[dict]] = None,
    risk: Optional[dict] = None,
    insights: Optional[List[str]] = None,
) -> bytes:
    """PDF فارسی گزارش پرتفوی را می‌سازد و بایت‌های آن را برمی‌گرداند.

    ساخت در حافظه انجام می‌شود (بدون فایل موقت) تا endpoint بتواند مستقیم
    `Response(content=..., media_type="application/pdf")` بدهد.
    """
    if not register_fonts():
        raise RuntimeError(
            "فونت فارسی پیدا نشد — گزارش PDF فارسی بدون فونت TTF فارسی قابل ساخت نیست"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=14 * mm, leftMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="گزارش پرتفوی",
    )
    st = _styles()
    story = []

    name = portfolio.get("name") or "پرتفوی"
    story.append(Paragraph(fa(f"گزارش پرتفوی — {name}"), st["title"]))
    story.append(Paragraph(
        fa(f"تاریخ گزارش: {to_jalali_str()}"), st["subtitle"],
    ))

    # ── خلاصه ──
    total_value = portfolio.get("total_value", 0)
    cash = portfolio.get("current_cash", 0)
    initial = portfolio.get("initial_cash", 0)
    try:
        pnl = float(total_value or 0) - float(initial or 0)
        pnl_pct = (pnl / float(initial) * 100) if initial else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        pnl, pnl_pct = 0.0, 0.0

    story.append(Paragraph(fa("خلاصه"), st["h2"]))
    summary = [
        [fa("ارزش کل"), fa("موجودی نقد"), fa("سرمایه اولیه"), fa("سود/زیان")],
        [
            _fmt_money(total_value), _fmt_money(cash),
            _fmt_money(initial),
            f"{_fmt_money(abs(pnl))} ({_fmt_pct(pnl_pct)})",
        ],
    ]
    story.append(_rtl_table(summary, [doc.width / 4.0] * 4))
    story.append(Spacer(1, 6 * mm))

    # ── پوزیشن‌ها ──
    if positions:
        story.append(Paragraph(fa("پوزیشن‌های باز"), st["h2"]))
        rows = [[
            fa("نماد"), fa("تعداد"), fa("میانگین قیمت"),
            fa("قیمت جاری"), fa("ارزش بازار"), fa("سود/زیان"),
        ]]
        for p in positions:
            upnl = p.get("unrealized_pnl", 0)
            rows.append([
                fa(p.get("symbol", "")),
                to_persian_digits(p.get("quantity", 0)),
                _fmt_money(p.get("average_price", 0)),
                _fmt_money(p.get("current_price", 0)),
                _fmt_money(p.get("market_value", 0)),
                _fmt_money(upnl),
            ])
        table = _rtl_table(rows, [doc.width / 6.0] * 6)
        # ستون سود/زیان را رنگ‌آمیزی می‌کنیم (آخرین ستون)
        for idx, p in enumerate(positions, start=1):
            try:
                val = float(p.get("unrealized_pnl", 0) or 0)
            except (TypeError, ValueError):
                val = 0.0
            table.setStyle(TableStyle([
                ("TEXTCOLOR", (5, idx), (5, idx), SUCCESS if val >= 0 else DANGER),
            ]))
        story.append(table)
        story.append(Spacer(1, 6 * mm))

    # ── ریسک ──
    if risk:
        story.append(Paragraph(fa("شاخص‌های ریسک"), st["h2"]))
        risk_rows = [[fa("شاخص"), fa("مقدار")]]
        labels = [
            ("var_1d", "ارزش در معرض خطر (۱ روزه)"),
            ("var_1w", "ارزش در معرض خطر (۱ هفته‌ای)"),
            ("sharpe_ratio", "نسبت شارپ"),
            ("max_drawdown", "حداکثر افت سرمایه"),
        ]
        for key, label in labels:
            if risk.get(key) is not None:
                risk_rows.append([fa(label), _fmt_money(risk[key])])
        if len(risk_rows) > 1:
            story.append(_rtl_table(risk_rows, [doc.width * 0.62, doc.width * 0.38]))
            story.append(Spacer(1, 6 * mm))

    # ── معاملات ──
    if trades:
        story.append(Paragraph(fa("آخرین معاملات"), st["h2"]))
        trows = [[
            fa("تاریخ"), fa("نماد"), fa("نوع"), fa("تعداد"), fa("قیمت"), fa("مبلغ"),
        ]]
        for t in trades:
            ttype = "خرید" if str(t.get("trade_type", "")).upper() == "BUY" else "فروش"
            tdate = t.get("trade_date")
            trows.append([
                to_persian_digits(str(tdate)[:10]) if tdate else "-",
                fa(t.get("symbol", "")),
                fa(ttype),
                to_persian_digits(t.get("quantity", 0)),
                _fmt_money(t.get("price", 0)),
                _fmt_money(t.get("total_amount", 0)),
            ])
        story.append(_rtl_table(trows, [doc.width / 6.0] * 6))
        story.append(Spacer(1, 6 * mm))

    # ── بینش‌های تحلیلی ──
    if insights:
        story.append(Paragraph(fa("تحلیل و بینش"), st["h2"]))
        for line in insights:
            story.append(Paragraph("• " + fa(line), st["body"]))
            story.append(Spacer(1, 1.5 * mm))

    doc.build(story)
    return buf.getvalue()
