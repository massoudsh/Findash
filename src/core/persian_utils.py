"""ابزارهای کوچک فارسی‌سازی مشترک (تاریخ شمسی + اعداد فارسی) — بدون وابستگی خارجی جدید.

الگوریتم تبدیل میلادی→شمسی، الگوریتم عمومی و شناخته‌شده (بر پایه سال کبیسه‌ی
تقویم جلالی) است؛ نیازی به نصب پکیج سنگین جدید (jdatetime/khayyam) نیست.
"""

from datetime import datetime

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


def to_persian_digits(value) -> str:
    """اعداد لاتین یک رشته/عدد را به ارقام فارسی تبدیل می‌کند."""
    s = str(value)
    return "".join(_PERSIAN_DIGITS[int(ch)] if ch.isdigit() else ch for ch in s)


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple:
    """تبدیل تاریخ میلادی به شمسی. برمی‌گرداند: (سال, ماه, روز)."""
    g_days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    j_days_in_month = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]

    gy2 = gy - 1600
    gm2 = gm - 1
    gd2 = gd - 1

    g_day_no = 365 * gy2 + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
    for i in range(gm2):
        g_day_no += g_days_in_month[i]
    if gm2 > 1 and ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        g_day_no += 1
    g_day_no += gd2

    j_day_no = g_day_no - 79

    j_np = j_day_no // 12053
    j_day_no %= 12053

    jy = 979 + 33 * j_np + 4 * (j_day_no // 1461)
    j_day_no %= 1461

    if j_day_no >= 366:
        jy += (j_day_no - 1) // 365
        j_day_no = (j_day_no - 1) % 365

    for i in range(11):
        if j_day_no < j_days_in_month[i]:
            jm = i + 1
            jd = j_day_no + 1
            break
        j_day_no -= j_days_in_month[i]
    else:
        jm = 12
        jd = j_day_no + 1

    return jy, jm, jd


def to_jalali_str(dt: datetime = None) -> str:
    """`YYYY/MM/DD` شمسی با ارقام فارسی — برای نمایش تاریخ در گزارش‌ها."""
    dt = dt or datetime.utcnow()
    jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
    return to_persian_digits(f"{jy}/{jm:02d}/{jd:02d}")
