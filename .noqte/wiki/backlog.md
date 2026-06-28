# Backlog — اختاپوس Trading Platform

> آخرین به‌روزرسانی: 2026-06-27

---

## اتفاقات اخیر
- ویکی پروژه bootstrap شد (overview، index، 4 entity، 2 concept)
- پروژه در حالت غیرفعال قرار دارد

---

## 🔴 High Priority

### TASK-001 — سکشن دارایی‌های ایرانی (Assets Section)
**وضعیت:** `✅ Done — 2026-06-27`
**اندازه:** L
**نوع:** Feature جدید

**توضیح:**
افزودن سکشن جدید «دارایی‌ها» برای پوشش بازارهای ایرانی و دارایی‌های فیزیکی در کنار بازارهای مالی فعلی.

**دارایی‌های هدف:**
| دارایی | نماد | منبع داده پیشنهادی |
|--------|------|-------------------|
| طلا (18 عیار) | XAU18 | tgju.org / nobitex |
| سکه بهار آزادی | COIN | tgju.org |
| نیم‌سکه | HALFCOIN | tgju.org |
| ربع‌سکه | QUARTERCOIN | tgju.org |
| دلار آمریکا | USD | tgju.org |
| یورو | EUR | tgju.org |
| درهم امارات | AED | tgju.org |
| نقره | XAG | tgju.org |
| مسکن (شاخص) | REALESTATE-IR | اطلاعات دستی/بانک مرکزی |
| بیت‌کوین (تومانی) | BTC-IRT | nobitex / wallex |

**زیرتسک‌ها:**
- [x] TASK-001a — طراحی UI صفحه `/assets` (کارت هر دارایی + نمودار + جزئیات)
- [x] TASK-001b — API endpoint: `/api/assets` (لیست، قیمت لحظه‌ای، تاریخچه)
- [x] TASK-001c — اتصال به منابع داده خارجی (tgju.org API)
- [x] TASK-001d — مدل DB برای ذخیره تاریخچه قیمت دارایی‌ها (TimescaleDB)
- [x] TASK-001e — widget دارایی‌ها در داشبورد اصلی
- [x] TASK-001f — افزودن دارایی‌ها به portfolio tracker
- [x] TASK-001g — ثبت router در `main_refactored.py`

---

## 🟠 Medium Priority

### TASK-002 — بهبود داشبورد اصلی
**وضعیت:** `✅ Done — 2026-06-28`
**اندازه:** M
**نوع:** Enhancement

**توضیح:**
داشبورد فعلی فقط بازارهای بین‌المللی دارد. نیاز به:
- [x] TASK-002a — ویجت خلاصه دارایی‌های ایرانی (بعد از TASK-001)
- [x] TASK-002b — نرخ تورم و شاخص‌های کلان ایران (IranMacroWidget)
- [x] TASK-002c — کارت مقایسه‌ای: ارزش دارایی به ریال vs دلار (CurrencyComparisonCard)

### TASK-003 — لوکالیزیشن فارسی (RTL)
**وضعیت:** `✅ Done — 2026-06-28`
**اندازه:** M
**نوع:** Enhancement

**توضیح:**
پلتفرم فعلاً کاملاً انگلیسی است. برای بازار ایران نیاز به:
- [x] TASK-003a — پشتیبانی از RTL در UI (layout.tsx با dir=rtl)
- [x] TASK-003b — ترجمه برچسب‌ها و منوها به فارسی
- [x] TASK-003c — نمایش اعداد به فارسی (toPersianDigits در locale.ts)
- [x] TASK-003d — فرمت تاریخ شمسی (formatJalali در locale.ts)

### TASK-004 — واحد پولی تومانی
**وضعیت:** `✅ Done — 2026-06-28`
**اندازه:** S
**نوع:** Feature

**توضیح:**
- [x] TASK-004a — افزودن IRT (تومان) به CurrencyContext
- [x] TASK-004b — CurrencyToggle در داشبورد — سوئیچ ت / $
- [x] TASK-004c — format() در CurrencyContext نمایش تومان و دلار هر دو

---

## 🟡 Low Priority

### TASK-005 — بهبود بک‌تستینگ
**وضعیت:** `✅ Done — 2026-06-28`
**اندازه:** L
**نوع:** Enhancement
- [x] پشتیبانی از داده‌های تاریخی دارایی‌های ایرانی (IranAssetBacktest)
- [x] استراتژی‌های Buy & Hold، DCA، Relative Strength
- [x] صفحه `/backtesting` با تب ایرانی + جهانی

### TASK-006 — مستندسازی و تست
**وضعیت:** `✅ Done — 2026-06-28`
**اندازه:** M
**نوع:** Chore
- [x] unit test کامل برای /api/assets endpoints (test_assets_api.py)
- [x] unit test برای AssetService و cache logic (test_asset_service.py)
- [x] README آپدیت با سکشن Iranian Market Features

---

## وضعیت کلی

| تسک | اندازه | اولویت | وضعیت |
|-----|--------|--------|--------|
| TASK-001 دارایی‌های ایرانی | L | 🔴 High | ✅ Done |
| TASK-002 بهبود داشبورد | M | 🟠 Medium | ✅ Done |
| TASK-003 لوکالیزیشن فارسی | M | 🟠 Medium | ✅ Done |
| TASK-004 واحد تومانی | S | 🟠 Medium | ✅ Done |
| TASK-005 بک‌تستینگ | L | 🟡 Low | ✅ Done |
| TASK-006 تست و مستندات | M | 🟡 Low | ✅ Done |
