### هدف اصلی کاربر
پروژه «اختاپوس / Findash» — داشبورد مالی همه‌کاره با تمرکز بر بازار ایران.

### وضعیت فعلی (اصلاح‌شده — نسخه قبلی این فایل نادرست/stale بود)
همه‌ی ۶ قابلیت زیر قبلاً به‌طور کامل پیاده‌سازی، commit و روی `main` merge شده‌اند (کامیت `5eb7ac7`، پیام: `feat: websocket realtime, trade tracker, alerts, news page, UI/UX, tests`):

1. ✅ WebSocket ریل‌تایم — `frontend-nextjs/src/lib/hooks/use-market-ws.ts` + `realtime-content.tsx`
2. ✅ Portfolio Trade Tracker — `frontend-nextjs/src/components/portfolio/trade-tracker.tsx`، وصل به تب مستقل «My Trades» در `dashboard/page.tsx`
3. ✅ Price Alert System — `use-price-alerts.ts` + `alerts-panel.tsx` + `/alerts` page
4. ✅ Iran Market News — `/api/news/route.ts` (RSS از tgju.org) + `/news` page
5. ✅ UI/UX — light-mode CSS vars، انیمیشن‌ها، nav items جدید
6. ✅ تست‌ها — `tests/unit/test_standalone_logic.py` (۳۱ تست منطق خالص)

TASK-001 تا TASK-006 (دارایی‌های ایرانی، تومان/دلار، RTL، بک‌تست) هم قبلاً کامل و merge شده.

### تغییر انجام‌شده در همین سشن (۲۰۲۶-۰۷-۱۷)
- `trade-tracker.tsx` بهبود یافت: اکنون خودکار به `useMarketWS` وصل می‌شود و قیمت لحظه‌ای نمادهای معامله‌شده کاربر را می‌گیرد (به‌جای فقط avgCost)، با نشانگر Live/Polling در هدر.
- یک تلاش اولیه برای mount کردن دوباره‌ی `TradeTracker` داخل `portfolio-content.tsx` انجام و سپس به‌خاطر duplicate بودن با تب «My Trades» برگردانده شد (net diff صفر روی آن فایل).

### باقی‌مانده / گام بعدی احتمالی
- اجرای لوکال برنامه (dev server) — هنوز انجام نشده در این سشن.
- Sync نهایی با GitHub (commit + push) برای تغییر `trade-tracker.tsx` — هنوز انجام نشده، منتظر تأیید کاربر.
- ابزارهای `eslint`/`pytest` در این کانتینر عملاً در دسترس/فعال نیستند (pytest اصلاً نصب نیست، eslint خروجی سالم نمی‌دهد) — تست/lint سنگین باید روی سرور SSH انجام شود.
