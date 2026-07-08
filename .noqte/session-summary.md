---

### هدف اصلی کاربر
توسعه داشبورد مالی جامع به نام **اختاپوس** (Findash) — یک وب‌اپلیکیشن Next.js + FastAPI با تمرکز بر بازار ایران (طلا، سکه، دلار، نقره، مسکن، کریپتو)، با پشتیبانی از RTL فارسی، واحد تومان، بک‌تست، ریل‌تایم، و پورتفولیو.

---

### وضعیت فعلی پروژه
- **مخزن:** `massoudsh/Findash` — شاخه `main`
- **آخرین commit push‌شده:** `642b3ea` — `feat: complete iranian market platform (TASK-002~006)`
- همه TASK-001 تا TASK-006 کامل و push شده‌اند
- جلسه فعلی (2026-06-29) در حال پیاده‌سازی مراحل 1 تا 6 ادامه توسعه بود اما **در میانه کار قطع شد** — Step 1 (WebSocket) و Step 2 (Portfolio Trade Tracker) تا حدی شروع شده بودند ولی تأیید نهایی نشدند

---

### فایل‌های مهم و تغییرات آن‌ها

**Backend (پروژه اصلی `/project/src/`)**
- `src/api/routes/assets.py` — 5 endpoint: list، detail، history، usd-rate، portfolio
- `src/services/asset_service.py` — fetch از tgju.org، cache Redis (60s)
- `src/models/asset.py` — 4 جدول SQLAlchemy + seed 16 نماد
- `src/schemas/asset_schema.py` — Pydantic types
- `src/main_refactored.py` — router assets ثبت شده
- `src/migrations/add_asset_tables.sql` — TimescaleDB hypertable + seed

**Frontend (`/project/frontend-nextjs/src/`)**
- `app/assets/page.tsx` — صفحه دارایی‌های ایرانی با tabs
- `app/assets/_components/` — AssetCard، AssetPriceChart، AssetGrid، AssetSummaryBar
- `app/dashboard/_components/IranMacroWidget.tsx` — 6 شاخص کلان
- `app/dashboard/_components/CurrencyComparisonCard.tsx`
- `app/backtesting/_components/IranAssetBacktest.tsx` — 3 استراتژی
- `app/portfolio/page.tsx` — پورتفولیو با تب دارایی‌های فیزیکی
- `context/CurrencyContext.tsx` — سوئیچ IRT/USD
- `components/ui/CurrencyToggle.tsx`
- `lib/locale.ts` — `formatJalali`، `toPersianDigits`
- `app/layout.tsx` — `dir="rtl"` + فونت Vazirmatn
- `lib/hooks/use-market-ws.ts` — WebSocket hook (نوشته شده، تأیید نشده)
- `components/realtime/realtime-content.tsx` — بازنویسی شده با WebSocket hook
- `components/portfolio/trade-tracker.tsx` — شروع شده، تکمیل نشده

**ویکی (`.noqte/wiki/`)**
- `backlog.md` — همه TASK-001 تا TASK-006 با وضعیت Done
- `pending-issues.md` — ایشوهای باقی‌مانده برای GitHub UI
- `entities/assets-feature.md`، `entities/frontend.md`، `entities/backend.md`، `entities/orchestrator.md`، `entities/data-layer.md`
- `concepts/trading-flow.md`، `concepts/data-pipeline.md`

**فایل گیت‌مول:** `MyProjects/Octopus/Modules` که یک submodule بود، به‌صورت دایرکتوری معمولی add شد

---

### تصمیمات معماری/طراحی
- **Source of Truth:** پروژه اصلی در `/project/` است (نه `MyProjects/Octopus/Modules`) — فایل‌های backend در `/project/src/`، frontend در `/project/frontend-nextjs/`
- **Data source برای قیمت‌ها:** tgju.org با cache Redis 60 ثانیه
- **DB:** PostgreSQL/TimescaleDB برای time-series قیمت‌ها
- **واحد پولی:** IRT (تومان) به‌عنوان پیش‌فرض، با CurrencyContext برای سوئیچ به USD
- **WebSocket:** برنامه‌ریزی شده برای ریل‌تایم قیمت (hook نوشته شده، backend endpoint تأیید نشده)
- **RTL:** سراسری در layout.tsx با فونت Vazirmatn

---

### کارهای انجام‌شده (به ترتیب زمانی)

1. **2026-06-27:** ویکی پروژه bootstrap شد (overview، entities، concepts)
2. **2026-06-27:** بک‌لاگ ساخته شد (TASK-001 تا TASK-006)
3. **2026-06-27:** TASK-001 پیاده‌سازی شد — سکشن دارایی‌های ایرانی (backend + frontend کامل)
4. **2026-06-27:** TASK-001 زیرتسک‌های باقیمانده (001e widget داشبورد، 001f portfolio tracker، 001g ثبت router) انجام و push شد (`commit: feat: adding persian market assets`)
5. **2026-06-28:** TASK-002 تا TASK-006 یکجا پیاده‌سازی و push شد (`commit: feat: complete iranian market platform`)
   - TASK-002: IranMacroWidget + CurrencyComparisonCard
   - TASK-003: RTL + Jalali + Vazirmatn
   - TASK-004: CurrencyContext + CurrencyToggle
   - TASK-005: بک‌تست 3 استراتژی
   - TASK-006: تست‌های pytest + README
6. **2026-06-29 (جلسه فعلی، ناتمام):**
   - Step 1 WebSocket: `lib/hooks/use-market-ws.ts` نوشته شد، `realtime-content.tsx` بازنویسی شد
   - Step 2 Portfolio Trade Tracker: `components/portfolio/trade-tracker.tsx` شروع شد
   - در حال خواندن `dashboard/page.tsx` برای افزودن تب Trades بود که جلسه قطع شد

---

### کارهای باقی‌مانده / گام بعدی

کاربر درخواست کرد همه ۶ مرحله را تکمیل کند:
1. **WebSocket ریل‌تایم** — hook نوشته شده، باید backend WebSocket endpoint هم بررسی/ساخته شود
2. **Portfolio Trade Tracker** — `trade-tracker.tsx` شروع شده، باید به `dashboard/page.tsx` و `portfolio/page.tsx` اضافه شود
3. **Alert سیستم** — اعلان قیمت هدف (email/telegram) — هنوز شروع نشده
4. **صفحه News** — اخبار بازار ایران — هنوز شروع نشده
5. **UI/UX بهبود** — dark mode، انیمیشن، mobile-friendly — هنوز شروع نشده
6. **اجرای pytest** — تأیید تست‌ها — هنوز اجرا نشده
7. **sync با GitHub** — commit + push نهایی — هنوز انجام نشده

---

### نکات، گاتچاها، و درخواست‌های اخیر کاربر

- **درخواست اخیر verbatim:** `"do the steps from 1 to 6 and run the app local and sync local and github."` + `"انتخاب مراحل: مرحله 1، مرحله 2، مرحله 3، مرحله 4، مرحله 5، مرحله 6"` + `"اجرای برنامه محلی: بله"` + `"همگام‌سازی با GitHub: بله"`
- **gh CLI نصب نیست** در محیط — issue ها باید از GitHub UI ساخته شوند (فایل `pending-issues.md` موجود است)
- پروژه در `/project/` است نه در `MyProjects/Octopus/Modules` — کدهای جلسه قبلی در مسیر اشتباه نوشته شدند اما کدهای جلسه 29 ام در مسیر درست `/project/frontend-nextjs/` نوشته می‌شوند
- `navigation-wrapper.tsx` در git status به‌عنوان modified نشان داده می‌شود
- `session-summary.md` در git status modified است