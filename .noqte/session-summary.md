---

### هدف اصلی کاربر
توسعه پروژه **اختاپوس (Octopus / Findash)** — یک داشبورد مالی جامع با تمرکز بر بازار ایران. شامل دارایی‌های ایرانی (طلا، سکه، دلار، نقره، مسکن، کریپتو)، لوکالیزیشن فارسی/RTL، بک‌تستینگ و قابلیت‌های ریل‌تایم.

---

### وضعیت فعلی پروژه
- **ریپو:** `massoudsh/Findash` — شاخه `main`
- **آخرین commit push شده:** `642b3ea` — `feat: complete iranian market platform (TASK-002~006)`
- **بک‌لاگ:** همه ۶ تسک اصلی (TASK-001 تا TASK-006) کامل و push شده‌اند
- **در حال اجرا:** مرحله اجرای **Steps 1–6** جدید (WebSocket، Portfolio، Alert، News، UI/UX، pytest) که **نیمه‌کاره** ماند — تا Step 2 پیش رفت ولی transcript قطع شد

---

### فایل‌های مهم و تغییرات آن‌ها

**پروژه اصلی (در `/project/frontend-nextjs/`):**
- `src/app/dashboard/page.tsx` — داشبورد اصلی (در حال ویرایش برای افزودن تب Trades)
- `src/app/portfolio/page.tsx` — صفحه پورتفولیو
- `src/components/realtime/realtime-content.tsx` — **بازنویسی شد** با WebSocket hook جدید
- `src/lib/hooks/use-market-ws.ts` — **جدید** — hook ریل‌تایم WebSocket
- `src/components/portfolio/trade-tracker.tsx` — **جدید** — ثبت خرید/فروش با P&L

**ماژول‌های ساخته‌شده (در `MyProjects/Octopus/Modules/`):**
- `src/schemas/asset_schema.py` — Pydantic types
- `src/models/asset.py` — 4 جدول SQLAlchemy + seed 16 نماد
- `src/services/asset_service.py` — fetch از tgju.org + Redis cache (60s TTL)
- `src/api/routes/assets.py` — 5 endpoint: list، detail، history، usd-rate، portfolio
- `src/main_refactored.py` — ثبت assets_router
- `src/migrations/add_asset_tables.sql` — TimescaleDB + seed
- `frontend-nextjs/src/context/CurrencyContext.tsx` — سوئیچ IRT/USD
- `frontend-nextjs/src/lib/locale.ts` — تاریخ شمسی + اعداد فارسی
- `frontend-nextjs/src/app/layout.tsx` — RTL + Vazirmatn
- `frontend-nextjs/src/app/dashboard/_components/IranMacroWidget.tsx`
- `frontend-nextjs/src/app/dashboard/_components/CurrencyComparisonCard.tsx`
- `frontend-nextjs/src/app/backtesting/_components/IranAssetBacktest.tsx`
- `tests/test_assets_api.py` — 15 تست endpoint
- `tests/test_asset_service.py` — تست cache/TGJU/error

**ویکی Noqte:**
- `.noqte/wiki/backlog.md` — همه TASK-001~006 با وضعیت Done
- `.noqte/wiki/pending-issues.md` — 5 issue آماده ایجاد در GitHub UI
- `.noqte/wiki/entities/assets-feature.md`، `frontend.md`، `backend.md`، `orchestrator.md`، `data-layer.md`
- `.noqte/wiki/concepts/trading-flow.md`، `data-pipeline.md`

---

### تصمیمات معماری/طراحی
- **پروژه اصلی در `/project/frontend-nextjs/`** — جدا از ماژول‌های ساخته‌شده که در `MyProjects/Octopus/Modules/` هستند
- `MyProjects/Octopus/Modules` به‌عنوان **submodule** وجود داشت — gitlink حذف و به‌عنوان directory معمولی add شد
- فونت: **Vazirmatn** برای فارسی، `dir="rtl"` در `layout.tsx`
- Cache: **Redis 60 ثانیه** برای قیمت‌های tgju.org
- DB: **TimescaleDB** (PostgreSQL) برای time-series داده‌های قیمت
- WebSocket hook: `use-market-ws.ts` با reconnect خودکار و fallback به polling

---

### کارهای انجام‌شده (به ترتیب زمانی)

1. **2026-06-27:** Bootstrap ویکی پروژه — overview، 4 entity، 2 concept
2. **2026-06-27:** ساخت بک‌لاگ — TASK-001 تا TASK-006
3. **2026-06-27:** اجرای TASK-001 (سکشن دارایی‌های ایرانی) — backend کامل + frontend کامل
4. **2026-06-27:** اجرای 001e/001f/001g — widget داشبورد، portfolio tracker، main_refactored.py
5. **2026-06-27:** Commit `feat: adding persian market assets` و push
6. **2026-06-28:** اجرای TASK-002~006 یکجا — داشبورد ایرانی، RTL، CurrencyContext، بک‌تستینگ، تست‌ها
7. **2026-06-28:** Commit `feat: complete iranian market platform` و push — `642b3ea`
8. **2026-06-29:** شروع Steps 1–6 جدید:
   - **Step 1 (WebSocket):** `use-market-ws.ts` ساخته شد، `realtime-content.tsx` بازنویسی شد ✅
   - **Step 2 (Portfolio):** `trade-tracker.tsx` ساخته شد ✅ — در حال افزودن تب به `dashboard/page.tsx` قطع شد

---

### کارهای باقی‌مانده / گام بعدی

درخواست کاربر در **2026-06-29T05:58:38** این بود:
> "do the steps from 1 to 6 and run the app local and sync local and github."

**Steps ناتمام:**
- **Step 2 (ادامه):** افزودن تب Trades به `dashboard/page.tsx` — در حال خواندن فایل قطع شد
- **Step 3 — Alert سیستم:** قیمت هدف + اطلاع‌رسانی (ایمیل/Telegram)
- **Step 4 — صفحه News:** اخبار بازار ایران (RSS/scrape)
- **Step 5 — UI/UX:** dark mode کامل، انیمیشن، mobile-friendly
- **Step 6 — pytest:** اجرای تست‌های موجود و validate
- **اجرای local:** `npm run dev` در `frontend-nextjs` و بالا آوردن backend
- **sync با GitHub:** commit نهایی و push

---

### نکات، گاتچاها، و درخواست‌های اخیر کاربر

- **`gh` CLI نصب نیست** — issue ها باید از GitHub UI ایجاد شوند (فایل `.noqte/wiki/pending-issues.md` آماده است)
- پروژه اصلی frontend در **`/project/frontend-nextjs/`** است — نه در `MyProjects/Octopus/Modules/`
- فایل `dashboard/page.tsx` در `/project/frontend-nextjs/src/app/dashboard/page.tsx` بود که در حال خواندن برای ویرایش قطع شد — **گام بعدی** همین ویرایش است
- آخرین درخواست verbatim: **"do the steps from 1 to 6 and run the app local and sync local and github."**
- ساختار navigation از `navigation-wrapper.tsx` مدیریت می‌شود
- toast/notification سیستم موجود است: `notification-center.tsx` و `toast.tsx`