---

### هدف اصلی کاربر
ساخت داشبورد مالی همه‌کاره به نام **اختاپوس** (Octopus) — اپلیکیشن موبایل/وب با تمرکز بر بازار مالی ایران (طلا، سکه، دلار، نقره، مسکن، ارز دیجیتال) روی ریپو `massoudsh/Findash`.

---

### وضعیت فعلی پروژه
- **همه TASK-001 تا TASK-006 کامل و push شده‌اند** (آخرین commit: `642b3ea`)
- کاربر در آخرین پیام (`2026-06-29T05:58:38`) خواسته مراحل ۱ تا ۶ (WebSocket، Portfolio Tracker، Alert، News، UI/UX، pytest) انجام شوند، اپ local اجرا شود و با GitHub sync شود
- دستیار شروع به بررسی ساختار پروژه واقعی در `/project/` کرده (نه Modules) و در حال پیاده‌سازی مرحله ۱ (WebSocket hook) بوده که مکالمه قطع شده

---

### فایل‌های مهم و تغییرات آن‌ها

**ساختار واقعی پروژه (در `/project/`):**
- `/project/frontend-nextjs/src/app/` — صفحات Next.js (dashboard، realtime، portfolio، notifications، backtesting)
- `/project/frontend-nextjs/src/components/` — کامپوننت‌ها (dashboard، portfolio، realtime، navigation، ui)
- `/project/frontend-nextjs/src/lib/` — utilitiesها شامل `i18n/locale-context.ts`، `backend-url.ts`، `utils.ts`
- `/project/src/api/routes/` — FastAPI routes
- `/project/src/realtime/` — سرویس realtime
- `/project/tests/` — تست‌های backend

**فایل‌های ساخته‌شده در Modules (commit شده):**
- `MyProjects/Octopus/Modules/src/schemas/asset_schema.py`
- `MyProjects/Octopus/Modules/src/models/asset.py` — 4 جدول SQLAlchemy + 16 نماد seed data
- `MyProjects/Octopus/Modules/src/services/asset_service.py` — fetch از tgju.org با Redis cache
- `MyProjects/Octopus/Modules/src/api/routes/assets.py` — 5 endpoint
- `MyProjects/Octopus/Modules/src/main_refactored.py` — ثبت router
- `MyProjects/Octopus/Modules/src/migrations/add_asset_tables.sql`
- `MyProjects/Octopus/Modules/frontend-nextjs/src/app/assets/page.tsx` + کامپوننت‌ها
- `MyProjects/Octopus/Modules/frontend-nextjs/src/context/CurrencyContext.tsx`
- `MyProjects/Octopus/Modules/frontend-nextjs/src/lib/locale.ts`
- `MyProjects/Octopus/Modules/frontend-nextjs/src/app/layout.tsx`
- `MyProjects/Octopus/Modules/frontend-nextjs/src/app/dashboard/_components/IranMacroWidget.tsx`
- `MyProjects/Octopus/Modules/frontend-nextjs/src/app/backtesting/page.tsx`
- `MyProjects/Octopus/Modules/tests/test_assets_api.py` (15 تست)، `test_asset_service.py`

**فایل‌های ساخته‌شده برای مراحل جدید (ناقص — مکالمه قطع شد):**
- `frontend-nextjs/src/lib/hooks/use-market-ws.ts` — WebSocket hook (نوشته شد)
- `frontend-nextjs/src/components/realtime/realtime-content.tsx` — بازنویسی با hook (نوشته شد)
- `frontend-nextjs/src/components/portfolio/trade-tracker.tsx` — در حال نوشتن بود
- `.noqte/wiki/pending-issues.md` — لیست issue های ثبت‌نشده

**ویکی `.noqte/wiki/`:**
- `overview.md`، `index.md`، `backlog.md`، `log.md`، `pending-issues.md`
- entities: `frontend.md`، `backend.md`، `orchestrator.md`، `data-layer.md`، `assets-feature.md`
- concepts: `trading-flow.md`، `data-pipeline.md`

---

### تصمیمات معماری/طراحی
- پروژه اصلی: Next.js 15 frontend + FastAPI backend + PostgreSQL/TimescaleDB + Redis + Kafka
- `MyProjects/Octopus/Modules` در git به‌عنوان submodule بود؛ دستیار آن را detach و به‌عنوان directory معمولی add کرد
- داده قیمت دارایی‌های ایران از `tgju.org` با Redis cache 60ثانیه‌ای
- RTL سراسری با فونت Vazirmatn و تاریخ شمسی
- CurrencyContext برای سوئیچ IRT/USD در کل اپ
- `gh` CLI روی محیط نصب نیست — issue ها باید از GitHub UI ساخته شوند

---

### کارهای انجام‌شده (به ترتیب زمانی)

1. **2026-06-27:** Bootstrap ویکی پروژه (overview، 4 entity، 2 concept)
2. **2026-06-27:** تعریف بک‌لاگ TASK-001 تا TASK-006
3. **2026-06-27:** پیاده‌سازی TASK-001 (بخش اول) — backend کامل (schema، model، service، routes، migration) + frontend (AssetCard، AssetGrid، AssetPriceChart، AssetSummaryBar، page.tsx)
4. **2026-06-27:** پیاده‌سازی TASK-001 (بخش دوم) — AssetsDashboardWidget، portfolio/page.tsx، main_refactored.py — **commit و push: `feat: adding persian market assets`**
5. **2026-06-28:** اجرای TASK-002 تا TASK-006 یکجا: IranMacroWidget، CurrencyComparisonCard، CurrencyContext، locale.ts، IranAssetBacktest، unit tests، README update — **commit و push: `642b3ea` — `feat: complete iranian market platform (TASK-002~006)`**
6. **2026-06-29:** بررسی ساختار واقعی `/project/` (کشف frontend-nextjs موجود با WebSocket، portfolio، notifications، backtesting از قبل)
7. **2026-06-29:** نوشتن `use-market-ws.ts` hook برای WebSocket ریل‌تایم
8. **2026-06-29:** بازنویسی `realtime-content.tsx` با WebSocket hook
9. **2026-06-29:** شروع نوشتن `trade-tracker.tsx` — **مکالمه اینجا قطع شد**

---

### کارهای باقی‌مانده / گام بعدی

**ناتمام از آخرین session (باید از همینجا ادامه داد):**
- [ ] تکمیل `trade-tracker.tsx` (Step 2 — Portfolio Trade Tracker با خرید/فروش و P&L)
- [ ] اضافه کردن tab «Trades» به dashboard یا portfolio page
- [ ] Step 3 — Alert سیستم (قیمت هدف، email/telegram)
- [ ] Step 4 — صفحه News (اخبار بازار ایران)
- [ ] Step 5 — UI/UX بهبود (dark mode کامل، انیمیشن، mobile-friendly)
- [ ] Step 6 — اجرای pytest و validate کردن تست‌ها
- [ ] اجرای اپ local (`npm run dev` برای frontend، uvicorn برای backend)
- [ ] commit و push نهایی به `massoudsh/Findash` → `main`

**از بک‌لاگ pending-issues.md (برای GitHub UI):**
- TASK-002 تا TASK-006 issue ساخته نشده‌اند (gh CLI نصب نیست)

---

### نکات، گاتچاها، و درخواست‌های اخیر کاربر

**درخواست verbatim آخرین کاربر:**
> "do the steps from 1 to 6 and run the app local and sync local and github."
> انتخاب مراحل: مرحله 1 تا 6، اجرای برنامه محلی: بله، همگام‌سازی با GitHub: بله، نام شاخه: massoudsh/Findash

**گاتچاها:**
- ساختار پروژه **دوگانه** است: فایل‌های commit‌شده در `MyProjects/Octopus/Modules/` هستند، اما پروژه اصلی در `/project/frontend-nextjs/` و `/project/src/` است — باید تغییرات جدید (Steps 1-6) روی `/project/` اعمال شوند نه Modules
- `gh` CLI نصب نیست — برای GitHub issues از UI استفاده شود
- frontend موجود قبلاً WebSocket route، notification center، backtesting page داشت؛ باید با کد جدید تلفیق شود نه بازنویسی کامل
- `use-market-ws.ts` و `realtime-content.tsx` نوشته شده اما هنوز commit نشده‌اند