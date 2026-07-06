---

### هدف اصلی کاربر
توسعه پروژه «اختاپوس» (Octopus/Findash) — یک داشبورد مالی همه‌کاره با تمرکز بر **بازار ایران** (طلا، سکه، دلار، نقره، مسکن، ارز دیجیتال). ریپو اصلی: `massoudsh/Findash` روی GitHub، شاخه `main`.

---

### وضعیت فعلی پروژه
- **همه TASK-001 تا TASK-006 کامل و push شده‌اند.**
- آخرین commit push‌شده: `642b3ea` — `feat: complete iranian market platform (TASK-002~006)`
- بک‌لاگ رسمی خالی است.
- در **آخرین session (2026-06-29)**، کاربر خواسته مراحل ۱ تا ۶ جدید (WebSocket، Portfolio Tracker بهتر، Alert، News، UI/UX، pytest) پیاده شود و اپ local اجرا و با GitHub sync شود.
- دستیار در حال بررسی ساختار پروژه واقعی (`/project/frontend-nextjs/` و `/project/src/`) بود و شروع پیاده‌سازی کرده بود، اما transcript ناتمام است — **Step 1 و Step 2 شروع شده‌اند اما کامل نشده‌اند.**

---

### فایل‌های مهم و تغییرات آن‌ها

**ساختار واقعی پروژه (در `/project/`):**
- `frontend-nextjs/src/app/` — صفحات Next.js (dashboard, portfolio, realtime, notifications, backtesting, assets)
- `frontend-nextjs/src/components/` — کامپوننت‌ها (dashboard, portfolio, realtime, navigation, ui)
- `frontend-nextjs/src/lib/` — utils, backend-url, i18n, hooks
- `src/api/routes/` — FastAPI route ها
- `src/realtime/` — WebSocket backend

**فایل‌های تازه ایجادشده (session آخر، ناتمام):**
- `frontend-nextjs/src/lib/hooks/use-market-ws.ts` — WebSocket hook جدید (نوشته شده)
- `frontend-nextjs/src/components/realtime/realtime-content.tsx` — بازنویسی شده با hook جدید (نوشته شده)
- `frontend-nextjs/src/components/portfolio/trade-tracker.tsx` — Portfolio trade tracker (شروع شده)

**فایل‌های موجود قبلی کلیدی:**
- `frontend-nextjs/src/components/portfolio/portfolio-content.tsx`
- `frontend-nextjs/src/app/dashboard/page.tsx`
- `frontend-nextjs/src/app/portfolio/page.tsx`
- `frontend-nextjs/src/app/realtime/page.tsx`
- `frontend-nextjs/src/components/navigation/navigation-wrapper.tsx`
- `frontend-nextjs/src/components/ui/theme-switcher.tsx`
- `frontend-nextjs/src/components/ui/notification-center.tsx`
- `frontend-nextjs/src/lib/i18n/locale-context` — i18n موجود

**فایل‌های ویکی:**
- `.noqte/wiki/backlog.md` — همه TASK-001~006 کامل
- `.noqte/wiki/pending-issues.md` — issue های ذخیره‌شده برای GitHub UI
- `.noqte/wiki/entities/assets-feature.md`
- `.noqte/wiki/log.md`

---

### تصمیمات معماری/طراحی
- **ساختار واقعی پروژه** در `/project/frontend-nextjs/` و `/project/src/` است (نه `MyProjects/Octopus/Modules/`)
- Frontend: Next.js 15 + TypeScript + RTL + فونت Vazirmatn
- Backend: FastAPI + Python
- DB: PostgreSQL/TimescaleDB + Redis (cache 60s TTL)
- داده قیمت: fetch از `tgju.org`
- WebSocket برای داده ریل‌تایم (backend موجود در `src/realtime/`)
- i18n موجود با `locale-context` + تاریخ شمسی + اعداد فارسی
- CurrencyContext: سوئیچ IRT/USD
- Toast system با `toast.tsx` موجود در `components/ui/`
- `gh` CLI موجود نیست، issue ها در `pending-issues.md` ذخیره می‌شوند

---

### کارهای انجام‌شده (به ترتیب زمانی)

**2026-06-27:**
- Bootstrap ویکی پروژه (overview، 4 entity، 2 concept)
- بک‌لاگ ساخته شد
- TASK-001 پیاده: backend (schemas, models, service, routes, migration) + frontend (AssetCard, AssetGrid, AssetPriceChart, AssetSummaryBar, assets/page.tsx)
- TASK-001 کامل: dashboard widget (AssetsDashboardWidget)، portfolio section (PortfolioAssetsSection)، main_refactored.py
- Commit و push: `feat: adding persian market assets`

**2026-06-28:**
- TASK-004: CurrencyContext + CurrencyToggle
- TASK-003: locale.ts (Jalali + toPersianDigits) + layout.tsx با RTL
- TASK-002: IranMacroWidget + CurrencyComparisonCard + dashboard page
- TASK-005: IranAssetBacktest (3 استراتژی) + backtesting/page.tsx
- TASK-006: test_assets_api.py (15 تست) + test_asset_service.py + README update
- Commit و push: `feat: complete iranian market platform (TASK-002~006)`

**2026-06-29 (session ناتمام):**
- کاربر: "do the steps from 1 to 6 and run the app local and sync local and github"
- دستیار ساختار واقعی پروژه را بررسی کرد (مسیرهای واقعی در `/project/`)
- **Step 1:** `use-market-ws.ts` hook نوشته شد، `realtime-content.tsx` بازنویسی شد
- **Step 2:** `trade-tracker.tsx` شروع شد، portfolio page بررسی شد
- **transcript قطع شده** — مراحل ۳ تا ۶ و اجرای local و push انجام نشده

---

### کارهای باقی‌مانده / گام بعدی

**مراحل ناتمام (از درخواست کاربر 2026-06-29):**
1. **Step 2 (ادامه)** — اضافه کردن تب «Trades» به `portfolio/page.tsx` یا `dashboard/page.tsx`
2. **Step 3** — Alert سیستم (اعلان قیمت هدف — email/telegram)
3. **Step 4** — صفحه News (اخبار بازار ایران)
4. **Step 5** — UI/UX بهبود (dark mode کامل، انیمیشن، mobile-friendly)
5. **Step 6** — اجرای pytest و validate کردن تست‌ها
6. **اجرای local** — راه‌اندازی اپ برای تست محلی
7. **Sync با GitHub** — commit و push همه تغییرات جدید

---

### نکات، گاتچاها، و درخواست‌های اخیر کاربر

**درخواست verbatim کاربر (آخرین پیام مهم):**
> "do the steps from 1 to 6 and run the app local and sync local and github."

**گاتچاها:**
- `MyProjects/Octopus/Modules/` submodule بود و با `git rm --cached` حذف شد — فایل‌های آن‌جا با `git add -f` اضافه شدند ولی ساختار واقعی کد در `/project/frontend-nextjs/` و `/project/src/` است
- `gh` CLI نصب نیست — issue ها را نمی‌توان از terminal ایجاد کرد
- toast system موجود است در `components/ui/toast.tsx`
- i18n با locale-context موجود است
- WebSocket backend در `src/realtime/` موجود است
- فایل‌های جدید Session آخر (use-market-ws.ts، realtime-content.tsx، trade-tracker.tsx) هنوز **commit و push نشده‌اند**