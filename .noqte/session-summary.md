### هدف اصلی کاربر
توسعه و تکمیل پلتفرم مالی «اختاپوس» (Octopus) — یک داشبورد مالی همه‌کاره با تمرکز بر بازار ایران (دارایی‌های فیزیکی: طلا، سکه، دلار، نقره، مسکن، ارز دیجیتال).

---

### وضعیت فعلی پروژه
**همه ۶ تسک اصلی (TASK-001 تا TASK-006) کامل شده‌اند.**
- آخرین commit: `642b3ea` — `feat: complete iranian market platform (TASK-002~006)`
- push به `massoudsh/Findash` → شاخه `main` انجام شده است.
- بک‌لاگ خالی از تسک باز است.

---

### فایل‌های مهم و تغییرات آن‌ها

**Backend (Python/FastAPI):**
- `MyProjects/Octopus/Modules/src/schemas/asset_schema.py` — Pydantic types
- `MyProjects/Octopus/Modules/src/models/asset.py` — 4 جدول SQLAlchemy + seed برای ۱۶ نماد
- `MyProjects/Octopus/Modules/src/services/asset_service.py` — fetch از `tgju.org` + Redis cache (60s TTL)
- `MyProjects/Octopus/Modules/src/api/routes/assets.py` — 5 endpoint: list، detail، history، usd-rate، portfolio
- `MyProjects/Octopus/Modules/src/main_refactored.py` — ثبت assets_router
- `MyProjects/Octopus/Modules/src/migrations/add_asset_tables.sql` — migration + TimescaleDB hypertable

**Frontend (Next.js/TypeScript):**
- `src/lib/assets.ts` — types، `formatToman`، `formatChange`، fetch helpers
- `src/app/assets/page.tsx` — صفحه دارایی‌ها با tabs
- `src/app/assets/_components/AssetCard.tsx`، `AssetPriceChart.tsx`، `AssetGrid.tsx`، `AssetSummaryBar.tsx`
- `src/app/_components/AssetsDashboardWidget.tsx` — top 3 گینر/لوزر در داشبورد
- `src/app/page.tsx` — داشبورد اصلی
- `src/app/portfolio/page.tsx` + `PortfolioAssetsSection.tsx` — پورتفولیو فیزیکی
- `src/context/CurrencyContext.tsx` — سوئیچ سراسری IRT/USD
- `src/app/_components/CurrencyToggle.tsx` — دکمه «ت / $»
- `src/lib/locale.ts` — `formatJalali`، `toPersianDigits`
- `src/app/layout.tsx` — `dir="rtl"` + فونت Vazirmatn
- `src/app/dashboard/_components/IranMacroWidget.tsx` — ۶ شاخص کلان اقتصادی
- `src/app/dashboard/_components/CurrencyComparisonCard.tsx` — مقایسه دارایی‌ها
- `src/app/backtesting/_components/IranAssetBacktest.tsx` — سه استراتژی بک‌تست
- `src/app/backtesting/page.tsx`

**Tests:**
- `MyProjects/Octopus/Modules/tests/test_assets_api.py` — ۱۵ تست endpoint
- `MyProjects/Octopus/Modules/tests/test_asset_service.py` — تست cache و TGJU fetch

**Wiki/مستندات:**
- `.noqte/wiki/backlog.md` — همه تسک‌ها Done
- `.noqte/wiki/entities/assets-feature.md`
- `.noqte/wiki/pending-issues.md` — issue های پیشنهادی برای GitHub UI
- `MyProjects/Octopus/README.md` — به‌روزشده با Iranian Market Features

---

### تصمیمات معماری/طراحی
- **Submodule مشکل:** `MyProjects/Octopus/Modules` به‌صورت gitlink (submodule) بود؛ با `git rm --cached` حذف و به directory معمولی تبدیل شد تا قابل track باشد.
- **منبع داده:** `tgju.org` برای قیمت‌های لحظه‌ای بازار ایران.
- **Cache:** Redis با TTL 60 ثانیه برای داده‌های قیمتی.
- **DB:** PostgreSQL/TimescaleDB با hypertable برای داده‌های سری زمانی دارایی‌ها.
- **gh CLI:** در محیط اجرا نصب نیست؛ issue ها به صورت فایل local ذخیره شدند.
- **RTL/فارسی:** `dir="rtl"` در layout سراسری، فونت Vazirmatn.
- **واحد پولی:** Context سراسری CurrencyContext با قابلیت سوئیچ IRT↔USD.

---

### کارهای انجام‌شده (به ترتیب زمانی)

1. **Bootstrap ویکی پروژه** — `overview.md`، 4 entity، 2 concept، `log.md`، `index.md`
2. **ایجاد بک‌لاگ** — 6 تسک با اولویت‌بندی
3. **TASK-001** — سکشن دارایی‌های ایرانی: backend کامل + frontend کامل (001a تا 001d)
4. **001e/001f/001g** — widget داشبورد، portfolio tracker، ثبت router در main_refactored.py
5. **Commit اول:** `11b642d` — `feat: adding persian market assets`
6. **TASK-002** — IranMacroWidget + CurrencyComparisonCard + به‌روز صفحه داشبورد
7. **TASK-003** — locale.ts + layout.tsx با RTL/Jalali
8. **TASK-004** — CurrencyContext + CurrencyToggle
9. **TASK-005** — IranAssetBacktest (3 استراتژی: Buy & Hold، DCA، قدرت نسبی)
10. **TASK-006** — 2 فایل test + README آپدیت
11. **Commit دوم:** `642b3ea` — `feat: complete iranian market platform (TASK-002~006)` → push شد

---

### کارهای باقی‌مانده / گام بعدی
**هیچ تسک بازی در بک‌لاگ وجود ندارد.** تسک‌های پیشنهادی برای آینده (در `pending-issues.md`):
- TASK-002 تا 006 که همه Done شدند
- issue های اضافه‌ای که از GitHub UI باید ساخته شوند (چون gh CLI نصب نیست)

---

### نکات، گاتچاها، و درخواست‌های اخیر کاربر

- **آخرین درخواست کاربر (2026-06-28):** `"continue the tasks"` — اجرای خودکار همه تسک‌های فردا که قبلاً تعیین شده بودند.
- **gh CLI:** در محیط موجود نیست؛ اگر نیاز به ساخت issue در GitHub است، باید از UI یا یک محیط با gh نصب‌شده استفاده کرد.
- **submodule pitfall:** `MyProjects/Octopus/Modules` قبلاً submodule بود؛ اکنون به directory معمولی تبدیل شده — این وضعیت باید حفظ شود.
- **داده TGJU:** سرویس از `tgju.org` fetch می‌کند؛ در صورت failure، باید error handling مناسب بررسی شود.
- **تست‌ها:** هنوز `pytest` اجرا نشده تا تسک‌های نوشته‌شده validate شوند.
- **pending-issues.md** در `.noqte/wiki/` محتوی متن ۵ issue برای ایجاد دستی در GitHub UI قرار دارد.