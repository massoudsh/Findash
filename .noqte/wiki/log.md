# Log

> تاریخچه append-only — هر ورود با `## [YYYY-MM-DD] <type> | <خلاصه>` شروع می‌شود.

## [2026-07-08] sync | وضعیت backlog sync شد — TASK-007,009-024 ✅ Done (commits 7641e3e & 8e6bcc3) | فقط TASK-008 (فونت IRANYekanX) باقی است
## [2026-07-07] feature | مانوال فارسی کامل (help page 4 tab) + tab integration trading page + social/alerts headers
## [2026-07-07] localization | فارسی‌سازی کامل پلتفرم — نام/شخصیت ۱۱ عامل (M1-M11) به فارسی، tabs و panel‌های trading، source labels، mock data، timestamp‌ها و aria-labels همه ترجمه شدند
## [2026-07-06] feature | قابلیت افزودن دارایی: AddAssetModal (15 نماد ایرانی، auto-calc، localStorage) + IranPortfolioSection (donut chart SVG، holdings list، تاریخچه) — وصل به portfolio tab
## [2026-07-04] health-check | بررسی تناقضات ویکی — رفع 4 مورد: agents count (11→M1-M5 documented+M6-M11 planned)، Redis TTL (clarify 300s vs 60s)، TASK-001b/f (open→done)، font (overview اصلاح شد)
## [2026-07-01] redesign | داشبورد `/dashboard` با سبک modern mill-flat و رنگ `#3B82F6` بازطراحی شد — grid KPI cards، SVG performance chart، donut allocation، positions table، widgets و activity timeline
## [2026-07-01] audit | آدیت کامل Hawk View: 18 تسک جدید در backlog — Auth broken (demo hardcoded)، font missing، market/analytics tab placeholder، Iranian assets absent، CI/CD placeholder، no Jalali calendar
## [2026-06-27] bootstrap | ویکی پروژه اختاپوس از README bootstrap شد — overview، index، 4 entity، 2 concept ساخته شد
## [2026-06-27] backlog | بک‌لاگ پروژه ساخته شد — TASK-001 (دارایی‌های ایرانی) با اولویت بالا + 5 تسک دیگر
## [2026-06-27] feature | TASK-001 پیاده‌سازی شد — 11 فایل جدید: schemas، models، service، API router، migration SQL، 6 فایل frontend
## [2026-06-27] feature | TASK-001 کامل شد — 001e widget داشبورد، 001f portfolio section، 001g main_refactored.py ساخته شد
## [2026-06-28] feature | TASK-002~006 کامل شد — IranMacro، CurrencyToggle، Jalali، backtesting، 2 test file، README update
## [2026-06-29] feature | 6 قابلیت جدید — WebSocket hook، TradeTracker، PriceAlerts، News page، light-mode CSS، 31 unit test، build ✓ 57 pages
## [2026-06-29] update | تغییر پورت frontend از 3000 به 3003 (Dockerfile, docker-compose-core, docker-compose-complete, package.json, Makefile)
## [2026-06-30] update | فارسی‌سازی کل اپ: lang=fa dir=rtl، فونت Vazirmatn، default locale=fa، fa-utils.ts، ترجمه صفحات auth/dashboard/error و ۲۰+ کامپوننت
## [2026-06-30] feature | یکپارچه‌سازی زرین‌پال: backend create/callback/verify، payment_orders SQL، صفحات checkout/success/failed/callback و env sandbox
## [2026-06-30] redesign | بازطراحی کامل UI/UX فارسی: IRANYekanX، سبز #22C55E، persian-card/border/pattern، داشبورد 5 تب یکپارچه، RiskGauge SVG، CreditScore، homepage phone mockup، README فارسی کامل
