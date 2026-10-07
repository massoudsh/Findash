# Assets Feature (TASK-001)

> سکشن دارایی‌های ایرانی — طلا، نقره، ارز، مسکن، کریپتو

## وضعیت
`✅ کامل — push شده در commit 642b3ea`

## فایل‌ها

| فایل | نوع | شرح |
|------|-----|-----|
| `Modules/src/schemas/asset_schema.py` | Backend | Pydantic schemas: AssetDetail, AssetListResponse, AssetHistory, PortfolioAsset |
| `Modules/src/models/asset.py` | Backend | SQLAlchemy models: Asset, AssetPriceSnapshot, AssetPriceHistory, UserPortfolioAsset + seed data |
| `Modules/src/services/asset_service.py` | Backend | TGJU fetcher، Redis cache (TTL: 60s)، DB upsert |
| `Modules/src/api/routes/assets.py` | Backend | FastAPI router — 5 endpoint |
| `Modules/src/migrations/add_asset_tables.sql` | DB | SQL migration + TimescaleDB hypertable + seed INSERT |
| `Modules/frontend-nextjs/src/lib/assets.ts` | Frontend | Types، formatToman، formatChange، API helpers |
| `Modules/frontend-nextjs/src/app/assets/page.tsx` | Frontend | صفحه اصلی — tabs + summary bar |
| `Modules/frontend-nextjs/src/app/assets/_components/AssetCard.tsx` | Frontend | کارت هر دارایی با sparkline |
| `Modules/frontend-nextjs/src/app/assets/_components/AssetPriceChart.tsx` | Frontend | SparklineChart + AssetFullChart (Recharts) |
| `Modules/frontend-nextjs/src/app/assets/_components/AssetGrid.tsx` | Frontend | Grid با auto-refresh هر 60s |
| `Modules/frontend-nextjs/src/app/assets/_components/AssetSummaryBar.tsx` | Frontend | نوار خلاصه: نرخ دلار، تعداد صعودی/نزولی، زمان آپدیت |

## دارایی‌های پشتیبانی‌شده (16 نماد)
| دسته | نمادها |
|------|--------|
| طلا | XAU18، XAU24، COIN_FULL، COIN_HALF، COIN_QUARTER، COIN_OLD، MESGHAL |
| نقره | XAG |
| ارز | USD، EUR، AED، GBP |
| مسکن | RE_TEHRAN |
| کریپتو | BTC، ETH، USDT |

## API Endpoints
| Method | Path | عملکرد |
|--------|------|--------|
| GET | `/api/assets` | همه دارایی‌ها (با فیلتر category) |
| GET | `/api/assets/usd-rate` | نرخ دلار به تومان |
| GET | `/api/assets/{symbol}` | جزئیات یک دارایی |
| GET | `/api/assets/{symbol}/history?days=30` | تاریخچه OHLCV |
| POST | `/api/assets/portfolio` | افزودن به پورتفولیو |

## معماری داده
```
TGJU API → AssetService._fetch_tgju()
→ Redis SETEX (TTL: 60s)     ← cache-first
→ AssetPriceSnapshot (upsert)
→ AssetPriceHistory (TimescaleDB hypertable)
```

## وابستگی‌ها
- [[entities/data-layer]] — Redis + TimescaleDB
- [[entities/backend]] — FastAPI router ثبت در main app
- [[entities/frontend]] — صفحه `/assets` در App Router

## قابلیت افزودن دارایی (Feature: Add Asset)

### کامپوننت‌های جدید
| فایل | عملکرد |
|------|--------|
| `frontend-nextjs/src/components/portfolio/add-asset-modal.tsx` | مودال ثبت دارایی — Select دارایی، کد، مقدار، قیمت واحد، ارزش کل، واحد پول، نوع تراکنش. auto-calculate قیمت↔ارزش. localStorage key: `iran_portfolio_v1` |
| `frontend-nextjs/src/components/portfolio/iran-portfolio-section.tsx` | سکشن «دارایی‌های من» — KPI cards، donut chart SVG، لیست holdings، تاریخچه تراکنش |

### دارایی‌های پیش‌فرض در مودال (15 نماد)
طلا ۱۸/۲۴ عیار، سکه تمام/نیم، نقره، دلار، یورو، درهم، BTC، ETH، USDT، سهام، اوراق، مسکن، نقدی

### ذخیره‌سازی
`localStorage['iran_portfolio_v1']` — آرایه‌ای از `IranPortfolioAsset` (id، type، name، code، quantity، unitPrice، totalValue، currency، side، timestamp)

### جایگاه در UI
`portfolio-content.tsx` → اول صفحه پورتفولیو (تب portfolio در dashboard)

## مرحله بعدی (باقی‌مانده از TASK-001)
- [x] TASK-001b: ثبت router در `main_refactored.py` (`app.include_router(assets_router)`) — انجام شد 2026-06-27
- [x] TASK-001f: اتصال دارایی‌ها به portfolio tracker موجود — trade-tracker.tsx ساخته شد 2026-06-29

## دستیار هوشمند تخصیص دارایی (AI Asset Allocation Copilot)

> فقط تحلیل ترکیب دارایی موجود کاربر — **بدون هیچ توصیه خرید/فروش یا سیگنال قیمتی** (تمایز استراتژیک محصول از کانال‌های سیگنال‌ده)

### وضعیت
`✅ کامل`

### فایل‌ها
| فایل | نوع | شرح |
|------|-----|-----|
| `src/api/endpoints/allocation_copilot.py` | Backend | Router `/api/copilot` — محاسبه HHI (Herfindahl-Hirschman Index) روی سهم هر دسته دارایی، سطح تمرکز (کم/متوسط/بالا)، امتیاز تنوع (0-100)، insights متنی rule-based فارسی + disclaimer اجباری |
| `frontend-nextjs/src/components/portfolio/allocation-copilot.tsx` | Frontend | کامپوننت `AllocationCopilot` — POST به backend با holdings، نمایش badge ریسک تمرکز/امتیاز تنوع/سهم بزرگ‌ترین دارایی + لیست insights + disclaimer |
| `tests/test_allocation_copilot.py` | Test | 3 تست: پرتفوی خالی، متمرکز (۹۰٪ یک دسته → بالا)، پخش‌شده در ۴ دسته مساوی (HHI=2500 → متوسط) |

### API
| Method | Path | ورودی | خروجی |
|--------|------|-------|-------|
| POST | `/api/copilot/allocation-analysis` | `{holdings: [{code,name,type,value}]}` | `total_value، category_breakdown، top_holding_pct، hhi، concentration_level، diversification_score، insights[]، disclaimer` |

### منطق HHI (thresholds روی مقیاس 0-10000)
`hhi < 1500` → کم | `hhi <= 2500` → متوسط | `hhi > 2500` → بالا. `diversification_score = 100 - hhi/100` (clamped 0-100).

### جایگاه در UI
داخل `IranPortfolioSection` (`iran-portfolio-section.tsx`)، بین KPI cards و بخش «تخصیص دارایی» donut — `holdings` از `positions` محاسبه‌شده (net buy-sell) پاس داده می‌شود.

### وابستگی‌ها
- [[entities/assets-feature]] — از همان `AssetType`/`CATEGORY_LABEL` در `add-asset-modal.tsx` استفاده می‌کند
- ثبت‌شده در `main_refactored.py` با tag `"Asset Allocation Copilot"`
