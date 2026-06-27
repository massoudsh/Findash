# Assets Feature (TASK-001)

> سکشن دارایی‌های ایرانی — طلا، نقره، ارز، مسکن، کریپتو

## وضعیت
`✅ پیاده‌سازی اولیه — آماده اتصال به سرویس‌های موجود`

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

## مرحله بعدی (باقی‌مانده از TASK-001)
- [ ] TASK-001b: ثبت router در `main_refactored.py` (`app.include_router(assets_router)`)
- [ ] TASK-001f: اتصال دارایی‌ها به portfolio tracker موجود
