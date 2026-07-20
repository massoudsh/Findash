# Backlog — فین‌دَش Trading Platform

> آخرین آدیت: 2026-07-08 | آدیتور: نقطه Pro
> بازار هدف: ایران 🇮🇷 | افق تحویل: ۳ ماه

---

## خلاصه آدیت کامل (Hawk View)

| لایه | وضعیت | مشکل اصلی |
|------|--------|-----------|
| Auth | 🔴 Broken | NextAuth → hardcoded demo@demo.com — backend واقعی وصل نیست |
| Data | 🔴 Empty | هیچ source ایرانی وصل نیست (tgju/Nobitex/TSETMC) |
| Font | 🔴 Missing | IRANYekanX فایل ندارد — fallback به system font |
| Market tab | 🔴 Placeholder | `"داده‌های بازار به‌زودی"` — هیچ چیز نیست |
| Analytics tab | 🔴 Placeholder | `"تحلیل پیشرفته به‌زودی"` — هیچ چیز نیست |
| Assets config | 🔴 Wrong | فقط AAPL/TSLA/MSFT — هیچ دارایی ایرانی ندارد |
| Ticker bar | 🟠 Fake | مقادیر hardcoded — زنده نیست |
| Calendar | 🟠 Wrong | Gregorian — باید شمسی باشد |
| OTP Auth | 🟠 Missing | بازار ایران موبایل+OTP انتظار دارد |
| DB Migration | 🟠 Manual | payment_orders migration دستی است |
| CI/CD | 🟠 Placeholder | deploy steps خالی |
| PWA | 🟡 Missing | بدون manifest برای نصب موبایل |
| Secrets | 🟡 Unsafe | default secrets در docker-compose |

---

## 🔴 CRITICAL — بلاکر (App غیرقابل استفاده)

---

### TASK-007 — Auth: وصل کردن NextAuth به Backend واقعی

**وضعیت:** `✅ Done` — commit `8e6bcc3` | route.ts از `BACKEND_URL` و `fetch` به `/api/auth/login` استفاده می‌کند
**اندازه:** M
**نوع:** Critical Bug
**تیم:** Backend + Frontend

**مشکل:**
فایل `/frontend-nextjs/src/app/api/auth/[...nextauth]/route.ts` فقط یک مقایسه hardcoded دارد:
```ts
// TODO: Replace with real backend call
if (email === "demo@demo.com" && password === "password") { return user; }
return null;
```
هیچ‌کس به جز demo@demo.com نمی‌تواند وارد شود. Backend یک endpoint احراز هویت کامل دارد اما frontend آن را صدا نمی‌زند.

**مسیر backend:**
- `src/api/endpoints/professional_auth.py` → `POST /api/auth/login`
- `src/api/endpoints/professional_auth.py` → `POST /api/auth/register`

**مراحل پیاده‌سازی:**
1. **Frontend (`route.ts`):** در `authorize()` ، یک `fetch` به `${BACKEND_URL}/api/auth/login` با `{email, password}` ارسال کن
2. JWT token بازگشتی را در NextAuth session ذخیره کن (با `callbacks.jwt` و `callbacks.session`)
3. Token را در `session.accessToken` قرار بده تا سایر API calls بتوانند از آن استفاده کنند
4. صفحه signup را به `POST /api/auth/register` وصل کن
5. خطاهای `401` و `422` را به پیام فارسی ترجمه کن
6. **تست:** ثبت‌نام، ورود، خروج، صفحه‌های protected

**فایل‌های تغییر:**
- `frontend-nextjs/src/app/api/auth/[...nextauth]/route.ts`
- `frontend-nextjs/src/app/auth/signup/page.tsx`
- `frontend-nextjs/src/middleware.ts` (اگر وجود دارد، protected routes)

**معیار پذیرش:**
- [ ] کاربر جدید ثبت‌نام می‌کند
- [ ] کاربر با email/password وارد می‌شود
- [ ] session در مرورگر ماندگار است
- [ ] صفحات protected (داشبورد) بدون auth به login redirect می‌کنند

---

### TASK-008 — Font: دانلود و آپلود IRANYekanX

**وضعیت:** `✅ Done` — commit `335019e` | رفع شد به روش جایگزین (نه دانلود IRANYekanX، بلکه رفع نام‌گذاری گمراه‌کننده)
**اندازه:** S
**نوع:** Critical Asset
**تیم:** DevOps / Frontend

**مشکل اصلی (تاریخی):**
پوشه `frontend-nextjs/public/fonts/` فقط یک `README.md` داشت. فایل‌های `IRANYekanX.woff2`/`.woff` هرگز دانلود نشدند.

**آنچه واقعاً اتفاق افتاد (راستی‌آزمایی این جلسه):**
در یک migration قبلی (خارج از این backlog)، فونت **Dana** (۱۲ وزن کامل، شامل italic) در `public/fonts/` اضافه شد و `@font-face` هایی در `globals.css` (خط ۷–۹۰) با فایل‌های واقعی `dana-*.woff2` تعریف شدند — اما `font-family` آن‌ها اشتباهاً هنوز `'IRANYekanX'` نام‌گذاری شده بود (یک نام مستعار گمراه‌کننده، نه فایل گمشده). یعنی فونت واقعاً لود می‌شد، صرفاً اسمش گمراه‌کننده بود.

**اصلاح انجام‌شده:**
1. تمام ۱۷ ارجاع `font-family: 'IRANYekanX'` در `globals.css` (خط‌های ۸ تا ۸۵ و usageهای ۱۹۴، ۲۰۸، ۶۱۰، ۶۹۷، ۷۲۲) به `'Dana'` تغییر نام یافت — بدون تغییر فایل‌های src (که از قبل درست و موجود بودند)
2. `tailwind.config.ts` (کلیدهای `iran-yekan` و `dana`) هم‌راستا اصلاح شد
3. `src/components/dashboard/risk-gauge.tsx` (SVG inline fontFamily) هم اصلاح شد
4. ترتیب fallback حفظ شد: `Dana → var(--font-vazir) → Vazirmatn → Tahoma → system` — یعنی Dana (فونت رسمی پروژه طبق تصمیم قبلی) همچنان اولویت اول است و به‌درستی لود می‌شود، نه fallback به Vazirmatn

**نکته برای تیم:** اگر تصمیم محصول عوض شده و `Vazirmatn` باید فونت اصلی شود (نه Dana)، این یک تصمیم طراحی جداست و نیاز به تأیید صریح دارد — در این اصلاح فقط باگ نام‌گذاری رفع شد، فونت رندرشده تغییر نکرد.

**معیار پذیرش:**
- [x] هیچ ارجاع `IRANYekanX` در کدبیس frontend باقی نمانده
- [x] فایل‌های src در @font-face همگی در `public/fonts/` موجودند (تأیید شد)
- [ ] تأیید بصری در مرورگر (نیازمند build واقعی روی سرور — طبق محدودیت build.md در کانتینر انجام نشد)

---

### TASK-009 — Market Tab: پیاده‌سازی تب بازار با داده زنده

**وضعیت:** `✅ Done` — commit `8e6bcc3` | کامپوننت `IranMarketOverview` در dashboard/page.tsx فعال است
**اندازه:** XL
**نوع:** Feature — Critical Gap
**تیم:** Backend + Frontend

**مشکل:**
تب «بازار» در داشبورد فقط یک `<div>داده‌های بازار به‌زودی</div>` است. برای یک پلتفرم فین‌تک ایرانی این تب باید قلب اپلیکیشن باشد.

**محتوای مورد نیاز:**

**بخش ۱ — شاخص‌های کلیدی (real-time):**
| دارایی | نماد | منبع |
|--------|------|------|
| شاخص کل بورس | TEDPIX | tgju.org یا TSETMC |
| شاخص هم‌وزن | TEDPIX-EW | TSETMC |
| دلار آزاد | USD-IRR | tgju.org |
| یورو | EUR-IRR | tgju.org |
| طلا ۱۸ عیار | GOLD18 | tgju.org |
| سکه تمام | COIN | tgju.org |
| بیت‌کوین تومانی | BTC-IRT | Nobitex API |
| اتریوم تومانی | ETH-IRT | Nobitex API |
| تتر تومانی | USDT-IRT | Nobitex API |

**بخش ۲ — جدول سهام برتر بورس:**
- نماد، نام، قیمت، تغییر، حجم، P/E
- فیلتر: بیشترین رشد / بیشترین افت / بیشترین حجم

**بخش ۳ — نمودار تاریخی:**
- Recharts LineChart با داده روزانه
- سوئیچ بازه: ۱روز / ۱هفته / ۱ماه / ۳ماه / ۱سال

**Backend مورد نیاز:**
```
GET /api/iran-market/overview     → شاخص‌های کلی
GET /api/iran-market/prices       → قیمت‌های لحظه‌ای
GET /api/iran-market/stocks/top   → سهام برتر
GET /api/iran-market/chart/{sym}  → تاریخچه قیمت
```

**منابع داده:**
- **tgju.org:** `https://api.tgju.org/v1/market/indicator/summary-table-data/price_dollar_rl` (بدون auth)
- **Nobitex:** `https://api.nobitex.ir/market/stats` (بدون auth)
- **TSETMC:** scraping یا `https://tsetmc.com/tsev2/data/TseClient2.aspx`

**مراحل:**
1. `src/api/endpoints/iran_market.py` بساز با endpoints بالا
2. Data fetcher برای tgju.org و Nobitex (15 دقیقه cache در Redis)
3. تب «بازار» در `dashboard/page.tsx` را با `<MarketOverview />` جایگزین کن
4. کامپوننت‌های: `<IranMarketOverview>`, `<StockTable>`, `<PriceChart>`

**معیار پذیرش:**
- [ ] قیمت دلار، طلا، سکه نمایش می‌یابند
- [ ] نمودار BTC-IRT رسم می‌شود
- [ ] جدول سهام فیلتر دارد
- [ ] داده هر ۳۰ ثانیه refresh می‌شود

---

### TASK-010 — Assets Config: دارایی‌های ایرانی در Backend

**وضعیت:** `✅ Done` — commit `7641e3e` | Iranian assets API پیاده‌سازی شده
**اندازه:** M
**نوع:** Data / Config
**تیم:** Backend

**مشکل:**
`src/core/assets_config.py` فقط AAPL، TSLA، MSFT و SPY دارد. هیچ دارایی ایرانی وجود ندارد. بدتر اینکه واحد پول همه USD است.

**مراحل:**
1. به `AssetType` اضافه کن: `COIN`, `REAL_ESTATE`, `CRYPTO_IRT`
2. به `Sector` اضافه کن: `GOLD`, `CURRENCY`, `IRAN_BOURSE`, `IRAN_CRYPTO`
3. دارایی‌های زیر را به `ASSETS` اضافه کن (واحد: IRT = تومان):

```python
"USD-IRR":   AssetMetadata(symbol="USD-IRR",   name="دلار آمریکا",      currency="IRR", asset_type=AssetType.FOREX, sector=Sector.CURRENCY, trading_hours="24/7"),
"EUR-IRR":   AssetMetadata(symbol="EUR-IRR",   name="یورو",              currency="IRR", ...),
"GOLD18-IRT":AssetMetadata(symbol="GOLD18-IRT",name="طلا ۱۸ عیار",      currency="IRT", asset_type=AssetType.COMMODITY, sector=Sector.GOLD, trading_hours="market_hours"),
"COIN-IRT":  AssetMetadata(symbol="COIN-IRT",  name="سکه بهار آزادی",   currency="IRT", asset_type=AssetType.COIN,  sector=Sector.GOLD),
"HALFCOIN":  AssetMetadata(symbol="HALFCOIN",  name="نیم‌سکه",           currency="IRT", ...),
"COIN-Q":    AssetMetadata(symbol="COIN-Q",    name="ربع‌سکه",           currency="IRT", ...),
"BTC-IRT":   AssetMetadata(symbol="BTC-IRT",   name="بیت‌کوین (تومانی)",currency="IRT", asset_type=AssetType.CRYPTOCURRENCY, trading_hours="24/7"),
"ETH-IRT":   AssetMetadata(symbol="ETH-IRT",   name="اتریوم (تومانی)", currency="IRT", ...),
"USDT-IRT":  AssetMetadata(symbol="USDT-IRT",  name="تتر (تومانی)",     currency="IRT", ...),
"TEDPIX":    AssetMetadata(symbol="TEDPIX",    name="شاخص کل بورس",     currency="IRR", asset_type=AssetType.ETF, sector=Sector.IRAN_BOURSE, trading_hours="market_hours"),
```

4. `IRANIAN_ASSETS` و `IRANIAN_CRYPTO_ASSETS` و `IRANIAN_COMMODITY_ASSETS` group list اضافه کن
5. تمام APIهایی که از `AssetsConfig.WATCHLIST_DEFAULT` استفاده می‌کنند را به‌روز کن

**معیار پذیرش:**
- [ ] `GET /api/assets` لیست دارایی‌های ایرانی برمی‌گرداند
- [ ] Portfolio tracker دارایی ایرانی می‌پذیرد
- [ ] هیچ endpoint ای break نمی‌شود

---

## 🟠 HIGH — برای تحویل به کاربر ایرانی

---

### TASK-011 — Ticker Bar: داده زنده به جای hardcoded

**وضعیت:** `✅ Done` — commit `7641e3e` | live ticker با flash animation پیاده‌سازی شده
**اندازه:** S
**نوع:** Feature
**تیم:** Frontend + Backend

**مشکل:**
TickerBar در `dashboard/page.tsx` مقادیر ثابت دارد:
```ts
const TICKERS = [
  { label: 'بیت‌کوین', value: '۶۷,۴۲۰', change: '+۱.۸٪', ... },
  // همه hardcoded
]
```

**مراحل:**
1. hook جدید: `use-iran-ticker.ts` با `setInterval` هر ۳۰ ثانیه
2. `GET /api/iran-market/overview` صدا بزن (از TASK-009)
3. نتیجه را format کن: عدد فارسی، علامت تغییر، رنگ
4. Skeleton loading هنگام fetch اول
5. در صورت خطا (network offline) مقادیر cache قبلی را نگه دار

**معیار پذیرش:**
- [ ] ۵ مقدار اول واقعی و زنده هستند
- [ ] هنگام تغییر قیمت، مقدار با animation update می‌شود

---

### TASK-012 — Auth: احراز هویت با شماره موبایل + OTP

**وضعیت:** `✅ Done` — commit `8e6bcc3` | send-otp / verify-otp + صفحات /auth/phone و /auth/otp
**اندازه:** L
**نوع:** Feature
**تیم:** Backend + Frontend

**چرا مهم است:**
کاربر ایرانی انتظار دارد با شماره موبایل و کد SMS وارد شود. ایمیل و رمز عبور برای ایران بسیار کم‌استفاده است.

**Backend مورد نیاز:**
```
POST /api/auth/send-otp    body: {phone: "09121234567"}
POST /api/auth/verify-otp  body: {phone, otp, device_id}
```

**سرویس SMS:**
- **KaveNegar:** محبوب‌ترین، مستندات خوب
- **Farapayamak:** جایگزین
- متغیر env: `SMS_PROVIDER`, `SMS_API_KEY`, `SMS_SENDER_LINE`

**مراحل Backend:**
1. OTP model در DB: `phone`, `code`, `expires_at`, `used`
2. Rate limit: ۳ بار در ۱۰ دقیقه برای هر شماره
3. OTP 6 رقمی، عمر ۵ دقیقه
4. بعد از verify: JWT صادر کن
5. SMS provider adapter (interface برای تعویض آسان)

**مراحل Frontend:**
1. صفحه `/auth/phone` — ورود شماره موبایل
2. صفحه `/auth/otp` — ورود کد ۶ رقمی (input با auto-advance)
3. تایمر countdown برای ارسال مجدد
4. NextAuth provider جدید برای phone auth

**معیار پذیرش:**
- [ ] ارسال SMS در کمتر از ۵ ثانیه
- [ ] OTP منقضی‌شده رد می‌شود
- [ ] بعد از ۳ بار تلاش اشتباه، شماره قفل می‌شود (۱۵ دقیقه)

---

### TASK-013 — Calendar: تاریخ شمسی در سراسر اپ

**وضعیت:** `✅ Done` — commit `8e6bcc3` | jalali.ts با formatJalali و formatRelative
**اندازه:** M
**نوع:** Localization
**تیم:** Frontend

**مشکل:**
تمام تاریخ‌ها به فرمت Gregorian (مثلاً `2026-07-01`) نمایش می‌یابند. کاربر ایرانی تاریخ شمسی انتظار دارد (۱۴۰۵/۰۴/۱۰).

**کتابخانه پیشنهادی:** `date-fns-jalali` یا `jalaali-js`

**مراحل:**
1. نصب: `npm install date-fns-jalali` در frontend
2. utility function: `toJalali(date: Date): string` → `"۱۴۰۵/۰۴/۱۰"`
3. `toJalaliRelative(date: Date): string` → `"۳ روز پیش"`
4. تمام مکان‌هایی که `new Date().toLocaleDateString()` یا `format()` دارند را جایگزین کن
5. DatePicker component با calendar شمسی (کتابخانه: `react-persian-datepicker` یا ساده)

**جاهایی که باید عوض شود:**
- تاریخ معاملات در TradeTracker
- تاریخ در charts (محور X)
- تاریخ پرداخت زرین‌پال
- notification timestamps

**معیار پذیرش:**
- [ ] همه تاریخ‌های نمایشی شمسی هستند
- [ ] DatePicker با calendar فارسی کار می‌کند

---

### TASK-014 — Analytics Tab: صفحه تحلیل پیشرفته

**وضعیت:** `✅ Done` — commit `8e6bcc3` | analytics-overview.tsx پیاده‌سازی شده
**اندازه:** L
**نوع:** Feature
**تیم:** Frontend + Backend

**مشکل:**
تب «تحلیل» placeholder است. باید شامل:

**محتوای مورد نیاز:**

1. **تحلیل تکنیکال:**
   - نمودار OHLC (candlestick) با Recharts یا lightweight-charts
   - اندیکاتورها: RSI، MACD، Moving Average
   - بازه: ۱ساعته، روزانه، هفتگی

2. **هیتمپ همبستگی:**
   - Correlation matrix بین دارایی‌های انتخابی
   - رنگ‌بندی: قرمز (منفی) تا سبز (مثبت)

3. **اخبار و سنتیمنت:**
   - اخبار مرتبط با دارایی انتخابی (از `/api/news`)
   - نمره سنتیمنت (مثبت/منفی/خنثی)

4. **گزارش پرتفوی:**
   - بهترین و بدترین دارایی‌های ماه
   - مقایسه با شاخص کل

**مراحل:**
1. کامپوننت `<TechnicalChart symbol={...} />` با candlestick
2. کامپوننت `<CorrelationHeatmap assets={[...]} />`
3. کامپوننت `<NewsAndSentiment symbol={...} />`
4. جایگزینی `AnalyticsPlaceholder` در `dashboard/page.tsx`

**معیار پذیرش:**
- [ ] نمودار کندل برای BTC-IRT نمایش می‌یابد
- [ ] RSI و MACD روی نمودار قابل فعال‌سازی هستند
- [ ] هیتمپ همبستگی ۵ دارایی را نشان می‌دهد

---

### TASK-015 — DB Migration: خودکارسازی migration در Docker

**وضعیت:** `✅ Done` — commit `8e6bcc3` | init-db.sql شامل payment_orders است
**اندازه:** S
**نوع:** DevOps
**تیم:** DevOps / Backend

**مشکل:**
`database/schemas/payment_orders.sql` باید دستی اجرا شود. در محیط جدید (مثلاً staging یا prod)، اگر این فایل اجرا نشود، زرین‌پال کار نمی‌کند.

**مراحل:**
1. فایل `payment_orders.sql` را به `scripts/init-db.sql` اضافه کن (یا symlink کن)
2. جایگزین: در `docker-compose-core.yml` یک service `db-migrate` تعریف کن:
```yaml
db-migrate:
  image: postgres:15
  command: >
    sh -c "psql $DATABASE_URL -f /migrations/payment_orders.sql &&
           psql $DATABASE_URL -f /migrations/01_initial_schema.sql"
  volumes:
    - ./database/schemas:/migrations:ro
  depends_on:
    db: { condition: service_healthy }
```
3. یا: Alembic را فعال کن و همه schemas را به migration تبدیل کن
4. در CI/CD: `pytest` قبل از اجرا migration اجرا کند

**معیار پذیرش:**
- [ ] `docker compose up` جدید بدون دستور manual همه جداول را می‌سازد
- [ ] migration idempotent است (دوبار اجرا مشکل نسازد — `CREATE TABLE IF NOT EXISTS`)

---

## 🟡 MEDIUM — برای تکمیل تجربه کاربری

---

### TASK-016 — PWA: قابلیت نصب اپ موبایل

**وضعیت:** `✅ Done` — commit `8e6bcc3` | manifest.json کامل است
**اندازه:** S
**نوع:** Enhancement
**تیم:** Frontend

**چرا مهم است:**
کاربر ایرانی موبایل محور است. PWA باعث می‌شود کاربر اپ را مثل native نصب کند بدون نیاز به App Store.

**مراحل:**
1. `frontend-nextjs/public/manifest.json` بساز:
```json
{
  "name": "فین‌دَش",
  "short_name": "فین‌دَش",
  "description": "داشبورد معاملاتی هوشمند",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#030712",
  "theme_color": "#22C55E",
  "dir": "rtl",
  "lang": "fa",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```
2. آیکون‌های ۱۹۲x192 و 512x512 طراحی کن (سبز با لوگو فین‌دَش)
3. `next.config.js` → `headers` برای manifest
4. Service Worker: `next-pwa` package نصب کن برای offline support
5. در `layout.tsx`: `<link rel="manifest" href="/manifest.json">`

**معیار پذیرش:**
- [ ] Chrome نشان «نصب» می‌دهد
- [ ] بعد از نصب، اپ بدون مرورگر باز می‌شود
- [ ] صفحه اصلی بدون اینترنت (offline) نمایش می‌یابد

---

### TASK-017 — Persian Digits: اعداد فارسی در همه جا

**وضعیت:** `✅ Done` — commit `8e6bcc3` | کامپوننت `<PNum>` ساخته شده
**اندازه:** S
**نوع:** Localization
**تیم:** Frontend

**مشکل:**
`toPersianDigits()` در `locale.ts` وجود دارد اما در بسیاری از جاهای برنامه هنوز اعداد لاتین نمایش داده می‌شوند.

**مراحل:**
1. `toFarsiNumber(n: number | string): string` utility را مرکزی کن
2. جست‌وجوی `tabular-nums` در کامپوننت‌ها — همه را با فارسی جایگزین کن
3. کامپوننت `<PNum>{value}</PNum>` بساز که خود تبدیل را انجام دهد
4. در تمام Recharts tooltip و axis، formatter فارسی اضافه کن
5. قیمت‌ها در TradeTracker، Portfolio، PaymentCheckout

**استثنا:** مقادیر `dir="ltr"` (symbol ها، IBAN، شماره کارت) لاتین بمانند.

**معیار پذیرش:**
- [ ] هیچ عدد لاتین در متن فارسی دیده نمی‌شود
- [ ] قیمت‌ها، درصدها، تاریخ‌ها همه فارسی هستند

---

### TASK-018 — Dark/Light Mode: تغییر تم

**وضعیت:** `✅ Done` — commit `8e6bcc3` | next-themes + toggle در navigation
**اندازه:** S
**نوع:** Enhancement
**تیم:** Frontend

**مراحل:**
1. `next-themes` نصب کن: `npm install next-themes`
2. در `layout.tsx` → `<ThemeProvider attribute="class" defaultTheme="dark">`
3. دکمه toggle در navigation sidebar: `<MoonIcon>` / `<SunIcon>`
4. CSS variables برای light mode را در `globals.css` کامل کن (primary سبز در هر دو تم)
5. `localStorage` ترجیح کاربر را ذخیره کند

**معیار پذیرش:**
- [ ] دکمه toggle در sidebar کار می‌کند
- [ ] تم در reload مرورگر حفظ می‌شود
- [ ] هیچ text unreadable در light mode نیست

---

### TASK-019 — Portfolio: ثبت دارایی ایرانی و P&L تومانی

**وضعیت:** `✅ Done` — commit `8e6bcc3` | سوئیچ تومان/دلار/ریال در portfolio-content.tsx
**اندازه:** M
**نوع:** Feature
**تیم:** Frontend + Backend

**مشکل:**
TradeTracker فعلی دارایی‌های ایرانی ندارد. P&L به دلار حساب می‌شود نه تومان.

**مراحل:**
1. Dropdown دارایی در TradeTracker را به لیست ایرانی از TASK-010 وصل کن
2. سوئیچ واحد: تومان / دلار / ریال
3. P&L هم به تومان و هم درصد نمایش دهد
4. در sidebar داشبورد: «بهترین دارایی» و «بدترین دارایی» این ماه
5. Export به Excel/CSV با اعداد فارسی

**معیار پذیرش:**
- [ ] می‌توانم طلا ۱۸ عیار به تومان به پرتفوی اضافه کنم
- [ ] P&L به تومان نمایش می‌یابد
- [ ] Export CSV کار می‌کند

---

## 🔵 DEVOPS — برای تحویل به DevOps

---

### TASK-020 — CI/CD: دیپلوی واقعی (نه placeholder)

**وضعیت:** `✅ Done` — commit `8e6bcc3` | CI/CD hardening انجام شده
**اندازه:** L
**نوع:** DevOps
**تیم:** DevOps

**مشکل:**
در `.github/workflows/ci-cd.yml`، هر دو مرحله `deploy-dev` و `deploy-prod` فقط `echo` هستند:
```yaml
- name: Deploy placeholder
  run: echo "Configure AWS/ECS or your dev target..."
```

**گزینه‌های پیشنهادی:**

**Option A — VPS ساده (مناسب شروع):**
```yaml
- name: SSH Deploy
  uses: appleboy/ssh-action@v1
  with:
    host: ${{ secrets.SERVER_HOST }}
    username: ${{ secrets.SERVER_USER }}
    key: ${{ secrets.SERVER_SSH_KEY }}
    script: |
      cd /opt/findash
      git pull origin main
      docker compose -f docker-compose-core.yml pull
      docker compose -f docker-compose-core.yml up -d --no-build
```

**Option B — Docker Hub + VPS:**
1. CI → build image → push به Docker Hub (یا GHCR)
2. Server → pull image → restart container

**مراحل:**
1. Secrets در GitHub repo تعریف کن: `SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`
2. `.env.production` template بساز با همه متغیرهای لازم
3. `Makefile` یا `deploy.sh` بساز برای دستورات deploy
4. Health check بعد از deploy: `curl https://domain/health`
5. Rollback strategy: در صورت failure، به version قبل برگرد

**فایل‌های مورد نیاز:**
- `.github/workflows/ci-cd.yml` → مراحل deploy واقعی
- `scripts/deploy.sh` → deploy script برای اجرا روی سرور
- `.env.production.example` → template با توضیحات

**معیار پذیرش:**
- [ ] push به `dev` → deploy خودکار به staging
- [ ] tag `v*` → deploy خودکار به production
- [ ] اگر health check fail شود، pipeline fail می‌شود

---

### TASK-021 — Production Secrets: مدیریت امن متغیرهای محیطی

**وضعیت:** `✅ Done` — commit `8e6bcc3` | DevOps hardening انجام شده
**اندازه:** M
**نوع:** Security / DevOps
**تیم:** DevOps + Backend

**مشکل‌های فعلی:**
```yaml
# docker-compose-core.yml
- SECRET_KEY=docker-dev-secret-key-32chars-minimum    # ⚠️ hardcoded
- JWT_SECRET_KEY=docker-dev-jwt-secret-key-32chars-min # ⚠️ hardcoded
- ZARINPAL_MERCHANT_ID=1344b5d4-0048-11e8-94db-005056a205be # sandbox ID
```

**مراحل:**
1. همه secrets را از docker-compose حذف کن — فقط از env_file بخوان:
```yaml
env_file:
  - .env.production  # یا .env.local در dev
```
2. `.env.example` جامع بساز با توضیح هر متغیر
3. **Secret rotation:** SECRET_KEY و JWT_SECRET_KEY باید ۶۴+ کاراکتر تصادفی باشند
4. برای production: Docker Secrets یا HashiCorp Vault یا GitHub Secrets
5. ZARINPAL_MERCHANT_ID در production باید ID واقعی باشد (نه sandbox)
6. **CORS:** در production، `allowed_origins` را به دامنه واقعی محدود کن

**متغیرهای ضروری production:**
```bash
SECRET_KEY=<64_char_random>
JWT_SECRET_KEY=<64_char_random>
DATABASE_URL=postgresql://user:pass@host:5432/findash_prod
REDIS_URL=redis://:password@host:6379/0
ZARINPAL_MERCHANT_ID=<real_merchant_id>
APP_BASE_URL=https://yourdomain.ir
NEXTAUTH_SECRET=<64_char_random>
NEXTAUTH_URL=https://yourdomain.ir
SMS_API_KEY=<kaveh_negar_key>
```

**معیار پذیرش:**
- [ ] هیچ secret در `docker-compose*.yml` hardcoded نیست
- [ ] `.env.example` همه متغیرها را با توضیح دارد
- [ ] در production، secrets از vault/CI loaded می‌شوند

---

### TASK-022 — Nginx: Reverse Proxy و SSL

**وضعیت:** `✅ Done` — commit `8e6bcc3` | DevOps hardening انجام شده
**اندازه:** M
**نوع:** DevOps / Infrastructure
**تیم:** DevOps

**مشکل:**
Frontend روی port `:3003` و Backend روی `:8011` expose هستند. برای production باید Nginx جلوی هر دو باشد با SSL.

**مراحل:**
1. `docker/nginx.conf` بساز:
```nginx
server {
    listen 80;
    server_name yourdomain.ir www.yourdomain.ir;
    return 301 https://$host$request_uri;
}
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://frontend:3003;
    }
    location /api/ {
        proxy_pass http://api:8000;
        proxy_set_header X-Real-IP $remote_addr;
    }
    location /ws {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```
2. Nginx service به `docker-compose-core.yml` اضافه کن
3. **SSL:** Certbot با Let's Encrypt (یا ایرانی: `pki.ir`)
4. Rate limiting در Nginx: `limit_req_zone` برای حفاظت از `/api/auth`
5. gzip compression برای static assets

**معیار پذیرش:**
- [ ] `https://yourdomain.ir` کار می‌کند
- [ ] HTTP redirect به HTTPS
- [ ] WebSocket از طریق Nginx کار می‌کند

---

### TASK-023 — Monitoring: Alert های Production

**وضعیت:** `✅ Done` — commit `8e6bcc3` | DevOps hardening انجام شده
**اندازه:** S
**نوع:** DevOps / Monitoring
**تیم:** DevOps

**مشکل:**
Prometheus و Grafana setup شده‌اند اما هیچ alert تعریف نشده. اگر سرویس down شود، کسی خبر نمی‌شود.

**مراحل:**
1. `monitoring/alertmanager.yml` را کامل کن — Telegram یا Email receiver:
```yaml
receivers:
  - name: 'telegram'
    telegram_configs:
      - bot_token: '${TELEGRAM_BOT_TOKEN}'
        chat_id: '${TELEGRAM_CHAT_ID}'
        message: '🚨 {{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'
```
2. Alert rules در `monitoring/prometheus.yml`:
   - سرویس API down بیش از ۱ دقیقه → alert
   - Database connection error → alert
   - Payment failure rate > ۵٪ → alert
   - Response time > ۲ ثانیه P95 → alert
3. Grafana dashboard: `monitoring/grafana/dashboards/findash.json`
4. Uptime check: `monitoring/prometheus.yml` → blackbox exporter

**معیار پذیرش:**
- [ ] اگر API down شود، ۱ دقیقه بعد Telegram پیام می‌آید
- [ ] Grafana داشبورد request rate، latency، error rate را نشان می‌دهد

---

### TASK-024 — Backup: پشتیبان‌گیری خودکار Database

**وضعیت:** `✅ Done` — commit `8e6bcc3` | DevOps hardening انجام شده
**اندازه:** S
**نوع:** DevOps
**تیم:** DevOps

**مراحل:**
1. Cron job روی سرور — هر شب ساعت ۳ بامداد:
```bash
#!/bin/bash
# scripts/backup-db.sh
BACKUP_FILE="backup-$(date +%Y%m%d-%H%M).sql.gz"
docker exec octopus-db pg_dump -U postgres trading_db | gzip > /opt/backups/$BACKUP_FILE
# نگه داشتن ۳۰ روز آخر
find /opt/backups -name "backup-*.sql.gz" -mtime +30 -delete
```
2. Upload backup به S3 یا FTP ایرانی (آروان‌کلاد)
3. Backup service در docker-compose (image: `prodrigestivill/postgres-backup-local`)
4. Test restore: ماهانه، backup را در محیط test restore کن

**معیار پذیرش:**
- [ ] هر شب یک فایل backup جدید ساخته می‌شود
- [ ] backup به storage خارجی upload می‌شود
- [ ] restore از backup آزمایش شده است

---

## وضعیت کلی Backlog

| تسک | اندازه | اولویت | وضعیت | تیم |
|-----|--------|--------|--------|-----|
| TASK-007 Auth → Backend | M | 🔴 Critical | ✅ Done (`8e6bcc3`) | Full-stack |
| TASK-008 Font IRANYekanX | S | 🔴 Critical | ✅ Done (`335019e`) | Frontend |
| TASK-009 Market Tab | XL | 🔴 Critical | ✅ Done (`8e6bcc3`) | Full-stack |
| TASK-010 Iranian Assets Config | M | 🔴 Critical | ✅ Done (`7641e3e`) | Backend |
| TASK-011 Ticker Live Data | S | 🟠 High | ✅ Done (`7641e3e`) | Full-stack |
| TASK-012 SMS OTP Auth | L | 🟠 High | ✅ Done (`8e6bcc3`) | Full-stack |
| TASK-013 Jalali Calendar | M | 🟠 High | ✅ Done (`8e6bcc3`) | Frontend |
| TASK-014 Analytics Tab | L | 🟠 High | ✅ Done (`8e6bcc3`) | Full-stack |
| TASK-015 DB Migration Auto | S | 🟠 High | ✅ Done (`8e6bcc3`) | DevOps |
| TASK-016 PWA | S | 🟡 Medium | ✅ Done (`8e6bcc3`) | Frontend |
| TASK-017 Persian Digits | S | 🟡 Medium | ✅ Done (`8e6bcc3`) | Frontend |
| TASK-018 Dark/Light Mode | S | 🟡 Medium | ✅ Done (`8e6bcc3`) | Frontend |
| TASK-019 Portfolio Toman | M | 🟡 Medium | ✅ Done (`8e6bcc3`) | Full-stack |
| TASK-020 CI/CD Real Deploy | L | 🔵 DevOps | ✅ Done (`8e6bcc3`) | DevOps |
| TASK-021 Production Secrets | M | 🔵 DevOps | ✅ Done (`8e6bcc3`) | DevOps |
| TASK-022 Nginx + SSL | M | 🔵 DevOps | ✅ Done (`8e6bcc3`) | DevOps |
| TASK-023 Monitoring Alerts | S | 🔵 DevOps | ✅ Done (`8e6bcc3`) | DevOps |
| TASK-024 DB Backup | S | 🔵 DevOps | ✅ Done (`8e6bcc3`) | DevOps |

---

## نقشه راه پیشنهادی (۳ ماه)

### ماه اول — Make It Work
> هدف: اپ قابل استفاده واقعی برای کاربر ایرانی

- TASK-007 (Auth واقعی)
- TASK-008 (Font)
- TASK-010 (Assets ایرانی)
- TASK-009 (Market Tab + داده زنده)
- TASK-015 (Migration خودکار)
- TASK-020 (CI/CD واقعی)
- TASK-021 (Secrets)

### ماه دوم — Make It Good
> هدف: تجربه کاربری کامل ایرانی

- TASK-011 (Ticker زنده)
- TASK-012 (SMS OTP)
- TASK-013 (Jalali)
- TASK-014 (Analytics)
- TASK-022 (Nginx + SSL)
- TASK-023 (Monitoring)

### ماه سوم — Make It Scale
> هدف: آماده برای رشد

- TASK-016 (PWA)
- TASK-017 (Persian Digits)
- TASK-018 (Dark/Light Mode)
- TASK-019 (Portfolio Toman)
- TASK-024 (Backup)
- تست بار، بهینه‌سازی cache، CDN

---

*آخرین آپدیت: 2026-07-01 — آدیت کامل توسط نقطه Pro*
