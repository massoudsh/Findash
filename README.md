# فین‌دَش | Findash

<div align="center">

**داشبورد هوشمند فین‌تک برای بازار ایران؛ تحلیل، پرتفوی، ریسک، پرداخت و هوش مصنوعی در یک تجربه فارسی و RTL**

![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791?style=flat&logo=postgresql&logoColor=white)
![RTL](https://img.shields.io/badge/RTL-Persian-22C55E?style=flat)

[شروع سریع](#شروع-سریع) · [قابلیت‌ها](#قابلیت‌ها) · [معماری](#معماری-سیستم) · [پرداخت](#پرداخت-زرین‌پال) · [امنیت](#امنیت-و-اعتماد)

</div>

---

## فین‌دَش چیست؟

فین‌دَش یک اپلیکیشن فین‌تک و داشبورد معاملاتی فارسی است که برای تجربه‌ی موبایل‌محور، تحلیل سریع بازار، مدیریت پرتفوی، پایش ریسک ریل‌تایم و پرداخت ایرانی طراحی شده است.

هدف پروژه این است که کاربر به‌جای جابه‌جایی بین چند ابزار پراکنده، همه چیز را در یک محیط تمیز و قابل فهم ببیند:

- وضعیت بازار
- ارزش پرتفوی
- معاملات و هشدارها
- مدیریت ریسک
- امتیاز اعتباری معاملاتی
- پرداخت و اشتراک
- گزارش‌ها و تحلیل‌های هوش مصنوعی

---

## قابلیت‌ها

### تجربه کاربری فارسی و موبایل‌محور

- رابط کاربری فارسی، راست‌به‌چپ و مناسب موبایل
- طراحی فین‌تک با رنگ اصلی `#22C55E`
- فرم‌های ورود و ثبت‌نام بازطراحی‌شده
- داشبورد ساده‌تر با تب‌های کمتر و قابل فهم‌تر
- فونت فارسی با fallback امن

### داشبورد معاملاتی

- نمای کلی پرتفوی و حساب‌ها
- نمودارهای عملکرد، جریان نقدی و تخصیص دارایی
- ticker زنده برای بازارهای مهم
- پرتفوی، معاملات، بازار و تحلیل در تب‌های جداگانه

### مدیریت ریسک ریل‌تایم

- گیج ریسک زنده
- سطح‌بندی ریسک: ایمن، متوسط، بالا، بحرانی
- نمایش VaR، افت حداکثری و بتای پرتفوی
- مناسب برای تصمیم‌گیری سریع در شرایط پرنوسان

### امتیاز اعتباری معاملاتی

- امتیاز ۳۰۰ تا ۸۵۰ برای سنجش کیفیت رفتار معاملاتی
- فاکتورهای امتیازدهی:
  - نرخ موفقیت
  - فعالیت معاملاتی
  - مدیریت ریسک
  - تنوع پرتفوی

### پرداخت زرین‌پال

- ایجاد پرداخت
- redirect به درگاه
- callback
- verify اجباری بعد از بازگشت از درگاه
- ذخیره وضعیت تراکنش در جدول `payment_orders`
- صفحات موفق و ناموفق پرداخت

### احراز هویت کاربر

- ورود با ایمیل و رمز عبور
- ثبت‌نام فارسی
- مسیرهای auth آماده در frontend و backend
- محافظت از مسیر پرداخت برای کاربران واردشده

### هوش مصنوعی و تحلیل

- معماری مبتنی بر ۱۱ Agent برای جمع‌آوری داده، تحلیل، ریسک، استراتژی و گزارش
- پشتیبانی از مدل‌های ML و گزارش‌سازی
- طراحی‌شده برای توسعه تحلیل‌های پیشرفته مالی

---

## معماری سیستم

```mermaid
flowchart LR
    User[کاربر] --> Web[Frontend - Next.js]
    Web <-->|REST / WebSocket| API[Backend - FastAPI]
    API --> Auth[Auth]
    API --> Payment[ZarinPal Payment]
    API --> Agents[AI Agents]
    API --> DB[(PostgreSQL / TimescaleDB)]
    API --> Redis[(Redis)]
    Agents --> DB
    Agents --> Redis
```

### لایه‌ها

| لایه | تکنولوژی |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS, Shadcn UI |
| Backend | FastAPI, Python 3.10+ |
| Database | PostgreSQL, TimescaleDB |
| Cache / Queue | Redis, Celery |
| Realtime | WebSocket |
| Monitoring | Prometheus, Grafana |
| Payment | ZarinPal |
| AI/ML | PyTorch, scikit-learn, orchestrated agents |

---

## شروع سریع

### پیش‌نیازها

- Node.js 18+
- Python 3.10+
- PostgreSQL 14+
- Redis
- Docker و Docker Compose برای اجرای ساده‌تر

---

## اجرای پروژه با Docker

```bash
git clone https://github.com/massoudsh/Findash.git
cd Findash

# ساخت و اجرای سرویس‌های اصلی
docker compose -f docker-compose-core.yml up --build -d
```

بعد از اجرا:

| سرویس | آدرس |
|---|---|
| Frontend | http://localhost:3003 |
| Backend API | http://localhost:8011 |
| Swagger | http://localhost:8011/docs |
| Grafana | http://localhost:3001 |

برای دیدن لاگ‌ها:

```bash
docker compose -f docker-compose-core.yml logs -f
```

برای توقف:

```bash
docker compose -f docker-compose-core.yml down
```

---

## اجرای پروژه بدون Docker

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
python3 start.py --reload
```

### Frontend

```bash
cd frontend-nextjs
npm install
npm run dev
```

Frontend روی این آدرس اجرا می‌شود:

```text
http://localhost:3003
```

---

## تنظیمات محیطی مهم

یک فایل env برای توسعه لازم است. نمونه متغیرهای مهم:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379/0

SECRET_KEY=change-this-secret-key-min-32-chars
JWT_SECRET_KEY=change-this-jwt-secret-min-32-chars

NEXT_PUBLIC_API_URL=http://localhost:8011
APP_BASE_URL=http://localhost:3003

ZARINPAL_MERCHANT_ID=your-zarinpal-merchant-id
```

برای sandbox زرین‌پال می‌توانید از merchant آزمایشی استفاده کنید، ولی برای production حتماً merchant واقعی و تنظیمات امن جداگانه قرار دهید.

---

## پرداخت زرین‌پال

فین‌دَش چرخه کامل پرداخت را پیاده‌سازی می‌کند:

```text
Create Payment → Redirect to ZarinPal → Callback → Verify → Success / Failed
```

### مسیرهای backend

| مسیر | توضیح |
|---|---|
| `POST /api/payment/zarinpal/create` | ایجاد سفارش پرداخت |
| `GET /api/payment/zarinpal/callback` | بازگشت از درگاه و verify |
| `GET /api/payment/zarinpal/status/{id}` | وضعیت سفارش |
| `GET /api/payment/zarinpal/history` | تاریخچه پرداخت کاربر |

### مسیرهای frontend

| مسیر | توضیح |
|---|---|
| `/payment/checkout` | انتخاب پلن و شروع پرداخت |
| `/payment/callback/zarinpal` | bridge برگشت از زرین‌پال |
| `/payment/success` | پرداخت موفق |
| `/payment/failed` | پرداخت ناموفق |

### Migration جدول پرداخت

```bash
psql -d trading_db -f database/schemas/payment_orders.sql
```

یا داخل Docker:

```bash
docker exec -i octopus-db psql -U postgres -d trading_db < database/schemas/payment_orders.sql
```

---

## احراز هویت

صفحات آماده:

| مسیر | توضیح |
|---|---|
| `/auth/signin` | ورود کاربر |
| `/auth/signup` | ثبت‌نام کاربر |

نکته‌های مهم برای production:

- `SECRET_KEY` و `JWT_SECRET_KEY` باید قوی و محرمانه باشند.
- رمزها نباید در frontend ذخیره شوند.
- پرداخت فقط برای کاربر احراز هویت‌شده انجام شود.
- callback پرداخت نباید منبع اعتماد باشد؛ verify سمت سرور الزامی است.

---

## ساختار مهم پروژه

```text
frontend-nextjs/
  src/app/
    page.tsx                         صفحه اصلی فارسی
    dashboard/page.tsx                داشبورد اصلی
    auth/signin/page.tsx              ورود
    auth/signup/page.tsx              ثبت‌نام
    payment/checkout/page.tsx         پرداخت
    payment/success/page.tsx          نتیجه موفق
    payment/failed/page.tsx           نتیجه ناموفق

src/
  main_refactored.py                  ثبت routerهای backend
  api/endpoints/payment_zarinpal.py   منطق زرین‌پال
  core/config.py                      تنظیمات پروژه

database/
  schemas/payment_orders.sql          جدول پرداخت‌ها
```

---

## امنیت و اعتماد

برای یک پروژه فین‌تک، این موارد حیاتی هستند:

- verify اجباری پرداخت در backend
- ذخیره مبلغ در دیتابیس به ریال
- عدم ذخیره merchant id یا secret در frontend
- استفاده از HTTPS در production
- rate limit برای auth و payment
- لاگ‌گیری callback و verify برای audit
- جداسازی sandbox و production
- پشتیبان‌گیری از PostgreSQL
- مانیتورینگ خطاها و latency

---

## چک‌لیست production

قبل از انتشار عمومی:

- [ ] مقدارهای واقعی `SECRET_KEY` و `JWT_SECRET_KEY` تنظیم شود
- [ ] merchant واقعی زرین‌پال جایگزین sandbox شود
- [ ] دامنه اصلی در `APP_BASE_URL` تنظیم شود
- [ ] SSL/HTTPS فعال شود
- [ ] migration دیتابیس اجرا شود
- [ ] backup دیتابیس فعال شود
- [ ] لاگ پرداخت‌ها مانیتور شود
- [ ] CORS فقط برای دامنه‌های مجاز تنظیم شود
- [ ] تست پرداخت موفق، ناموفق و callback تکراری انجام شود
- [ ] تست موبایل، RTL و خوانایی فارسی انجام شود

---

## صفحات اصلی برای تست دستی

بعد از اجرا، این مسیرها را بررسی کنید:

```text
/
/auth/signin
/auth/signup
/dashboard
/payment/checkout
/payment/success
/payment/failed
```

---

## نکات توسعه UI/UX

اصول طراحی فعلی پروژه:

- mobile-first
- فارسی و RTL از ابتدا
- حداکثر ۵ آیتم در navigation موبایل
- تب‌های کم و قابل فهم
- کارت‌های داده با تراکم کنترل‌شده
- رنگ سبز برای primary action و وضعیت سالم
- رنگ قرمز/نارنجی برای ریسک و هشدار
- نمایش عددها با خوانایی بالا و tabular figures

---

## Roadmap پیشنهادی

- اتصال کامل پرداخت به پلن اشتراک
- صفحه مدیریت اشتراک کاربر
- سیستم KYC / احراز هویت مالی
- کیف پول ریالی
- گزارش PDF فارسی
- هشدارهای push و SMS
- Risk policy برای محدودیت معامله
- اتصال به داده‌های واقعی بازار ایران
- پنل admin برای تراکنش‌ها و کاربران

---

## مشارکت

برای توسعه:

1. یک branch جدید بسازید
2. تغییرات را کوچک و قابل review نگه دارید
3. قبل از commit، مسیرهای اصلی را دستی تست کنید
4. برای تغییرات پرداخت یا auth، سناریوهای خطا را هم تست کنید

---

## License

MIT — برای جزئیات فایل `LICENSE` را ببینید.
