<div align="center">

<br/>

<img src="https://raw.githubusercontent.com/massoudsh/Findash/main/frontend-nextjs/public/logo.png" alt="Findash" width="72" height="72" />

# Findash

**داشبورد فین‌تک فول‌استک، ساخته‌شده برای بازار ایران.**
داده‌های لحظه‌ای بازار · رهگیری پرتفوی · مدیریت ریسک · رابط کاربری فارسی (راست‌به‌چپ) · درگاه پرداخت زرین‌پال

<br/>

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **دموی زنده · اسکرین‌شات · ویدیوی معرفی به‌زودی**
> *کلون کن ← `.env` را تنظیم کن ← `docker compose up` — همین!*

</div>

---

## Findash چیست؟

Findash یک **داشبورد فین‌تک ایرانی متن‌باز** است که هر چیزی که یک معامله‌گر یا سرمایه‌گذار نیاز دارد را در یک رابط کاربری تمیز و فارسی‌محور کنار هم می‌آورد:

- قیمت لحظه‌ای طلا، ارز، رمزارز و مسکن
- سود و زیان پرتفوی همراه با تاریخچه معاملات
- گیج ریسک لحظه‌ای (VaR، افت سرمایه، بتا)
- یکپارچگی کامل با درگاه زرین‌پال (ایجاد ← ریدایرکت ← تأیید)
- تحلیل هوشمند با ۱۱ ایجنت هوش مصنوعی هماهنگ‌شده
- پشتیبانی کامل راست‌به‌چپ با فونت دانا

---

## دمو

<div align="center">

<!-- پس از آماده شدن، با GIF/اسکرین‌شات واقعی جایگزین شود -->
```
┌─────────────────────────────────────────────────────┐
│  📊 داشبورد  💼 پرتفوی  📈 بازارها  ⚠️ هشدارها      │
│ ─────────────────────────────────────────────────── │
│  ارزش پرتفوی        ↑ 12.4%    سطح ریسک: متوسط      │
│  ₿ BTC  47,200 $     طلا  3,850,000 ت               │
│  دلار   58,200 ت     سکه  42,000,000 ت              │
│  ─────────────────────────────────────── ──────────  │
│  [نمودار] ████████████░░░  شارپ: 1.42               │
└─────────────────────────────────────────────────────┘
```

*ویدیوی متحرک کامل دمو به‌زودی اینجا اضافه می‌شود.*

</div>

---

## امکانات

| دسته | نکات برجسته |
|---|---|
| 📊 **داشبورد** | تیکر لحظه‌ای، نمای کلی پرتفوی، نمودار جریان نقدی، تخصیص دارایی |
| 💼 **پرتفوی** | رهگیر معاملات، سود و زیان، دارایی‌های فیزیکی ایرانی (طلا، نقره، مسکن، رمزارز) |
| ⚡ **لحظه‌ای** | فید بازار با WebSocket و اتصال مجدد خودکار |
| ⚠️ **موتور ریسک** | گیج ریسک لحظه‌ای، VaR، حداکثر افت سرمایه، بتای پرتفوی |
| 🧠 **ایجنت‌های هوش مصنوعی** | ارکستریتور ۱۱ ایجنته برای جمع‌آوری داده، تحلیل، استراتژی و گزارش |
| 💳 **پرداخت** | چرخه کامل زرین‌پال — ایجاد، ریدایرکت، بازگشت، تأیید، تاریخچه |
| 🔐 **احراز هویت** | ورود/ثبت‌نام مبتنی بر JWT با محافظت مسیر |
| 🌐 **فارسی‌محور** | چیدمان راست‌به‌چپ، تاریخ جلالی، تبدیل تومان/دلار، فونت دانا |
| 📱 **آماده موبایل** | طراحی موبایل‌محور، حداکثر ۵ آیتم ناوبری، تراکم کارت خوانا |

---

## معماری

```
کاربر
 │
 ├─► Next.js 15 (فرانت‌اند · پورت 3003)
 │       │ REST / WebSocket
 │       ▼
 └─► FastAPI (بک‌اند · پورت 8011)
         ├─► احراز هویت (JWT)
         ├─► پرداخت زرین‌پال
         ├─► ایجنت‌های هوش مصنوعی (×۱۱)
         ├─► PostgreSQL / TimescaleDB
         └─► کش Redis
```

| لایه | فناوری |
|---|---|
| فرانت‌اند | Next.js 15، TypeScript، Tailwind CSS، Shadcn UI، Recharts |
| بک‌اند | FastAPI، Python 3.10+، Celery |
| پایگاه داده | PostgreSQL 14+، TimescaleDB |
| کش / صف | Redis، Celery Workers |
| لحظه‌ای | WebSocket (هوک اختصاصی) |
| هوش مصنوعی / یادگیری ماشین | PyTorch، scikit-learn، ۱۱ ایجنت هماهنگ‌شده |
| پرداخت | زرین‌پال (سندباکس + محیط عملیاتی) |
| مانیتورینگ | Prometheus (9090)، Grafana (3001) |

---

## شروع سریع

### گزینه الف — داکر (پیشنهادی)

```bash
git clone https://github.com/massoudsh/Findash.git
cd Findash
cp .env.example .env          # مقادیر خودتان را وارد کنید
docker compose -f docker-compose-core.yml up --build -d
```

| سرویس | آدرس |
|---|---|
| فرانت‌اند | http://localhost:3003 |
| API بک‌اند | http://localhost:8011 |
| مستندات Swagger | http://localhost:8011/docs |
| Grafana | http://localhost:3001 |

```bash
# مشاهده لاگ‌ها
docker compose -f docker-compose-core.yml logs -f

# توقف
docker compose -f docker-compose-core.yml down
```

---

### گزینه ب — اجرای محلی (بدون داکر)

**بک‌اند**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements/requirements.txt
python3 start.py --reload
```

**فرانت‌اند**

```bash
cd frontend-nextjs
npm install
npm run dev        # http://localhost:3003
```

---

## متغیرهای محیطی

یک فایل `.env` در ریشه پروژه بسازید:

```env
# پایگاه داده
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_db

# کش
REDIS_URL=redis://localhost:6379/0

# امنیت (پیش از استفاده عملیاتی تغییر دهید!)
SECRET_KEY=change-this-secret-key-min-32-chars
JWT_SECRET_KEY=change-this-jwt-secret-min-32-chars

# API
NEXT_PUBLIC_API_URL=http://localhost:8011
APP_BASE_URL=http://localhost:3003

# زرین‌پال
ZARINPAL_MERCHANT_ID=your-zarinpal-merchant-id
```

> **حالت سندباکس:** برای توسعه محلی از مرچنت آزمایشی زرین‌پال استفاده کنید. برای محیط عملیاتی، مرچنت واقعی را جایگزین کنید.

---

## گردش پرداخت

```
POST /create  →  ریدایرکت به زرین‌پال  →  GET /callback  →  POST verify  →  ✅ / ❌
```

**مسیرهای بک‌اند**

| مسیر | توضیح |
|---|---|
| `POST /api/payment/zarinpal/create` | ایجاد سفارش پرداخت |
| `GET  /api/payment/zarinpal/callback` | مدیریت بازگشت از درگاه و تأیید |
| `GET  /api/payment/zarinpal/status/{id}` | وضعیت سفارش |
| `GET  /api/payment/zarinpal/history` | تاریخچه پرداخت کاربر |

**صفحات فرانت‌اند**

| مسیر | توضیح |
|---|---|
| `/payment/checkout` | انتخاب پلن و آغاز پرداخت |
| `/payment/callback/zarinpal` | پل بازگشت از درگاه |
| `/payment/success` | تأیید موفقیت |
| `/payment/failed` | صفحه خطا |

مایگریشن دیتابیس را یک‌بار اجرا کنید:

```bash
psql -d trading_db -f database/schemas/payment_orders.sql
```

---

## ساختار پروژه

```
Findash/
├── frontend-nextjs/
│   └── src/
│       ├── app/
│       │   ├── dashboard/          # داشبورد اصلی
│       │   ├── portfolio/          # پرتفوی و معاملات
│       │   ├── auth/               # ورود / ثبت‌نام
│       │   └── payment/            # پرداخت، موفقیت، خطا
│       ├── components/
│       │   ├── realtime/           # فید WebSocket
│       │   ├── portfolio/          # رهگیر معاملات، سود و زیان
│       │   └── risk/               # گیج ریسک
│       └── lib/
│           └── hooks/              # use-market-ws و غیره
├── src/
│   ├── main_refactored.py          # نقطه ورود اپ FastAPI
│   ├── api/endpoints/              # پرداخت، احراز هویت، دارایی‌ها ...
│   └── core/config.py              # تنظیمات اپ
└── database/
    └── schemas/                    # مایگریشن‌های SQL
```

---

## نقشه راه

- [x] داده لحظه‌ای بازار ایران (tgju، نوبیتکس) — تب «بازار» و ticker زنده
- [ ] مدیریت پلن اشتراک ([#13](https://github.com/massoudsh/Findash/issues/13))
- [ ] احراز هویت مالی / KYC ([#20](https://github.com/massoudsh/Findash/issues/20))
- [ ] یکپارچگی کیف پول ریالی ([#21](https://github.com/massoudsh/Findash/issues/21))
- [ ] تولید گزارش PDF (فارسی) ([#18](https://github.com/massoudsh/Findash/issues/18))
- [ ] هشدارهای پوش و پیامک ([#19](https://github.com/massoudsh/Findash/issues/19))
- [ ] موتور سیاست ریسک ([#22](https://github.com/massoudsh/Findash/issues/22))
- [ ] پنل مدیریت برای تراکنش‌ها و کاربران ([#12](https://github.com/massoudsh/Findash/issues/12))

برای فهرست کامل issue های باز، به [GitHub Issues](https://github.com/massoudsh/Findash/issues) مراجعه کنید.

---

## مشارکت

۱. ریپازیتوری را فورک کرده و یک برنچ برای فیچر بسازید
۲. تغییرات را کوچک و متمرکز نگه دارید
۳. پیش از باز کردن PR، جریان‌های اصلی کاربر را به‌صورت دستی تست کنید
۴. برای تغییرات پرداخت یا احراز هویت — سناریوهای خطا را هم پوشش دهید

---

## مجوز

MIT © [massoudsh](https://github.com/massoudsh) — برای جزئیات به [`LICENSE`](LICENSE) مراجعه کنید.
