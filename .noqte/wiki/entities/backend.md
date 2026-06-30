# Backend

> سرور FastAPI پلتفرم اختاپوس — پردازش معاملات، داده بازار، AI/ML، ریسک.

## مسئولیت‌ها
- احراز هویت و مدیریت session
- دریافت و سرویس‌دهی داده‌های بازار
- اجرا و مدیریت سفارش‌های معاملاتی
- ارائه endpoint های هوش مصنوعی
- آنالیز ریسک و گزارش‌دهی
- ارسال داده ریل‌تایم از طریق WebSocket

## APIهای اصلی
| مسیر | عملکرد |
|------|--------|
| `/api/market-data` | داده‌های بازار |
| `/api/trades` | عملیات معاملاتی |
| `/api/portfolio` | مدیریت پورتفولیو |
| `/api/risk` | آنالیز ریسک |
| `/api/ai-models` | endpoint های مدل AI |
| `/api/websocket` | اتصال WebSocket |
| `/api/payment/zarinpal/create` | ایجاد سفارش پرداخت زرین‌پال |
| `/api/payment/zarinpal/callback` | callback زرین‌پال + verify اجباری |
| `/api/payment/zarinpal/status/{id}` | وضعیت سفارش پرداخت |
| `/api/payment/zarinpal/history` | تاریخچه پرداخت کاربر |
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |

## وابستگی‌ها
- [[entities/data-layer]] — خواندن/نوشتن داده
- [[entities/orchestrator]] — هماهنگی task های AI
- [[concepts/trading-flow]] — flow اجرای معامله

## تکنولوژی
- FastAPI, Python 3.10+
- Celery (async tasks), WebSockets
- PyTorch, TensorFlow, scikit-learn (AI/ML)
- Alembic (migrations)

## منابع کد
- `src/main_refactored.py` — نقطه ورود اصلی و ثبت routerها
- `src/api/endpoints/payment_zarinpal.py` — یکپارچه‌سازی زرین‌پال: create/callback/verify/status/history
- `database/schemas/payment_orders.sql` — schema جدول payment_orders
- پورت پیش‌فرض: `localhost:8000`
