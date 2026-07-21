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
| `/api/startup-tracker/hypotheses` | CRUD فرضیه‌های GTM (داخلی/ادمین) |
| `/api/startup-tracker/conversations` | CRUD مکالمات با مشتری، قابل لینک به یک فرضیه |
| `/api/startup-tracker/traction` | CRUD داده‌های Traction (کاربر، درآمد، تعامل، نگه‌داشت) |
| `/api/startup-tracker/summary` | آمار تجمیعی برای کارت‌های داشبورد استارتاپ‌تراکر |
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
- `start.py` (ریشه ریپو) — **نقطهٔ ورود واقعیِ اجرا/production** (`CMD ["python", "start.py"]` در `docker/Dockerfile.fastapi`؛ خودش با `uvicorn.Config(app="src.main_refactored:app", ...)` سرور را بالا می‌آورد؛ همچنین validation محیط/dependency قبل از start انجام می‌دهد)
- `src/main_refactored.py` — تعریف واقعی FastAPI `app` و ثبت routerها (خودِ این فایل مستقیماً اجرا نمی‌شود؛ `Makefile`'s dev target هم `uvicorn src.main_refactored:app --reload` است، نه `start.py`)
- `src/api/endpoints/payment_zarinpal.py` — یکپارچه‌سازی زرین‌پال: create/callback/verify/status/history
- `src/api/endpoints/startup_tracker.py` — استارتاپ‌تراکر: hypotheses/conversations/traction/summary (in-memory store، همان الگوی `strategies_crud.py`)
- `src/api/bots_persistence.py` — persistence ساده JSON برای Trading Bots (`load_bots`/`save_bots`, فایل در `data/trading_bots.json`)؛ قبلاً این فایل مفقود بود و کل `src.main_refactored` (و در نتیجه کل pytest suite) را می‌شکست — در TASK-025 اضافه شد
- `database/schemas/payment_orders.sql` — schema جدول payment_orders
- پورت پیش‌فرض: `localhost:8000`

## ✅ رفع‌شده: ناسازگاری bcrypt/passlib در auth (`professional_auth.py`, `security.py`)
`bcrypt>=5.0.0` ویژگی `__about__` را که `passlib==1.7.4` برای تشخیص نسخه به آن وابسته است حذف کرده؛ در نتیجه `hash_password()`/`verify_password()` در `src/core/security.py` استثنا پرتاب می‌کردند. این استثنا در بلاک‌های `except Exception` عمومی endpointهای `professional_auth.py` (login/register/refresh/profile/logout/api-keys) بلعیده می‌شد و `AuthResponse(success=False)` با کد `200` برمی‌گشت — یعنی login نامعتبر به‌جای `401` کد `200` می‌داد، register همیشه «Registration failed» می‌داد و مسیرهای احرازشده به‌طور غیرمنتظره fail می‌شدند.
رفع: پین کردن `bcrypt==4.1.2` (سازگار با `passlib==1.7.4`) در `requirements/requirements.txt` و `requirements/requirements-basic.txt` (هم‌راستا با `requirements-quickstart.txt` که همین pin را از قبل داشت) + نصب واقعی در محیط. بعد از رفع: `tests/test_auth.py` کامل ۲۷/۲۷ pass می‌شود.
