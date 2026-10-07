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
| `/api/payment/zarinpal/create` | ایجاد سفارش پرداخت زرین‌پال (فقط purpose=general/wallet_topup — subscription مجاز نیست) |
| `/api/payment/zarinpal/callback` | callback زرین‌پال + verify اجباری + dispatch اثر جانبی بر اساس purpose |
| `/api/payment/zarinpal/status/{id}` | وضعیت سفارش پرداخت |
| `/api/payment/zarinpal/history` | تاریخچه پرداخت کاربر |
| `/api/subscriptions/plans` | لیست پلن‌های اشتراک فعال (basic/pro/elite) |
| `/api/subscriptions/subscribe` | ایجاد سفارش زرین‌پال برای خرید اشتراک؛ قیمت سمت سرور از DB خوانده می‌شود |
| `/api/subscriptions/me` | وضعیت اشتراک کاربر جاری (برای gating) |
| `/api/wallet/balances`, `/transactions` | موجودی و تراکنش‌های واقعی کیف پول ریالی (IRT) |
| `/api/wallet/deposit` | شارژ کیف پول از طریق زرین‌پال (purpose=wallet_topup) |
| `/api/wallet/withdraw` | درخواست برداشت به شبا (pending تا تسویه عملیاتی/دستی؛ بدون اتصال Payout) |
| `/api/wallet/bank-accounts*` | مدیریت حساب‌های شبا با اعتبارسنجی checksum مود-۹۷ |
| `/api/admin/users`, `/audit-log` | پنل مدیریت واقعی، gated با `require_admin`، بدون mock |
| `/api/startup-tracker/hypotheses` | CRUD فرضیه‌های GTM (داخلی/ادمین) |
| `/api/startup-tracker/conversations` | CRUD مکالمات با مشتری، قابل لینک به یک فرضیه |
| `/api/startup-tracker/traction` | CRUD داده‌های Traction (کاربر، درآمد، تعامل، نگه‌داشت) |
| `/api/startup-tracker/summary` | آمار تجمیعی برای کارت‌های داشبورد استارتاپ‌تراکر |
| `/api/investor-tools/watchlists` | CRUD واچ‌لیست‌های کاربر (حداکثر ۵ مورد) |
| `/api/investor-tools/screener` | اسکرینر دارایی‌های پشتیبانی‌شده با دادهٔ زنده ایران |
| `/api/investor-tools/paper` | حساب و سفارش‌های شبیه‌سازی‌شده، بدون ارسال به کارگزار |
| `/api/investor-tools/events` | قرارداد تقویم رویداد؛ فقط دادهٔ تأییدشده واقعی نمایش داده می‌شود |
| `/api/investor-tools/dividends` | ثبت و خلاصهٔ سود نقدی دریافت‌شده یا در انتظار |
| `/api/kyc/submit`, `/me` | ثبت مدارک KYC (کدملی با اعتبارسنجی checksum واقعی + موبایل) و وضعیت خود کاربر |
| `/api/kyc/pending`, `/{kyc_id}/review` | صف بررسی و تأیید/رد KYC توسط ادمین (`AuditLog` + اعلان به کاربر) |
| `/api/alerts/rules` (GET/POST/DELETE) | هشدار قیمت کاربر (نماد + جهت + قیمت هدف) |
| `/api/alerts/push-subscribe` (POST/DELETE) | ثبت/حذف اشتراک Web Push (VAPID) |
| `/api/alerts/evaluate` | اجرای دستی ارزیابی هشدارها (ادمین) — همان تابعی که Celery beat هر ۱ دقیقه صدا می‌زند |
| `/api/risk-policy/me` (GET/PUT) | مشاهده/تنظیم سیاست ریسک کاربر (حداکثر افت روزانه، حداکثر تمرکز یک دارایی، اقدام هنگام نقض) |
| `/api/risk-policy/breaches` | تاریخچهٔ نقض‌های ثبت‌شده |
| `/api/risk-policy/me/evaluate`, `/evaluate-all` | ارزیابی دستی (خود کاربر / همه — ادمین)؛ `evaluate-all` همان تابعی که Celery beat هر ۵ دقیقه صدا می‌زند |
| `/api/reports/portfolio.pdf` | گزارش PDF فارسیِ پورتفوی کاربر جاری؛ شامل خلاصه ارزش/سود‌وزیان، پوزیشن‌ها، ریسک، معاملات و بینش‌های ساده. فقط با access token و تنها برای پرتفوی همان کاربر. |
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
- `start.py:16` — مسیر absolute ریشهٔ پروژه را پیش از importهای `src` به `sys.path` اضافه می‌کند؛ `src/__init__.py` نیز پکیج backend را صریح می‌کند. اجرای توصیه‌شده: `python3 start.py --reload`؛ در صورت اجرای مستقیم uvicorn، حتماً از ریشهٔ repository اجرا شود.
- `src/main_refactored.py` — تعریف واقعی FastAPI `app` و ثبت routerها (خودِ این فایل مستقیماً اجرا نمی‌شود؛ `Makefile`'s dev target هم `uvicorn src.main_refactored:app --reload` است، نه `start.py`)
- `src/api/endpoints/payment_zarinpal.py` — یکپارچه‌سازی زرین‌پال: create/callback/verify/status/history
- `src/api/endpoints/startup_tracker.py` — استارتاپ‌تراکر: hypotheses/conversations/traction/summary (in-memory store، همان الگوی `strategies_crud.py`)
- `src/api/bots_persistence.py` — persistence ساده JSON برای Trading Bots (`load_bots`/`save_bots`, فایل در `data/trading_bots.json`)؛ قبلاً این فایل مفقود بود و کل `src.main_refactored` (و در نتیجه کل pytest suite) را می‌شکست — در TASK-025 اضافه شد
- `src/api/endpoints/investor_tools.py` — APIهای user-scoped ابزار سرمایه‌گذاری؛ state موقت JSON در `data/investor_tools.json` و اسکرینر متصل به `iran_market.get_overview()`
- `frontend-nextjs/src/app/investing/page.tsx` — رابط RTL ابزارهای سرمایه‌گذاری در مسیر `/investing`
- `database/schemas/payment_orders.sql` — schema جدول payment_orders
- پورت پیش‌فرض: `localhost:8000`

## ✅ رفع‌شده: ناسازگاری bcrypt/passlib در auth (`professional_auth.py`, `security.py`)
`bcrypt>=5.0.0` ویژگی `__about__` را که `passlib==1.7.4` برای تشخیص نسخه به آن وابسته است حذف کرده؛ در نتیجه `hash_password()`/`verify_password()` در `src/core/security.py` استثنا پرتاب می‌کردند. این استثنا در بلاک‌های `except Exception` عمومی endpointهای `professional_auth.py` (login/register/refresh/profile/logout/api-keys) بلعیده می‌شد و `AuthResponse(success=False)` با کد `200` برمی‌گشت — یعنی login نامعتبر به‌جای `401` کد `200` می‌داد، register همیشه «Registration failed» می‌داد و مسیرهای احرازشده به‌طور غیرمنتظره fail می‌شدند.
رفع: پین کردن `bcrypt==4.1.2` (سازگار با `passlib==1.7.4`) در `requirements/requirements.txt` و `requirements/requirements-basic.txt` (هم‌راستا با `requirements-quickstart.txt` که همین pin را از قبل داشت) + نصب واقعی در محیط. بعد از رفع: `tests/test_auth.py` کامل ۲۷/۲۷ pass می‌شود.

## ✅ رفع‌شده: `tests/test_main_endpoints.py` stale (route های legacy غیرموجود)
این فایل تست به یک `/auth/token` (فرم OAuth2 روی مدل SQLAlchemy `User`) و یک `/strategies/backtest` + `/strategies/results/{id}` مبتنی بر Celery `AsyncResult` (ماژول فرضی `api.endpoints.strategies` بدون پیشوند `src.`) اشاره می‌کرد که هیچ‌کدام در اپ واقعی وجود ندارند — `/api/auth/*` (`professional_auth.py`, از قبل با `tests/test_auth.py` پوشش کامل دارد) و `/api/backtesting/run`+`/api/backtesting/results/{id}` (`src/api/endpoints/backtesting.py`, همگام و auth-gated) معادل واقعی موجود هستند. به‌جای اضافه‌کردن دوباره یک router قدیمی موازی (که صرفاً یک feature تکراری می‌ساخت)، فایل تست بازنویسی شد تا route های واقعی موجود را تست کند. نتیجه: `tests/test_main_endpoints.py` کامل ۹/۹ pass؛ کل suite 163 passed/1 error (فقط `test_ingestion_pipeline.py` که به Postgres واقعی نیاز دارد، مستند/خارج از scope).

## ✅ رفع‌شده (۲۰۲۶-۰۹-۰۱): زیرساخت مشترک پرداخت + اشتراک + کیف پول + ادمین
- **باگ JWT roles**: `verify_token()` در `security.py` فقط claim جمع `roles` را می‌خواند اما لاگین claim مفرد `role` می‌ساخت → `TokenData.roles` همیشه خالی بود و هیچ role-based authorization کار نمی‌کرد. رفع شد (هر دو شکل نرمالایز می‌شوند) + `require_admin` dependency اضافه شد.
- **باگ OTP SMS**: `otp_auth.py` با `getattr(settings, "SMS_PROVIDER", None)` که چنین attribute ای هرگز وجود نداشت، همیشه silent-fallback به dev-log می‌کرد حتی با credential واقعی. رفع شد با `SMSSettings`/`WebPushSettings` جدید در `config.py`.
- **باگ کرش TokenData.get()**: در `payment_zarinpal.py` سه endpoint، `current_user` را `dict` تایپ کرده و `.get()` صدا می‌زدند در حالی که `get_current_active_user` یک شیء Pydantic `TokenData` برمی‌گرداند (بدون متد `.get`) — هر تماس احرازشده با این endpointها 500 می‌داد. رفع شد.
- **زیرساخت مشترک پرداخت**: `create_order()`/`order_redirect_url()` در `payment_zarinpal.py` قابل reuse برای هر جریان دیگر است؛ `_dispatch_payment_success()` در callback بر اساس `PaymentOrder.purpose` موجودی کیف پول را شارژ یا اشتراک را فعال/تمدید می‌کند — همیشه بعد از verify موفق، نه قبل از آن.
- **جدول‌های جدید**: `AuditLog`, `Notification`, `SubscriptionPlan`, `UserSubscription`, `RiskPolicy`/`RiskPolicyBreach`, `PriceAlertRule`, `PushSubscription`, `KYCProfile` در `models.py`. `BankAccount` از `routing_number` آمریکایی به `sheba_number` ایرانی تغییر کرد.
- **کیف پول واقعی (`wallet.py`)**: قبلاً ۱۰۰٪ mock بود (موجودی fake USD/BTC/ETH، مدل‌های Pydantic محلی هم‌نام مدل‌های DB واقعی را shadow می‌کردند، auth کامنت بود). اکنون فقط IRT، از DB واقعی، واریز از طریق `create_order()`، برداشت با اعتبارسنجی شماره شبا (الگوریتم استاندارد IBAN mod-97).
- **پنل ادمین واقعی (`admin_panel.py`)**: کوئری واقعی روی `User`/`Portfolio`/`Trade` برای لیست کاربران + تعداد معاملات/ارزش پورتفولیو؛ تغییر role/is_active با ثبت `AuditLog`.
- محدودیت شناخته‌شده و مستند (خارج از scope کد): برداشت واقعی وجه به شبا نیازمند اتصال به یک ارائه‌دهنده Payout بانکی (مثلاً جیبیت) و تسویه عملیاتی/دستی است — این تصمیم تجاری/قراردادی است، نه یک باگ کد.

## ✅ رفع‌شده (۲۰۲۶-۰۹-۰۳): issue های #14/#19/#20/#22 پیاده‌سازی شدند
- **KYC واقعی (`kyc.py`, #20)**: اعتبارسنجی چک‌سام واقعی کدملی ایران (نه فقط رشتهٔ آزاد) + نرمال‌سازی موبایل، جریان `pending_review` → `verified`/`rejected` با endpoint بررسی مخصوص ادمین، ثبت `AuditLog` و اعلان درون‌برنامه‌ای به کاربر. استعلام خودکار از رجیستری هویت واقعی نیازمند قرارداد با ارائه‌دهندهٔ مجاز (فینوتک/جیبیت/زیبال) است — تصمیم تجاری، ستون‌های `provider`/`provider_reference` برای آن آماده نگه داشته شده‌اند؛ فعلاً بررسی توسط ادمین دستی است.
- **هشدار قیمت (`price_alerts.py`, #19)**: قانون نماد+جهت+قیمت‌هدف روی دادهٔ واقعی `iran_market.get_overview()`؛ کانال‌های SMS/Web Push (VAPID)/درون‌برنامه‌ای از طریق `services/notifications.py` مشترک.
- **سیاست ریسک (`risk_policy.py`, #22)**: حداکثر افت روزانه٪ و حداکثر تمرکز یک دارایی٪ قابل‌تنظیم روی داده واقعی `PortfolioSnapshot`/`Position`؛ در صورت نقض با `action_on_breach='stop_bots'`، همهٔ ربات‌های فعال کاربر واقعاً متوقف می‌شوند (همان لایهٔ persistence که `trading_bots.py` استفاده می‌کند). گارد ضد-اسپم: هر نقض فقط یک‌بار در روز اعلان می‌شود.
- **gating اشتراک (`subscriptions.require_active_subscription`, #14)**: قبلاً معاملهٔ خودکار (ویژگی پلن pro/elite) حتی برای کاربر مهمان باز بود چون شروع ربات از `get_optional_user` استفاده می‌کرد. اکنون شروع ربات نیازمند اشتراک فعال است (ادمین معاف)؛ ساخت/توقف/تست paper ربات محدود نشده.
- **Celery beat واقعی (`src/notifications/tasks.py`)**: `evaluate_price_alerts` (هر ۱ دقیقه) و `evaluate_all_risk_policies` (هر ۵ دقیقه) در `core/celery_app.py` رجیستر شدند و همان تابع endpoint دستی ادمین را صدا می‌زنند (بدون دوبار‌نویسی منطق).
- روترهای `kyc`/`price_alerts`/`risk_policy` قبلاً کد کامل داشتند ولی در `main_refactored.py` mount نشده بودند (پس هیچ endpoint‌شان واقعاً در دسترس نبود) — رفع شد.
- ابزار کوچک جدید `core/persian_utils.py` (بدون وابستگی خارجی) برای تبدیل ارقام لاتین به فارسی در متن پیامک/اعلان (که برخلاف صفحات وب، از فرمتر Jalali فرانت‌اند رد نمی‌شود) — در متن اعلان هشدار قیمت و نقض ریسک استفاده شده.

## ✅ رفع‌شده: اتصال واقعیِ auth (`professional_auth.py`) به جدول `users` در PostgreSQL
قبلاً login/register/profile/list-users/refresh/change-password همگی از یک dict درون‌حافظه‌ای با ۳ کاربر دمو استفاده می‌کردند (توضیح کامل قبلی در [[concepts/auth-flow]]). اکنون همه این endpointها `db: Session = Depends(get_db)` می‌گیرند و واقعاً روی مدل `User` (`src/database/models.py`) کوئری می‌زنند؛ ۳ اکانت دمو ثابت با `_ensure_demo_users()` به‌صورت idempotent seed می‌شوند تا رفتار قبلی (login با `demo@octopus.trading`/... ) حفظ شود اما این‌بار روی رکوردهای واقعی دیتابیس. جزئیات کامل و مسائل باقی‌مانده (اسکیمای SQL orphan، migration خالی آلمبیک) در [[concepts/auth-flow]] مستند شده.
