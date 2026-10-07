# Auth Flow (Login)

> مسیر کامل ورود کاربر از فرم `/auth/signin` در Frontend تا پاسخ نهایی. این مسیر اکنون واقعاً به جدول `users` در PostgreSQL وصل است (از طریق مدل SQLAlchemy در `src/database/models.py`)؛ دیگر از in-memory dict استفاده نمی‌کند.

## مراحل

1. **Frontend UI** — `frontend-nextjs/src/app/auth/signin/page.tsx`: از ۲۰۲۶-۰۹-۲۵ دیگر فرم email/password را نمایش نمی‌دهد؛ صفحه فقط preview و CTA مستقیم به `/dashboard` دارد، بنابراین مسیر عمومی کاربر جدید بدون credential وارد داشبورد نمونه می‌شود (`signin/page.tsx:40`). مسیر Credentials/NextAuth زیر برای استفاده‌های backend/admin و سازگاری باقی مانده، اما از UI عمومی signin فراخوانی نمی‌شود.
2. **NextAuth Route Handler** — `frontend-nextjs/src/app/api/auth/[...nextauth]/route.ts`: provider از نوع `CredentialsProvider`؛ در `authorize()` یک `POST` به `${BACKEND_URL}/api/auth/login` می‌زند (`BACKEND_INTERNAL_URL` یا `NEXT_PUBLIC_API_URL`، پیش‌فرض `http://localhost:8011`).
3. **Backend endpoint** — `src/api/endpoints/professional_auth.py`:
   - `POST /api/auth/login` یک `db: Session = Depends(get_db)` می‌گیرد و delegate می‌کند به `authenticate_credentials(credentials, request, db)` (همان تابع پشت `POST /api/auth/credentials`).
   - ابتدا `auth_rate_limit` (`src/core/rate_limiter.py`) چک می‌شود — Redis-backed با fallback درون‌حافظه‌ای؛ ۵ تلاش ناموفق → قفل موقت (۴۲۹).
   - `_ensure_demo_users(db)` ۳ اکانت دمو ثابت (`admin@octopus.trading`, `trader@octopus.trading`, `demo@octopus.trading`) را در جدول واقعی `users` seed می‌کند (پسورد از env `DEMO_ADMIN_PASSWORD`/`DEMO_TRADER_PASSWORD`/`DEMO_USER_PASSWORD` یا مقدار پیش‌فرض). **باگ رفع‌شده:** قبلاً فقط در صورت نبودن کاربر آن را می‌ساخت؛ اگر ردیف با هش پسورد قدیمی/متفاوت از قبل در دیتابیس وجود داشت (مثلاً از دوره‌ای که هش‌سازی خراب بود، یا env var عوض شده بود)، دکمه‌های راهنمای «حساب‌های آزمایشی» در `/auth/signin` برای همیشه با خطای «ایمیل یا رمز عبور اشتباه است» fail می‌شدند. اکنون هر بار با `verify_password()` هش موجود چک می‌شود و در صورت عدم تطابق (یا `is_active=False`) خودش را با پسورد جاری sync می‌کند — self-healing، idempotent. تست رگرسیون: `tests/test_auth.py::TestDemoAccountSelfHeal`.
   - کاربر با `db_crud.get_user_by_email(db, email)` از جدول واقعی `users` (مدل `User` در `src/database/models.py`) خوانده می‌شود — **کوئری واقعی به PostgreSQL**، نه dict درون‌حافظه‌ای.
   - `verify_password()` (`src/core/security.py`, بر پایه‌ی `passlib`/`bcrypt`) پسورد را با `user.password_hash` مقایسه می‌کند؛ همچنین `user.is_active` چک می‌شود.
   - در موفقیت: `user.last_login = datetime.utcnow()` روی رکورد واقعی commit می‌شود؛ سپس `create_access_token()`/`create_refresh_token()` یک JWT با `jose.jwt.encode` (کلید `settings.auth.jwt_secret_key`، claim ها شامل `user.id`, `user.email`, `user.role`, `user.permissions`) می‌سازند و `AuthResponse(success=True, user=..., access_token=..., refresh_token=...)` برگردانده می‌شود.
4. **بازگشت به NextAuth**: `authorize()` نتیجه را به `{id, email, name, accessToken, refreshToken}` map می‌کند → callback `jwt()` توکن‌ها را در JWT session ذخیره می‌کند → callback `session()` آن‌ها را در `session.accessToken` قرار می‌دهد (`session: {strategy: "jwt"}`، یعنی session خودِ NextAuth هم در دیتابیس ذخیره نمی‌شود، فقط JWT کوکی).
5. **Frontend**: در موفقیت `window.location.href = "/dashboard"`؛ در خطا پیام «ایمیل یا رمز عبور اشتباه است» نمایش داده می‌شود.

## اتصال واقعی به PostgreSQL (وضعیت فعلی)
- مدل `User` (جدول `users`) در `src/database/models.py:7` گسترش داده شد: فیلدهای جدید `first_name`, `last_name`, `role` (`admin`/`trader`/`demo`)، `permissions` (JSON)، `last_login` (TIMESTAMP) اضافه شدند.
- تمام endpointهای `professional_auth.py` (`authenticate_credentials`, `/login`, `register_user`, `list_users`, `get_user_profile`, `refresh_token_endpoint`, `change_password`) اکنون `db: Session = Depends(get_db)` می‌گیرند و از `src/database/crud.py` (`get_user_by_email` — تازه اضافه شده — و `get_user_by_id`, `get_user_by_username`) یا کوئری مستقیم SQLAlchemy استفاده می‌کنند.
- ثبت‌نام کاربر جدید (`register_user`) یک ردیف واقعی `User` می‌سازد؛ username به‌صورت خودکار و یکتا از ایمیل مشتق می‌شود (`_derive_unique_username`).
- **باگ رفع‌شده:** دو `declarative_base()` جدا در کدبیس وجود داشت — یکی در `models.py` (که `User`/`Portfolio`/... از آن استفاده می‌کنند) و یکی دیگر در `src/database/postgres_connection.py` (فقط برای `PaymentOrder`). تابع `create_tables()` در `postgres_connection.py` فقط Base دوم را می‌ساخت، یعنی جدول `users` هرگز واقعاً روی یک Postgres تازه ساخته نمی‌شد. اکنون `create_tables()`/`drop_tables()` هر دو `Base.metadata` را می‌سازند/حذف می‌کنند. `src/core/initialization.py` هم اکنون بعد از `init_db_connection()` این `create_tables()` را صدا می‌زند (با try/except تا اگر DB در دسترس نبود اپ crash نکند).
- Rate limiting از طریق Redis (با fallback درون‌حافظه‌ای) بدون تغییر باقی مانده.
- تست‌ها: `tests/test_auth.py` اکنون از fixture `client` در `tests/conftest.py` استفاده می‌کند (SQLite in-memory + `app.dependency_overrides[get_db]`) به‌جای `TestClient(app)` مستقیم، چون همه endpointها اکنون واقعاً به `get_db` وابسته‌اند.

## گیت‌کردن مسیرها (`middleware.ts`) — وضعیت فعلی: موقتاً باز
- `frontend-nextjs/src/middleware.ts` روی `/dashboard`, `/portfolio`, `/trading`, `/analytics`, `/settings` matcher دارد اما به‌درخواست کاربر callback `authorized` به `() => true` تغییر کرد — یعنی این مسیرها **فعلاً بدون لاگین هم در دسترس‌اند** تا کاربر جدید بتواند کل پلتفرم را بدون ثبت‌نام ببیند.
- از ۲۰۲۶-۰۹-۲۵ صفحه signin هم دیگر credential نمی‌گیرد و مستقیم به داشبورد نمونه لینک می‌دهد (`signin/page.tsx:44`). داشبورد با متن عمومی/نمونه نمایش داده می‌شود (`dashboard/page.tsx:72`)؛ تب پرتفولیو از `MOCK_PORTFOLIOS`/`MOCK_POSITIONS` داخلی استفاده می‌کند (`portfolio-content.tsx:58`) و تب بازار در نبود API یا داده معتبر به `MOCK_MARKET_ITEMS` برمی‌گردد (`iran-market-overview.tsx:27`).
- **استثنا: `/admin` گیت‌شده ماند.** `frontend-nextjs/src/app/admin/layout.tsx` یک server component است که با `getServerSession(authOptions)` سشن را می‌خواند، بدون سشن به `/auth/signin?callbackUrl=/admin` ریدایرکت می‌کند، و اگر `session.user.role !== 'admin'` باشد پیام «دسترسی محدود به مدیران سیستم است» نشان می‌دهد. دلیل: تصمیم صریح کاربر (گزینه «b») — بازکردن `/admin` روی مهمان ناشناس یک حفره‌ی امنیتی مستقیم است چون این صفحه نقش/وضعیت کاربران را تغییر می‌دهد؛ در عوض داشبورد و بقیه‌ی پلتفرم عمومی می‌مانند. همچنین `/risk/policy` و `/account/subscription` سشن نیاز دارند (داده‌شان per-user است).

## ارجاع توکن به APIهای پشت‌صحنه — `lib/auth-options.ts`
- `frontend-nextjs/src/lib/auth-options.ts` (فایل جدید) تنها محل تعریف `authOptions` است (provider `CredentialsProvider` + callbackهای `jwt()`/`session()` که `role`/`accessToken` را propagate می‌کنند). `app/api/auth/[...nextauth]/route.ts` فقط این آبجکت را import می‌کند.
- **دام:** `getServerSession()` بدون آرگومان در route handlerهای Next.js 15 سشن را resolve نمی‌کند (NextAuth v4 در App Router به `authOptions` نیاز دارد). هر route proxy جدید پشت auth باید `getServerSession(authOptions)` با همین import بنویسد، وگرنه همیشه 401 می‌دهد حتی با کوکی معتبر. routeهای `/api/admin/*`، `/api/risk-policy/*` و `/api/subscriptions/*` همه از همین الگو استفاده می‌کنند و علاوه بر چک سشن، توکن را به‌صورت `Bearer ${session.accessToken}` به بک‌اند پاس می‌دهند.
- توجه: صفحه‌ی `/notifications` از `useSession()` برای `session.user.id` استفاده می‌کند (`concepts/auth-flow` ارجاع قبلی) — برای بازدیدکننده‌ی بدون لاگین `session` وجود ندارد، پس این صفحه‌ی خاص برای کاربر مهمان می‌تواند خالی/غیرفعال بماند؛ رفع کامل آن (نمایش مناسب برای مهمان) خارج از scope این تغییر بود.

## مسائل شناخته‌شده (خارج از scope این تغییر)
- یک اسکیمای SQL قدیمی/orphan مبتنی بر UUID (`database/schemas/01_initial_schema.sql` + `database/seeds/01_default_users.sql`) در ریپو وجود دارد که **اسکیمای واقعی نیست** — اسکیمای واقعی همان `models.py` است که Alembic (`src/alembic/env.py` → `target_metadata = Base.metadata` از `models.py`) هم به آن اشاره می‌کند. این فایل SQL قدیمی می‌تواند در آینده حذف یا با migration جایگزین شود تا سردرگمی ایجاد نکند.
- آخرین فایل migration آلمبیک (`src/alembic/versions/2025_07_04_2002-c5dc4ecea576_.py`) خالی (۰ بایت) است — یعنی زنجیره migration شکسته است. **جزئی رفع شد (۲۰۲۶-۰۹-۱۹):** migration `src/alembic/versions/20260919_admin_risk_subscription_schema.py` (revision `admin_risk_sub_001`، down_revision `unify_schema_001`) ساخته شد و ۶ جدولی که قبلاً فقط `create_tables()` هنگام startup می‌ساخت را رسماً به زنجیره migration اضافه می‌کند: `audit_logs`, `notifications`, `subscription_plans`, `user_subscriptions`, `risk_policies`, `risk_policy_breaches`. صحت ستون‌ها با `models.py` تک‌به‌تک تطبیق داده شد. زنجیره هنوز یک head واحد دارد. این فایل هنوز **اجرا نشده** است.
- ناسازگاری claim نام `role` (singular، هنگام encode) در برابر `roles` (plural، هنگام decode در `TokenData`) در `src/core/security.py` — از قبل موجود بود، در این تغییر دست نخورده.

## اجزای درگیر
- [[entities/frontend]] — فرم signin و NextAuth route handler
- [[entities/backend]] — `professional_auth.py`, `core/security.py`, `core/rate_limiter.py`, `database/crud.py`, `database/models.py`, `database/postgres_connection.py`
- [[entities/data-layer]] — Redis (rate limit) + PostgreSQL جدول `users` (اکنون واقعاً استفاده می‌شود)

## منابع کد
- `frontend-nextjs/src/app/auth/signin/page.tsx`
- `frontend-nextjs/src/app/api/auth/[...nextauth]/route.ts`
- `src/api/endpoints/professional_auth.py:161` (`authenticate_credentials`), `:582` (`/login` → delegate با `db`)
- `src/core/security.py:44` (`hash_password`), `:51` (`verify_password`), `:74` (`create_access_token`)
- `src/core/rate_limiter.py` — Redis-backed rate limiter با fallback درون‌حافظه‌ای
- `src/database/models.py:7` (`User`، با فیلدهای جدید `role`/`permissions`/`last_login`/`first_name`/`last_name`)
- `src/database/crud.py` (`get_user_by_email` — جدید، `get_user_by_id`, `get_user_by_username`)
- `src/database/postgres_connection.py` (`get_db`, `create_tables`, `drop_tables` — رفع باگ دو Base جدا)
- `tests/conftest.py` (`client`, `db_session`, `test_user` fixtures — SQLite in-memory)
