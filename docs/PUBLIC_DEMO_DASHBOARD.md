# Public demo dashboard and local startup

Intent: the product can be opened without a session. `/dashboard` is a **public sample view**. Account, payment, admin, and OTP still exist, but they are not on the demo path.

Verified against `frontend-nextjs` (sign-in, dashboard, portfolio, Iran market, ticker, middleware, NextAuth) and backend (`start.py`, `src/__init__.py`, `src/core/config.py`, `src/api/endpoints/iran_market.py`, `otp_auth.py`).

---

## What a visitor sees

```
/  (landing)
 ├─ hero “دمو”     → /demo
 ├─ hero “اشتراک”  → /payment/checkout   (plans BFF; pay requires session)
 └─ bottom CTA     → /auth/signup
/dashboard                    still public sample data, no login required
/auth/signin                  email/password via NextAuth + “مشاهده داشبورد نمونه”
/auth/signup                  POST {getBackendUrl()}/api/auth/register → /auth/signin?signup=success
/auth/phone → /auth/otp       leftover SMS login UI (see pitfalls)
/admin                        NextAuth session with role admin; callbackUrl=/admin now has a form
```

`middleware.ts` matches `/dashboard`, `/portfolio`, `/trading`, `/analytics`, and `/settings`, then always returns `authorized: () => true`. The comment still mentions restoring `LOGIN_ENABLED` on the sign-in page; that flag is **not** in the current sign-in UI.

Admin is gated in `app/admin/layout.tsx` (`getServerSession(authOptions)` + `role === 'admin'`). Unauthenticated admin hits `/auth/signin?callbackUrl=/admin`. `safeCallbackPath` keeps `/admin` after a successful login.

---

## Dashboard tabs and data sources

Route: `frontend-nextjs/src/app/dashboard/page.tsx`. Tab state is `?tab=` (`overview` is the default and is stripped from the URL). The header labels the page as a public sample dashboard.

| Tab | Component | Data |
|-----|-----------|------|
| overview | `OverviewDashboard` | Hardcoded Persian cards, allocation, sparkline, positions, activity. Risk gauge ticks locally every 3s (`Math.random()`), not from the API. |
| portfolio | `PortfolioContent` | localStorage holdings + client Dietz + in-component `MOCK_PORTFOLIOS` / `MOCK_POSITIONS`. `?addAsset=1` opens the add modal. No `getPortfolios()` call. |
| market | `IranMarketOverview` | Live `GET /api/iran-market/overview` only. No mock seed; failure keeps last good list. Quote unit is rial — [IRAN_MARKET_PRICING.md](IRAN_MARKET_PRICING.md). |
| trades / analytics / help | `TradeTracker`, `AnalyticsOverview`, `HelpCenter` | UI-only; help copy notes TEDPIX is not live. |

Blue ticker (`BlueTickerBar` → `useIranTicker`): `GET {API}/api/iran-market/ticker` every 60s. On failure it keeps the previous list (empty on first load). Label “بازار زنده” means the fetch ran, not that every symbol is live.

Backend ticker (`src/api/endpoints/iran_market.py`):

- Overview is cached 60s and pulls tgju.org + Nobitex.
- Ticker symbols: `USD-IRR`, `GOLD18-IRT`, `COIN-IRT`, `BTC-IRT`, `TEDPIX`.
- **TEDPIX is a placeholder** (`available: false`, `price: null`). It is not in the tgju/Nobitex payload.

Do not treat overview cards, mock portfolio rows, or TEDPIX as live holdings. The **market tab** is live quotes (or empty), not a second mock dataset.

---

## Ports (the usual mix-up)

| How you start | Frontend | FastAPI |
|---------------|----------|---------|
| `cd frontend-nextjs && npm run dev` | **3003** (`package.json`) | — |
| `python3 start.py --reload` or `make dev` | — | **8000** (`API_PORT`, default) |
| `docker compose -f docker-compose-core.yml up` | **3003** | host **8011** → container 8000 |

Client ticker (`useIranTicker`), market tab, signup, and NextAuth `authorize()` all fall back through `NEXT_PUBLIC_API_URL` / `getBackendUrl()` to **`http://localhost:8000`**. Compose still publishes the API on **8011**.

Local UI + Compose without `.env.local`: ticker/overview call **8000** while the host mapping is **8011**. Set `NEXT_PUBLIC_API_URL=http://localhost:8011` in `frontend-nextjs/.env.local` (and rebuild the frontend image if that var is baked at build time).

---

## Local backend startup (`start.py`)

Run from **repo root**, not `src/`:

```bash
python3 start.py --reload          # default :8000
python3 start.py --validate-only   # env + Redis/Postgres ping, then exit
```

`start.py` inserts the project root on `sys.path` **before** `from src.core.config import get_settings`. `src/__init__.py` makes `src` a package. Importing `src.*` before that insert used to fail with `ModuleNotFoundError: No module named 'src'`.

Development (`ENVIRONMENT=development`, the default):

- `SECRET_KEY` / `JWT_SECRET_KEY` / `DATABASE_URL` / `REDIS_URL` are recommended, not required.
- Redis/Postgres failures log a warning and the process still starts.

Production: those four env vars are required; Redis/Postgres failure exits 1.

`make dev` starts uvicorn on `:8000` directly (`src.main_refactored:app`) and does not run `start.py`’s env checks.

---

## Auth that still exists (not the demo path)

| Path | Behavior |
|------|----------|
| NextAuth `authorize()` | `POST {BACKEND_INTERNAL_URL \|\| getBackendUrl()}/api/auth/login` with **email + password**. Fallback is **`:8000`**. |
| `/auth/signin` | Credentials form + sample-dashboard link. `safeCallbackPath` keeps same-origin `callbackUrl` (checkout, admin). |
| `/auth/signup` | `POST {getBackendUrl()}/api/auth/register`. Success → `/auth/signin?signup=success` (forwards `callbackUrl`). |
| `/auth/phone`, `/auth/otp` | Call **`/api/proxy/auth/send-otp`** and **`/api/proxy/auth/verify-otp`**. There is **no** `frontend-nextjs/src/app/api/proxy/` tree. OTP verify then `signIn('credentials', { phone, otp_token })`, but `authOptions` only reads `email`/`password`. |
| FastAPI OTP | `POST /api/auth/send-otp` and `/verify-otp` in `otp_auth.py`. In-memory store; KaveNegar if `SMS_PROVIDER`/`SMS_API_KEY` are set, otherwise the code is logged. |

Session-backed pages (`/account/subscription`, `/risk/policy`, `/payment/checkout`, admin BFF) still call `useSession` / `getServerSession`. Checkout loads plans from `GET /api/subscriptions/plans` (public BFF) and sends pay to `POST /api/subscriptions/subscribe`. Unauthenticated pay → `/auth/signin?callbackUrl=/payment/checkout`.

---

## i18n constraint (Sep 2026 page labels)

- Default locale is **fa** (`LocaleProvider`, `getStoredLocale()`).
- `translations.ts` + `useTranslations()` cover **nav and command palette**.
- Dashboard, trading, fundamental, options, and the rest of the Sep 26 label pass use **hardcoded Persian**. The language switcher does not rewrite those strings.

Add nav keys in `translations.ts`. For feature pages, either keep Persian source or plumb `t('…')` — mixing both is how English chrome + Persian body happens.

---

## Troubleshooting

| Symptom | Cause | What to do |
|---------|--------|------------|
| `ModuleNotFoundError: src` | `start.py` not run from repo root, or an old copy that inserts `sys.path` after the `src` import | From repo root: `python3 start.py --reload` |
| Ticker / market empty under Compose | Client default API is now `:8000`; host mapping is `:8011` | `NEXT_PUBLIC_API_URL=http://localhost:8011` or use local `start.py` on 8000 |
| Market tab empty (no fake cards) | Mock seed removed; need ≥1 `available` overview item | Expected. `curl` `/api/iran-market/overview` |
| Login form errors “ورود ناموفق بود” | `authorize()` cannot reach FastAPI or body is not `{success, user, access_token}` | Match `NEXT_PUBLIC_API_URL` / `BACKEND_INTERNAL_URL` to the API |
| Admin redirect used to dump you on the sample dashboard | Pre-`0aee22b` sign-in had no form | Sign in with an `admin` user; `callbackUrl=/admin` is kept |
| Frontend README said `:3000` | `npm run dev` binds **3003** | Use 3003 |

---

## Related

- App routes and nav: [APP_FLOW.md](APP_FLOW.md)
- Session / BFF: [FRONTEND_SESSION.md](FRONTEND_SESSION.md)
- Quote units: [IRAN_MARKET_PRICING.md](IRAN_MARKET_PRICING.md)
- Repo layout and commands: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Wiki setup: [Getting Started](../wiki-content/Getting-Started.md)
- Frontend overview: [frontend-nextjs/README.md](../frontend-nextjs/README.md)
