# Public demo dashboard and local startup

Intent: the product can be opened without a session. `/dashboard` is a **public sample view**. Account, payment, admin, and OTP still exist, but they are not on the demo path.

Verified against `frontend-nextjs` (sign-in, dashboard, portfolio, Iran market, ticker, middleware, NextAuth) and backend (`start.py`, `src/__init__.py`, `src/core/config.py`, `src/api/endpoints/iran_market.py`, `otp_auth.py`).

---

## What a visitor sees

```
/  (landing, frontend-nextjs/src/app/page.tsx)
 ├─ Hero primary CTA     → /demo
 ├─ Hero secondary CTA   → /payment/checkout
 ├─ Research directory   → static links (/technical, /macro, /fundamental-data, /data, /social, /dashboard, /ai-models)
 ├─ “راهنمای جریان کار” → /workflow
 └─ Bottom CTA “شروع کنید” → /auth/signup
/auth/signin                  CTA only: “مشاهده داشبورد نمونه” → /dashboard
/auth/signup                  still posts email/password to FastAPI /api/auth/register
/auth/phone → /auth/otp       leftover SMS login UI (see pitfalls)
/admin                        still requires a NextAuth session with role admin
```

The research directory is hardcoded `researchDirectory` in `page.tsx` (three `<details open>` groups). It is not a CMS or search index.

The features grid below that directory maps a `features` constant that is **not declared** in `page.tsx`. Rendering `/` throws `ReferenceError: features is not defined` when that section evaluates. Hero + research directory are in the same component, so the landing page currently does not mount.

`middleware.ts` matches `/dashboard`, `/portfolio`, `/trading`, `/analytics`, and `/settings`, then always returns `authorized: () => true`. The comment still mentions restoring `LOGIN_ENABLED` on the sign-in page; that flag is **not** in the current sign-in UI.

Admin is gated in `app/admin/layout.tsx` (`getServerSession(authOptions)` + `role === 'admin'`). Unauthenticated admin hits redirect to `/auth/signin?callbackUrl=/admin`, which now has **no credentials form**.

---

## Dashboard tabs and data sources

Route: `frontend-nextjs/src/app/dashboard/page.tsx`. Tab state is `?tab=` (`overview` is the default and is stripped from the URL). The header labels the page as a public sample dashboard.

| Tab | Component | Data |
|-----|-----------|------|
| overview | `OverviewDashboard` (eager) | Hardcoded Persian cards, allocation, sparkline, positions, activity. Risk gauge ticks locally every 3s (`Math.random()`), not from the API. |
| portfolio | `PortfolioContent` (`next/dynamic`, `ssr: false`) | `IranPortfolioSection` + Modified Dietz calculator + `MOCK_PORTFOLIOS` / `MOCK_POSITIONS` (فولاد, شستا, خودرو, BTC). No `getPortfolios()` call. Dietz is typed values only — [MODIFIED_DIETZ.md](MODIFIED_DIETZ.md). |
| market | `IranMarketOverview` (dynamic) | Starts on `MOCK_MARKET_ITEMS`. Tries `GET {API}/api/iran-market/overview`; on non-OK or empty `items`, stays on mock. **Default API `:8011`.** |
| trades / analytics / help | `TradeTracker`, `AnalyticsOverview`, `HelpCenter` (dynamic) | UI-only; help copy notes TEDPIX is not live. |

Blue ticker (`BlueTickerBar` → `useIranTicker`): `GET {NEXT_PUBLIC_API_URL\|\|http://localhost:8000}/api/iran-market/ticker` every 60s. `loading` is cleared in `finally` (failed first fetch no longer sticks on “در حال دریافت”). On error it keeps the previous list (empty on first load). Label “بازار زنده” means the fetch ran, not that every symbol is live. The bar prints **no currency suffix**.

Nav `NewsTicker` (`navigation-wrapper.tsx`) is a second consumer of the same hook plus `/api/news`. Available quotes are shown as **ریال**; the marquee duration is **75s**.

Backend (`src/api/endpoints/iran_market.py`):

- Overview is cached 60s and pulls tgju.org + Nobitex.
- Ticker symbols: `USD-IRR`, `GOLD18-IRT`, `COIN-IRT`, `BTC-IRT`, `TEDPIX`.
- **TEDPIX is a placeholder** (`available: false`, `price: null`). It is not in the tgju/Nobitex payload.
- Nobitex crypto `price` is kept in **Rial** (`currency: "IRR"`). Do not divide by 10 for Toman in new callers.

Do not treat overview cards, mock portfolio rows, Dietz inputs, or TEDPIX as live holdings.

---

## Ports (the usual mix-up)

| How you start | Frontend | FastAPI |
|---------------|----------|---------|
| `cd frontend-nextjs && npm run dev` | **3003** (`package.json`) | — |
| `python3 start.py --reload` or `make dev` | — | **8000** (`API_PORT`, default) |
| `docker compose -f docker-compose-core.yml up` | **3003** | host **8011** → container 8000 |

Fallbacks are **not** uniform:

| Caller | Default without `NEXT_PUBLIC_API_URL` |
|--------|----------------------------------------|
| `useIranTicker` (blue bar + `NewsTicker`) | `http://localhost:8000` |
| `getBackendUrl()` / `lib/services/api.ts` | `http://localhost:8000` |
| `IranMarketOverview`, `/auth/signup`, NextAuth `authorize()` | `http://localhost:8011` |

Local UI + local `start.py` without `.env.local`: the **blue ticker** hits **8000** (matches `start.py`); the **market tab and signup** still call **8011**. Compose without `.env.local`: ticker misses **8011**. Set `NEXT_PUBLIC_API_URL` to the API you actually run.

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
| NextAuth `authorize()` | `POST {backend}/api/auth/login` with **email + password**. Backend default in auth-options is `:8011`. |
| `/auth/signup` | `POST {API}/api/auth/register` (same `:8011` fallback). Success redirects to `/auth/signin?signup=success`, which no longer shows a login form. |
| `/auth/phone`, `/auth/otp` | Call **`/api/proxy/auth/send-otp`** and **`/api/proxy/auth/verify-otp`**. There is **no** `frontend-nextjs/src/app/api/proxy/` tree. OTP verify then `signIn('credentials', { phone, otp_token })`, but `authOptions` only reads `email`/`password`. |
| FastAPI OTP | `POST /api/auth/send-otp` and `/verify-otp` in `otp_auth.py`. In-memory store; KaveNegar if `SMS_PROVIDER`/`SMS_API_KEY` are set, otherwise the code is logged. |

Session-backed pages (`/account/subscription`, `/risk/policy`, `/payment/checkout`, admin BFF) still call `useSession` / `getServerSession`. Without a session they prompt “وارد شوید” and send the user to the CTA-only sign-in page.

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
| Blue ticker empty on Compose | `useIranTicker` default `:8000`, Compose API `:8011` | `NEXT_PUBLIC_API_URL=http://localhost:8011` |
| Market tab empty on local `start.py` | `IranMarketOverview` still defaults to `:8011` | `NEXT_PUBLIC_API_URL=http://localhost:8000` or run Compose |
| Market tab looks “live” with no backend | `IranMarketOverview` seed + fallback is mock | Expected; only a 200 with non-empty `items` replaces it |
| Landing `/` white-screens | `page.tsx` maps undeclared `features` | Hero/research/CTA are in the same module; fix or stub `features` |
| Crypto quotes ~10× older screenshots | Nobitex path dropped `/10` Toman conversion | API `price` is Rial; nav ticker labels ریال |
| Cannot log in at `/auth/signin` | Page is a dashboard CTA, not a form | Expected for the demo path. Email login is NextAuth `authorize()` only (no UI). OTP UI is not wired (missing proxy + credential fields). |
| Admin redirect loops to dashboard | Admin layout → sign-in → sample dashboard link | There is no admin login form on `/auth/signin` |
| Frontend README said `:3000` | `npm run dev` binds **3003** | Use 3003 |

---

## Related

- App routes and nav: [APP_FLOW.md](APP_FLOW.md)
- Repo layout and commands: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Wiki setup: [Getting Started](../wiki-content/Getting-Started.md)
- Frontend overview: [frontend-nextjs/README.md](../frontend-nextjs/README.md)
