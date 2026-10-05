# Frontend session, BFF, dashboard, and navigation

Developer runbook for how the Next.js app (Findash / Octopus) authenticates, talks to FastAPI, and composes `/dashboard`. Verified against `frontend-nextjs` (including `app/api/reports/portfolio/route.ts`) and `src/api/endpoints/professional_auth.py` / `pdf_reports.py` as of October 2026.

Account-platform APIs (wallet, ZarinPal, KYC, alerts) are FastAPI-side; this page covers the **browser session** that carries the JWT. Portfolio PDF is the exception: `/reports` uses a session BFF.

---

## Intent

The UI is a Next.js 15 App Router app (`frontend-nextjs/`, **port 3003**). Login is **Credentials → FastAPI** `/api/auth/login`. The FastAPI JWT is stored on the NextAuth JWT (`session.accessToken`) and forwarded by some Route Handlers. Most pages are **not** login-gated today; FastAPI still requires Bearer tokens on protected endpoints.

---

## Architecture

```
Browser
  │  signIn("credentials")
  ▼
NextAuth  GET/POST /api/auth/[...nextauth]
  │  authorize() → POST {BACKEND_INTERNAL_URL|NEXT_PUBLIC_API_URL}/api/auth/login
  ▼
FastAPI  src/api/endpoints/professional_auth.py
  │  returns { success, user: { id, email, name, role }, access_token }
  ▼
NextAuth JWT callback copies user.id, user.role, accessToken
  │
  ├─► Client fetch  NEXT_PUBLIC_API_URL  (ticker, axios api.ts, wallet/KYC)
  └─► Next.js BFF   src/app/api/**/route.ts
         getServerSession(authOptions) → Authorization: Bearer {accessToken}
         getBackendUrl() → FastAPI
         (includes GET /api/reports/portfolio → /api/reports/portfolio.pdf)
```

**Auth files**

| Piece | Path |
|-------|------|
| NextAuth options | `frontend-nextjs/src/lib/auth-options.ts` |
| App Router handler | `frontend-nextjs/src/app/api/auth/[...nextauth]/route.ts` |
| Session types | `frontend-nextjs/src/types/next-auth.d.ts` |
| Sign-in UI | `frontend-nextjs/src/app/auth/signin/page.tsx` |
| Middleware | `frontend-nextjs/src/middleware.ts` |
| Backend URL helper | `frontend-nextjs/src/lib/backend-url.ts` |
| FastAPI login | `src/api/endpoints/professional_auth.py` |
| JWT decode / `require_admin` | `src/core/security.py` |

---

## 1. Sign-in flow

> **Current UI:** `/auth/signin` is a CTA to `/dashboard` (no email/password form). The Credentials `authorize()` path below still exists in `auth-options.ts` and is what the PDF BFF needs for `session.accessToken`. See [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md).

1. When a credentials sign-in runs, NextAuth calls `signIn("credentials", { redirect: false, email, password })`.
2. `authorize()` POSTs JSON `{ email, password }` to FastAPI `/api/auth/login`.
3. Login must return `success === true`, a `user` object, and `access_token`. Inactive users get `success=false` without a 401 — NextAuth treats that as a failed sign-in.
4. On success the page does a **full navigation** (`window.location.href`) to `callbackUrl` if it is same-origin; otherwise `/dashboard`. Open-redirects to other hosts are rejected (`safeRedirect`).
5. FastAPI JWT claims include singular `"role"` (e.g. `"admin"`). `verify_token` copies that into `TokenData.roles` so `require_admin` (`"admin" in roles`) works.

### Example — login then call a gated API

```bash
# Local uvicorn default is :8000. Docker host port is :8011.
API=http://localhost:8011

TOKEN=$(curl -s -X POST "$API/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@octopus.trading","password":"SecureAdmin2025!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))")

curl -s -H "Authorization: Bearer $TOKEN" "$API/api/admin/users"
```

Demo emails (`trader@octopus.trading`, `admin@octopus.trading`) still work **only** if those users exist in the database **and** something still calls `signIn("credentials")`. The current `/auth/signin` page does not.

---

## 2. Route protection (what is actually gated)

`middleware.ts` wraps `/dashboard`, `/portfolio`, `/trading`, `/analytics`, `/settings` with NextAuth `withAuth`, but:

```ts
authorized: () => true
```

Every matched route is allowed **without a session**. The comment in that file still mentions restoring `({ token }) => !!token` and a `LOGIN_ENABLED` flag; **`LOGIN_ENABLED` is not in `signin/page.tsx` anymore**.

What *is* gated:

| Layer | Behavior |
|-------|----------|
| `app/admin/layout.tsx` | Server: no session → redirect `/auth/signin?callbackUrl=/admin`. `session.user.role !== 'admin'` → Persian alert, no children. |
| Admin BFF (`app/api/admin/*`) | 401 if no session; 403 if `session.user.role !== 'admin'`. |
| Risk-policy / subscriptions BFF | 401 if no session (no extra role check). |
| Portfolio PDF BFF (`GET /api/reports/portfolio`) | 401 if `session.accessToken` is missing. |
| FastAPI `require_admin` | 403 unless JWT `roles` contains `"admin"`. |
| FastAPI `get_current_active_user` | 401 without a valid Bearer token (missing header is 401, not 403 — `Bearer401`). |

`/admin` is **not** in the middleware matcher. Closing the middleware callback does **not** by itself protect the admin UI; the layout does.

---

## 3. BFF vs direct FastAPI

`getBackendUrl()` (`lib/backend-url.ts`) is used by Route Handlers:

1. `NEXT_PUBLIC_API_URL`
2. `BACKEND_URL`
3. fallback `http://localhost:8000`

NextAuth `authorize()` uses a **different** chain:

1. `BACKEND_INTERNAL_URL` (Docker Compose sets `http://api:8000`)
2. `NEXT_PUBLIC_API_URL`
3. fallback `http://localhost:8011`

| Next.js route | Forwards JWT? | `getServerSession(authOptions)`? |
|---------------|---------------|----------------------------------|
| `/api/subscriptions/{plans,me,subscribe}` | me/subscribe yes; plans is public | yes (me/subscribe) |
| `/api/risk-policy/{me,breaches}` | yes | yes |
| `/api/admin/{users,users/[id],audit-log}` | yes + role check | yes |
| `/api/payment/zarinpal/create` | yes if token present | **no** — `getServerSession()` with no options |
| `/api/reports/portfolio` | yes | **yes** — `getServerSession(authOptions)` |
| Wallet, KYC, alerts | no BFF | call FastAPI with Bearer |

Older BFF files (`llm/*`, `funding/*`, `portfolio/*`, …) still default `BACKEND_URL` to `:8000` and often do **not** attach the session JWT.

### Pitfall — payment create BFF

`app/api/payment/zarinpal/create/route.ts` calls `getServerSession()` **without** `authOptions`. In this App Router setup the typed `accessToken` is populated only when `authOptions` is passed (see admin/subscriptions routes). A session cookie can exist while `accessToken` is missing → FastAPI 401 on create.

Always pass `authOptions` in new BFF handlers:

```ts
const session = await getServerSession(authOptions);
```

---

## 4. Port and URL matrix

| How you run | Browser → API | Next.js server → API |
|-------------|----------------|----------------------|
| `docker compose -f docker-compose-core.yml` | `http://localhost:8011` (`NEXT_PUBLIC_API_URL`) | `BACKEND_INTERNAL_URL=http://api:8000` |
| `python3 start.py` / uvicorn | `http://localhost:8000` (`API_PORT` default) | same host, `:8000` |
| Frontend `npm run dev` | port **3003** (`package.json` scripts) | — |

`useIranTicker` defaults to `http://localhost:8011`. `getBackendUrl` and `lib/services/api.ts` default to `http://localhost:8000`. `use-backend-health` also defaults to `:8000`.

**Symptom:** ticker empty on local uvicorn, or BFF 503 / ECONNREFUSED on Docker, while `/health` works in the browser.

**Fix:** set both in `frontend-nextjs/.env.local` (and rebuild Docker frontend if `NEXT_PUBLIC_*` is baked at build time):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8011   # Docker host mapping
BACKEND_INTERNAL_URL=http://api:8000        # only inside Compose
# local uvicorn instead:
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

Compose already sets those on the `frontend` service. `config/env.example` still shows `:8000` for `NEXT_PUBLIC_API_URL`.

---

## 5. Dashboard composition

**Route:** `frontend-nextjs/src/app/dashboard/page.tsx` (client). Tabs are URL state: `?tab=` via `useSearchParams` and `router.replace` (not `nuqs`).

| `?tab=` | Component | Data |
|---------|-----------|------|
| (default) `overview` | `OverviewDashboard` | **Hardcoded** stats, allocation, positions, activity. Risk gauge uses a local timer, not VaR. |
| `portfolio` | `PortfolioContent` | Portfolio API / fallback |
| `market` | `IranMarketOverview` | Iran market |
| `trades` | `TradeTracker` | Trades |
| `analytics` | `AnalyticsOverview` | Analytics |
| `help` | `HelpCenter` | Static help cards |

`OverviewDashboard` and `BlueTickerBar` live in `frontend-nextjs/src/components/dashboard/overview-dashboard.tsx` so the **page module stays small** (Next.js build; see commit that extracted it from `page.tsx`). `/demo` reuses `OverviewDashboard` with a banner that the numbers are sample data.

**Live ticker:** `BlueTickerBar` → `useIranTicker` → `GET /api/iran-market/ticker` every 60s. Backend (`src/api/endpoints/iran_market.py`) returns five symbols: `USD-IRR`, `GOLD18-IRT`, `COIN-IRT`, `BTC-IRT`, `TEDPIX`. TEDPIX is a placeholder (`available: false`, null price) when not in the overview feed.

Header buttons “گزارش امروز” / “افزودن دارایی” on `/dashboard` are **not wired**.

---

## 6. Navigation

`NavigationWrapper` (`frontend-nextjs/src/components/navigation/navigation-wrapper.tsx`):

- **lg+:** two collapsible sidebars (left: Trading + research; right: tools, reports, admin, account). Default collapsed (`w-16`).
- **&lt; lg:** header + Radix `Sheet` (side follows RTL) + bottom bar (Dashboard, Command Center, Portfolio, Technical).
- Skip link `#main-content`, `aria-current="page"`, `min-h-11` hit targets, `SheetTitle` / `SheetDescription` for the drawer.

Left items include `/alerts`. Right items include `/account`, `/admin`, `/reports`, `/workflow`. Options, bots, and backtesting are **not** in this sidebar list (Command Center `/trading` is).

---

## 7. Persian PDF vs LLM reports

`/reports` hosts **two independent** download paths. The Next.js BFF is `GET /api/reports/portfolio` (no `.pdf` suffix). FastAPI is `GET /api/reports/portfolio.pdf`. Mixing the two URLs is the usual 404.

| Surface | Code | Auth | What it actually does |
|---------|------|------|------------------------|
| Page `/reports` | `app/reports/page.tsx` + `reports-content.tsx` | Page is public | LLM insights UI plus a PDF button |
| Button «PDF پرتفوی» | `downloadPortfolioPdf()` | NextAuth cookie | Same-origin `GET /api/reports/portfolio`; blob save as `portfolio-report.pdf` |
| Next.js BFF | `app/api/reports/portfolio/route.ts` | `getServerSession(authOptions)` | Forwards `Authorization: Bearer {accessToken}` to FastAPI |
| FastAPI PDF | `src/api/endpoints/pdf_reports.py` | JWT `get_current_active_user` | Own portfolio only; wrong/missing id → **404** (not 403) |
| Preview | `GET /api/reports/portfolio/preview` | JWT | Metadata + `download_url`. **The UI never calls this.** |
| Button «دانلود» | `downloadReport()` | none | Markdown of the last LLM insight text — **not** a PDF |
| LLM generate | `app/api/llm/reports/*` | **no JWT** | Proxies `/llm/reports/*` (`docs/llm-report-models.md`) |

```
Browser  GET /api/reports/portfolio          (:3003, session cookie)
   │
   ▼
Next.js  getServerSession(authOptions)
   │     no accessToken → 401 JSON
   │     optional ?portfolio_id= digits only
   ▼
FastAPI  GET {getBackendUrl()}/api/reports/portfolio.pdf
         Authorization: Bearer {accessToken}
```

### Constraints (verified)

- The UI **does not** send `portfolio_id`. FastAPI then picks the caller’s latest **active** portfolio (`is_active`, newest `created_at`).
- The BFF **does not** forward `include_trades`. FastAPI defaults it to `true` and includes the last 20 trades.
- FastAPI `Content-Disposition` is `inline`; the BFF rewrites it to `attachment; filename="portfolio-report.pdf"`. The button also sets `anchor.download`.
- Insights in the PDF are deterministic (concentration ≥ 30%, return vs `initial_cash`). No LLM.
- `/auth/signin` is a CTA to `/dashboard`, not a credentials form ([PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md)). Visitors clicking «PDF پرتفوی» get **401** (`برای دریافت گزارش PDF باید وارد حساب خود شوید.`).
- BFF uses `getBackendUrl()` (fallback `:8000`). Compose publishes the API on **:8011**. Unreachable API → BFF **503** (`سرور گزارش در دسترس نیست`).
- Missing Vazirmatn → FastAPI **503**; the BFF forwards that JSON. Font paths and image gap: `docker/README.md`.

### Example — FastAPI PDF (bypasses the BFF)

```bash
# Local uvicorn is :8000. Docker host port is :8011.
API=http://localhost:8011

TOKEN=$(curl -s -X POST "$API/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@octopus.trading","password":"SecureAdmin2025!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))")

curl -sS -H "Authorization: Bearer $TOKEN" \
  -o portfolio-report.pdf \
  "$API/api/reports/portfolio.pdf"
```

Omit `portfolio_id` unless you own that integer id. The browser button cannot be exercised with curl unless you also have a NextAuth session cookie.

---

## Operator / developer pitfalls

| Symptom | Likely cause |
|---------|----------------|
| Sign-in always “ایمیل یا رمز عبور اشتباه است” | `authorize()` cannot reach FastAPI (port mismatch) or login body is not `{success, user, access_token}`. |
| Admin page shows “دسترسی محدود…” after login | NextAuth `session.user.role` is not the string `admin` (JWT/user.role from FastAPI). |
| Admin BFF 401 with a session cookie | Handler used `getServerSession()` without `authOptions`, so `accessToken` is missing. |
| Ticker shows “—” on local API | `useIranTicker` default `:8011` while uvicorn is `:8000`. |
| Docker BFF 503 | Server-side fetch used `localhost:8000` inside the frontend container; need `BACKEND_INTERNAL_URL=http://api:8000` or `getBackendUrl()` pointing at the Docker DNS name. |
| PDF button toast «باید وارد حساب خود شوید» | No NextAuth `accessToken`. `/auth/signin` is a dashboard CTA, not a login form. |
| PDF 404 JSON from the button | JWT user has no active `Portfolio` row (sample dashboard holdings are mock / localStorage, not this table). |
| PDF 503 from the button, API `/health` works | BFF called `:8000` while Compose API is `:8011`, or Vazirmatn TTF is missing on the API host. |
| FastAPI PDF 503 | Vazirmatn TTF not on the API image/host (`docker/README.md`). |
| Middleware “restore LOGIN_ENABLED” comment | Flag removed from sign-in; only `authorized: () => true` remains. |
