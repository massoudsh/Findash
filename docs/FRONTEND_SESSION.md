# Frontend session, BFF, dashboard, and navigation

Developer runbook for how the Next.js app (Findash / Octopus) authenticates, talks to FastAPI, and composes `/dashboard`. Verified against `frontend-nextjs` and `src/api/endpoints/professional_auth.py` as of October 2026.

Account-platform APIs (wallet, ZarinPal, KYC, alerts) are FastAPI-side; this page covers the **browser session** that carries the JWT.

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
  ├─► Client fetch  NEXT_PUBLIC_API_URL  (ticker, axios api.ts, wallet/KYC/PDF)
  └─► Next.js BFF   src/app/api/**/route.ts
         getServerSession(authOptions) → Authorization: Bearer {accessToken}
         getBackendUrl() → FastAPI
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

1. `/auth/signin` calls `signIn("credentials", { redirect: false, email, password })`.
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

The sign-in form can fill demo emails (`trader@octopus.trading`, `admin@octopus.trading`); passwords still have to exist in the database.

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
| Wallet, KYC, alerts, PDF | no BFF | call FastAPI with Bearer |

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

`useIranTicker` (dashboard blue bar + nav `NewsTicker`) defaults to `http://localhost:8000`. `getBackendUrl` and `lib/services/api.ts` also default to `:8000`. Still on **`:8011`**: NextAuth `authorize()`, `/auth/signup`, and `IranMarketOverview`.

**Symptom:** market tab empty on local uvicorn, or BFF 503 / ECONNREFUSED on Docker, while `/health` works in the browser. The blue ticker now matches local `start.py` (`:8000`) and **misses Compose** unless `NEXT_PUBLIC_API_URL=http://localhost:8011`.

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

Secondary tabs are `next/dynamic(..., { ssr: false })` so overview does not download portfolio/market/trades/analytics/help until selected. `OverviewDashboard` + `BlueTickerBar` stay in the page bundle.

| `?tab=` | Component | Data |
|---------|-----------|------|
| (default) `overview` | `OverviewDashboard` | **Hardcoded** stats, allocation, positions, activity. Risk gauge uses a local timer, not VaR. |
| `portfolio` | `PortfolioContent` | `IranPortfolioSection` (localStorage) + **Modified Dietz** (typed values only) + in-component `MOCK_PORTFOLIOS` / `MOCK_POSITIONS`. No `getPortfolios()`. |
| `market` | `IranMarketOverview` | Mock seed, then `GET /api/iran-market/overview` |
| `trades` | `TradeTracker` | UI-only |
| `analytics` | `AnalyticsOverview` | UI-only |
| `help` | `HelpCenter` | Static help cards |

`OverviewDashboard` and `BlueTickerBar` live in `frontend-nextjs/src/components/dashboard/overview-dashboard.tsx`. `/demo` reuses `OverviewDashboard` with a banner that the numbers are sample data.

**Live ticker (two surfaces, one hook):**

- Dashboard: `BlueTickerBar` → `useIranTicker` → `GET {NEXT_PUBLIC_API_URL\|\|http://localhost:8000}/api/iran-market/ticker` every 60s. Prints the raw number with **no unit**.
- Global chrome: `NewsTicker` in `navigation-wrapper.tsx` (header + lg sticky). Mixes `/api/news?category=all` (120s) with the same ticker hook. Available quotes are labeled **ریال**, marquee is **75s** (`globals.css` `.animate-news-marquee`).

Backend (`src/api/endpoints/iran_market.py`) ticker symbols: `USD-IRR`, `GOLD18-IRT`, `COIN-IRT`, `BTC-IRT`, `TEDPIX`. TEDPIX is a placeholder (`available: false`, null price). Nobitex crypto `price` is **Rial** (`currency: "IRR"`); the old `/10` Toman conversion is gone. Symbol suffixes (`-IRT` vs `-IRR`) are names, not a second conversion.

`useIranTicker` sets `loading` to `false` in `finally`. A failed or non-OK first fetch no longer leaves the bar on “در حال دریافت”; it keeps `items = []` (or the last good list).

Header buttons “گزارش امروز” / “افزودن دارایی” on `/dashboard` are **not wired**.

The lazy-load refactor dropped the local `type Tab = …` alias while `VALID_TABS: Tab[]` and `useState<Tab>` remain — `next build` typechecks that page as an error until the alias is restored.

Dietz formula, examples, tests: [MODIFIED_DIETZ.md](MODIFIED_DIETZ.md).

---

## 6. Navigation

`NavigationWrapper` (`frontend-nextjs/src/components/navigation/navigation-wrapper.tsx`):

- **lg+:** two collapsible sidebars (left: Trading + research; right: tools, reports, admin, account). Default collapsed (`w-16`).
- **&lt; lg:** header + Radix `Sheet` (side follows RTL) + bottom bar (Dashboard, Command Center, Portfolio, Technical).
- Skip link `#main-content`, `aria-current="page"`, `min-h-11` hit targets, `SheetTitle` / `SheetDescription` for the drawer.

Left items include `/alerts`. Right items include `/account`, `/admin`, `/reports`, `/workflow`. Options, bots, and backtesting are **not** in this sidebar list (Command Center `/trading` is).

---

## 7. Persian PDF vs LLM reports

| Surface | Code | Notes |
|---------|------|--------|
| UI `/reports` | `app/reports/page.tsx` | LLM / FinGPT insights (`docs/llm-report-models.md`). Not the PDF endpoint. |
| `GET /api/reports/portfolio.pdf` | `src/api/endpoints/pdf_reports.py` | JWT required. Own portfolio only; wrong id → **404**. Deterministic insights, no LLM. |
| Preview | `GET /api/reports/portfolio/preview` | Metadata + `download_url`. |

PDF generation (`src/services/pdf_reports.py`) needs Vazirmatn at `/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf` (optional Bold). Missing font → **503**. `docker/Dockerfile.fastapi` does **not** install that font — see `docker/README.md`.

---

## Operator / developer pitfalls

| Symptom | Likely cause |
|---------|----------------|
| Sign-in always “ایمیل یا رمز عبور اشتباه است” | `authorize()` cannot reach FastAPI (port mismatch) or login body is not `{success, user, access_token}`. |
| Admin page shows “دسترسی محدود…” after login | NextAuth `session.user.role` is not the string `admin` (JWT/user.role from FastAPI). |
| Admin BFF 401 with a session cookie | Handler used `getServerSession()` without `authOptions`, so `accessToken` is missing. |
| Ticker shows “—” on **Compose** | `useIranTicker` default is now `:8000`; Compose publishes the API on `:8011`. Set `NEXT_PUBLIC_API_URL`. |
| Market tab empty on local `start.py` | `IranMarketOverview` still defaults to `:8011`. |
| Nav ticker numbers look 10× the old crypto quotes | Nobitex path no longer divides Rial by 10; `NewsTicker` always suffixes ریال. |
| Docker BFF 503 | Server-side fetch used `localhost:8000` inside the frontend container; need `BACKEND_INTERNAL_URL=http://api:8000` or `getBackendUrl()` pointing at the Docker DNS name. |
| PDF 503 | Vazirmatn TTF not on the API image/host. |
| Middleware “restore LOGIN_ENABLED” comment | Flag removed from sign-in; only `authorized: () => true` remains. |
| `next build` errors on `Tab` in `dashboard/page.tsx` | Lazy-load commit removed `type Tab = …` but still uses `Tab`. |
| Landing `/` white-screens | `app/page.tsx` maps undeclared `features`. |
