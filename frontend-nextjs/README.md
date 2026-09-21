# Frontend (Next.js)

Next.js 15 App Router UI for Findash / Octopus. Port **3003** (`npm run dev` / `npm start`). Session, BFF, dashboard, and navigation: **[docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md)**.

## Getting started

```bash
cd frontend-nextjs
npm install
# optional: frontend-nextjs/.env.local
# NEXT_PUBLIC_API_URL=http://localhost:8000   # local uvicorn
# NEXT_PUBLIC_API_URL=http://localhost:8011   # docker-compose-core.yml host mapping
npm run dev
```

App: [http://localhost:3003](http://localhost:3003). Sign-in: `/auth/signin` (Credentials → FastAPI `POST /api/auth/login`).

## Project structure

- `src/app` — App Router pages and Route Handlers (`src/app/api/`).
- `src/components` — feature UI (`dashboard/`, `navigation/`, `portfolio/`, `ui/`).
- `src/lib` — `auth-options.ts`, `backend-url.ts`, `services/api.ts`, hooks (`use-iran-ticker.ts`).

Pages stay thin: route `page.tsx` composes a content component, usually behind `Suspense`.

## How the UI talks to FastAPI

Two paths (not “mocks first, proxies second”):

1. **Browser → FastAPI** using `NEXT_PUBLIC_API_URL` (axios in `lib/services/api.ts`, Iran ticker, wallet/KYC/PDF).
2. **Browser → Next.js BFF → FastAPI** for subscriptions, risk-policy, admin, and ZarinPal create. Those handlers attach `session.accessToken` when `getServerSession(authOptions)` is used.

`middleware.ts` currently authorizes all matched routes (`authorized: () => true`). `/admin` is gated in `app/admin/layout.tsx` by session + `role === 'admin'`.

Default URL **fallbacks differ** (`getBackendUrl` → `:8000`, NextAuth `authorize` → `:8011`). Set `NEXT_PUBLIC_API_URL` (and in Docker, `BACKEND_INTERNAL_URL=http://api:8000`) or login/ticker/BFF will disagree. Full matrix: [docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md).

## Dashboard

`/dashboard` tabs: overview (mostly static `OverviewDashboard`), portfolio, market, trades, analytics, help. Live strip: `BlueTickerBar` → `GET /api/iran-market/ticker`. `/demo` reuses the overview with a sample-data banner.

## Scripts

```bash
npm run dev    # next dev -p 3003
npm run build
npm start      # next start -p 3003
npm run lint
```
