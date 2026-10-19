# Frontend (Next.js)

Persian RTL UI for Findash. **Next.js 15**, App Router, Tailwind, Shadcn/Radix.

Dev server: **http://localhost:3003** (`npm run dev` / `npm start` both pass `-p 3003`).

## Getting started

```bash
cd frontend-nextjs
npm install
npm run dev
```

Optional `frontend-nextjs/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000   # local start.py / make dev
# NEXT_PUBLIC_API_URL=http://localhost:8011 # docker-compose-core.yml
```

Without this file, ticker, market tab, signup, and NextAuth fall back to **`:8000`**. Compose publishes the API on **`:8011`** — set `NEXT_PUBLIC_API_URL` to match. Details: [docs/PUBLIC_DEMO_DASHBOARD.md](../docs/PUBLIC_DEMO_DASHBOARD.md), [docs/IRAN_MARKET_PRICING.md](../docs/IRAN_MARKET_PRICING.md).

## Layout

| Path | Role |
|------|------|
| `src/app/` | App Router pages and `src/app/api/` BFF routes |
| `src/components/` | Feature UI (`dashboard`, `portfolio`, `navigation`, `ui`) |
| `src/lib/` | `auth-options.ts`, `backend-url.ts`, `i18n/`, hooks, services |

Root layout (`app/layout.tsx`): `LocaleProvider` (default **fa**), NextAuth `SessionProviderWrapper`, `NavigationWrapper`, Vazirmatn.

## Public sample dashboard

`/dashboard` does **not** require a session. `/auth/signin` is an email/password form (NextAuth) plus a «مشاهده داشبورد نمونه» link.

| Surface | Source |
|---------|--------|
| Overview cards, mock portfolio | Hardcoded in the component |
| Iran holdings | `localStorage` `iran_portfolio_v1`; `?addAsset=1` opens the modal |
| Market tab | Live `GET /api/iran-market/overview` only (rial). No mock seed |
| Blue ticker | `useIranTicker` → `GET /api/iran-market/ticker` (TEDPIX placeholder; no unit suffix) |

`middleware.ts` allow-lists dashboard/trading/settings routes with `authorized: () => true`. Admin is still gated in `app/admin/layout.tsx`.

## API communication

1. **Browser → FastAPI** for Iran market (`NEXT_PUBLIC_API_URL`, default `:8000`).
2. **Browser → Next.js BFF** (`src/app/api/subscriptions`, `risk-policy`, `admin`, `payment`, …) which call FastAPI with `getServerSession`.
3. **Hardcoded sample data** on the public dashboard overview / mock portfolio cards.

NextAuth `authorize()` posts email/password to `POST /api/auth/login` (`BACKEND_INTERNAL_URL` or `getBackendUrl()`). `/auth/phone` and `/auth/otp` still call `/api/proxy/auth/*`, which is not implemented under `src/app/api/`.

## i18n

`src/lib/i18n/translations.ts` + `useTranslations()`: navigation and command palette (`en` / `es` / `fa`). Feature pages (dashboard, trading, fundamental, options, …) use hardcoded Persian. Switching language does not rewrite those pages.

## Commands

```bash
npm run dev    # :3003
npm run build
npm start      # :3003
npm run lint
```
