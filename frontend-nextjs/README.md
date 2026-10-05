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

Without this file, ticker/market/signup fall back to **`:8011`**. Some server BFF helpers (`src/lib/backend-url.ts`) fall back to **`:8000`**. Match the URL to how you started FastAPI. Details: [docs/PUBLIC_DEMO_DASHBOARD.md](../docs/PUBLIC_DEMO_DASHBOARD.md).

## Layout

| Path | Role |
|------|------|
| `src/app/` | App Router pages and `src/app/api/` BFF routes |
| `src/components/` | Feature UI (`dashboard`, `portfolio`, `navigation`, `ui`) |
| `src/lib/` | `auth-options.ts`, `backend-url.ts`, `i18n/`, hooks, services |

Root layout (`app/layout.tsx`): `LocaleProvider` (default **fa**), NextAuth `SessionProviderWrapper`, `NavigationWrapper`, Vazirmatn.

## Public sample dashboard

`/dashboard` does **not** require a session. `/auth/signin` is a link to that page, not a credentials form.

| Surface | Source |
|---------|--------|
| Overview cards, mock portfolio | Hardcoded in the component |
| Market tab | Mock seed; replaced only if `GET /api/iran-market/overview` returns items |
| Blue ticker | `useIranTicker` → `GET /api/iran-market/ticker` (TEDPIX is a backend placeholder) |

`middleware.ts` allow-lists dashboard/trading/settings routes with `authorized: () => true`. Admin is still gated in `app/admin/layout.tsx`.

## API communication

1. **Browser → FastAPI** for Iran market (`NEXT_PUBLIC_API_URL`, default `:8011`).
2. **Browser → Next.js BFF** (`src/app/api/subscriptions`, `risk-policy`, `admin`, `payment`, `reports/portfolio`, …) which call FastAPI. The portfolio PDF handler uses `getServerSession(authOptions)` and streams `application/pdf`. Details: [docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md).
3. **Hardcoded sample data** on the public dashboard/portfolio (not the old `lib/services/*_api.ts` mock layer).

NextAuth `authorize()` still posts email/password to `POST /api/auth/login`. There is no matching form on `/auth/signin`. `/auth/phone` and `/auth/otp` call `/api/proxy/auth/*`, which is not implemented under `src/app/api/`.

## i18n

`src/lib/i18n/translations.ts` + `useTranslations()`: navigation and command palette (`en` / `es` / `fa`). Feature pages (dashboard, trading, fundamental, options, …) use hardcoded Persian. Switching language does not rewrite those pages.

## Commands

```bash
npm run dev    # :3003
npm run build
npm start      # :3003
npm run lint
```
