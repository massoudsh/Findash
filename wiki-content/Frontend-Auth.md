# Frontend auth and dashboard

How the Next.js app signs in and talks to FastAPI. Full runbook (ports, BFF table, dashboard data, PDF 503): **[docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md)** in the repo.

## Short version

- UI: Next.js 15, **port 3003**.
- Login: Credentials provider → FastAPI `POST /api/auth/login` → JWT stored as NextAuth `session.accessToken`.
- `middleware.ts` currently allows all matched routes (`authorized: () => true`). `/admin` is gated in `app/admin/layout.tsx` (`role === 'admin'`).
- Docker host API is **8011**; local uvicorn is **8000**. `getBackendUrl()` and NextAuth `authorize()` (without `BACKEND_INTERNAL_URL`) both default to `:8000`. Set `NEXT_PUBLIC_API_URL` for Compose (`:8011`) and `BACKEND_INTERNAL_URL=http://api:8000` inside the frontend container.
- `/auth/signin` is a credentials form (`safeCallbackPath` on `callbackUrl`) plus a sample-dashboard link. Checkout pay without a session goes to `/auth/signin?callbackUrl=/payment/checkout`.
- `/dashboard` overview numbers are mostly static; the blue ticker and market tab are live rial quotes. See [[Iran Market Pricing]]. `/reports` is LLM insights; Persian PDF is `GET /api/reports/portfolio.pdf` (Vazirmatn required).

See also [[Frontend]] and [[Configuration]].
