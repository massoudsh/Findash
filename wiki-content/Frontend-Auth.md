# Frontend auth and dashboard

How the Next.js app signs in and talks to FastAPI. Full runbook (ports, BFF table, dashboard data, PDF 503): **[docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md)** in the repo.

## Short version

- UI: Next.js 15, **port 3003**.
- Login: Credentials provider → FastAPI `POST /api/auth/login` → JWT stored as NextAuth `session.accessToken`.
- `middleware.ts` currently allows all matched routes (`authorized: () => true`). `/admin` is gated in `app/admin/layout.tsx` (`role === 'admin'`).
- Docker host API is **8011**; local uvicorn is **8000**. `getBackendUrl()` defaults to `:8000`; NextAuth `authorize()` defaults to `:8011`. Set `NEXT_PUBLIC_API_URL` (and Compose `BACKEND_INTERNAL_URL=http://api:8000`).
- `/dashboard` overview numbers are mostly static; the blue ticker is live (`GET /api/iran-market/ticker`). `/reports` is LLM insights; Persian PDF is `GET /api/reports/portfolio.pdf` (Vazirmatn required).

See also [[Frontend]] and [[Configuration]].
