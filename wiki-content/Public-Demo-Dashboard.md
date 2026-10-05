# Public Demo Dashboard

The UI opens without login. `/dashboard` is a **sample** view; only the Iran-market ticker is a live FastAPI call (TEDPIX is a placeholder).

Full runbook (ports, `start.py`, leftover OTP/admin login): **[docs/PUBLIC_DEMO_DASHBOARD.md](https://github.com/massoudsh/Findash/blob/main/docs/PUBLIC_DEMO_DASHBOARD.md)** in the repo.

## Quick facts

| Item | Value |
|------|--------|
| Frontend | http://localhost:3003 |
| Local API (`start.py` / `make dev`) | http://localhost:8000 |
| Compose API | http://localhost:8011 |
| Sign-in | CTA → `/dashboard`, not a credentials form |
| Admin | Still needs a NextAuth admin session; sign-in no longer collects one |

## Next

- [[Getting Started]] — install, `start.py` from repo root, port mismatch
- [[Frontend]] — App Router, BFF, i18n split
- [[Home]]
