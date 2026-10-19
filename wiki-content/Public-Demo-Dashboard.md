# Public Demo Dashboard

The UI opens without login. `/dashboard` is a **sample** view for overview/mock portfolio. The Iran-market ticker **and** market tab are live FastAPI calls (rial; TEDPIX is a placeholder).

Full runbook (ports, `start.py`, leftover OTP): **[docs/PUBLIC_DEMO_DASHBOARD.md](https://github.com/massoudsh/Findash/blob/main/docs/PUBLIC_DEMO_DASHBOARD.md)** in the repo. Quote contract: [[Iran Market Pricing]].

## Quick facts

| Item | Value |
|------|--------|
| Frontend | http://localhost:3003 |
| Local API (`start.py` / `make dev`) | http://localhost:8000 |
| Compose API | http://localhost:8011 |
| Sign-in | Credentials form + sample-dashboard link |
| Market tab | Live overview; no mock seed |
| Admin | NextAuth `role === 'admin'`; `callbackUrl=/admin` is kept |

## Next

- [[Getting Started]] — install, `start.py` from repo root, port mismatch
- [[Frontend]] — App Router, BFF, i18n split
- [[Home]]
