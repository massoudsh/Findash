# Investing Tools

Rule-based **allocation copilot** and the **investor-tools** API (watchlists, screener, paper ledger, dividends).

Full runbook with examples, constraints, and troubleshooting: [docs/INVESTING_TOOLS.md](https://github.com/massoudsh/Findash/blob/main/docs/INVESTING_TOOLS.md).

## Start here

| If you want to… | Go to |
|-----------------|--------|
| Score concentration of local Iran holdings | `POST /api/copilot/allocation-analysis` — UI on `/dashboard?tab=portfolio` |
| Watchlists / paper / dividends | FastAPI `/api/investor-tools/*` — UI page `/investing` (not in the sidebar) |
| Live Iran quotes used by the screener | `GET /api/iran-market/overview` |

## Do not assume

- Copilot is **not** advice and is **not** mark-to-market (cost basis from `localStorage`).
- `/investing` `fetch('/api/investor-tools/…')` hits Next.js `:3003` — there is **no** BFF or rewrite. Call FastAPI (`:8000` local, `:8011` Compose) until a proxy exists.
- Anonymous users share JSON key `"default"` in `data/investor_tools.json`.
- Calendar UI does not call `GET /events`; that endpoint is a manual placeholder.

## Next

- [[Frontend]] — pages and BFF pattern
- [[API Reference]] — other registered routers
- [[Database]] — SQLAlchemy tables (investor-tools is **file-backed**, not these tables)
