# Investing tools — allocation copilot and `/investing`

Operational runbook for the two investor-facing modules that are registered in `src/main_refactored.py` but missing from `docs/api.md` and the wiki API index.

Draft PRs #28–#30 cover account platform, session/BFF, and the public sample dashboard. This page covers **what those PRs do not**: rule-based allocation analysis and the JSON-backed investor-tools API.

---

## Intent

| Module | What it is | What it is not |
|--------|------------|----------------|
| **Allocation copilot** | Educational concentration / diversification readout of holdings the user already entered | Not a buy/sell advisor, price predictor, or live mark-to-market engine |
| **Investor tools** | Watchlists, Iran-market screener, paper ledger, manual dividends, calendar **contract** | Not a broker, payout rail, or verified economic calendar |

Both are trust / literacy layers. The copilot docstring and every analysis response include a Persian disclaimer that the output is **not** investment advice.

---

## Architecture

```
Browser
  │
  ├─ /dashboard?tab=portfolio
  │     IranPortfolioSection (localStorage `iran_portfolio_v1`)
  │        └─ AllocationCopilot ──POST──► FastAPI /api/copilot/allocation-analysis
  │                                         (getBackendUrl(), default :8000)
  │
  └─ /investing
        fetch('/api/investor-tools/…')  ──► Next.js origin (:3003)
              │                           NO App Router handler, NO rewrite
              ✗ 404 unless you hit FastAPI directly
              │
              FastAPI /api/investor-tools/*  ──► data/investor_tools.json
                    screener ──► src/api/endpoints/iran_market.get_overview()
```

Routers are included in `src/main_refactored.py` (no extra prefix on `include_router`; each file sets its own prefix).

---

## 1. Allocation copilot

**Code:** `src/api/endpoints/allocation_copilot.py`  
**UI:** `frontend-nextjs/src/components/portfolio/allocation-copilot.tsx`  
**Holdings source:** `iran-portfolio-section.tsx` + `add-asset-modal.tsx`  
**Tests:** `tests/test_allocation_copilot.py`

### Endpoint

`POST /api/copilot/allocation-analysis` — **no auth**.

Request:

```json
{
  "holdings": [
    { "code": "XAU18", "name": "طلای 18 عیار", "type": "gold", "value": 90000000 },
    { "code": "USD", "name": "دلار", "type": "currency", "value": 10000000 }
  ]
}
```

`type` is a free string on the API; the UI sends one of: `gold` | `silver` | `currency` | `crypto` | `stock` | `bond` | `real_estate` | `cash`. `value` must be `>= 0`. Rows with `value == 0` are dropped before scoring.

Response fields:

| Field | Meaning |
|-------|---------|
| `total_value` | Sum of positive `value`s (no FX conversion) |
| `category_breakdown` | Per-`type` sum, `pct` of total, sorted descending |
| `top_holding_pct` | Largest **single holding** / total (not largest category) |
| `hhi` | Σ (category pct)² on the 0–10000 scale |
| `concentration_level` | `کم` if HHI < 1500; `متوسط` if ≤ 2500; else `بالا` |
| `diversification_score` | `clamp(round(100 - hhi/100), 0, 100)` |
| `insights` | Persian rule-based sentences (empty holdings → prompt to add assets) |
| `disclaimer` | Always present; same text as `DISCLAIMER` in the router |

Example (matches the concentrated unit test):

```bash
curl -s -X POST http://localhost:8000/api/copilot/allocation-analysis \
  -H 'Content-Type: application/json' \
  -d '{"holdings":[{"code":"XAU18","name":"طلا","type":"gold","value":90000000},{"code":"USD","name":"دلار","type":"currency","value":10000000}]}'
```

Four equal categories → HHI `2500` → `concentration_level` is `متوسط` (the `<= 2500` boundary).

Empty or all-zero holdings return `total_value: 0`, `hhi: 0`, `concentration_level: "کم"`, `diversification_score: 0`.

### UI wiring

1. `/portfolio` redirects to `/dashboard?tab=portfolio`.
2. `PortfolioContent` renders `IranPortfolioSection` **above** the mock USD portfolio cards.
3. Assets live in **browser** `localStorage` key `iran_portfolio_v1` — not Postgres.
4. Positions net buy minus sell per `code`; `qty <= 0` is dropped.
5. Copilot `value` is **`totalSpent` (cost basis)**, not a live quote.
6. The card is hidden when there are no positions (`holdings.length === 0`).
7. Fetch uses `getBackendUrl()` (`NEXT_PUBLIC_API_URL` → `BACKEND_URL` → `http://localhost:8000`).

### Constraints

- Mixed `IRT` / `USD` lots are summed as raw numbers. There is no FX step.
- Not mark-to-market. A gold lot entered at purchase cost will not track tgju/Nobitex.
- Unknown `type` strings still enter HHI; Persian labels fall back to the raw type.
- CORS / port: Compose publishes the API on **`:8011`**. A frontend without `NEXT_PUBLIC_API_URL=http://localhost:8011` posts to `:8000` and the card shows `خطا در دریافت تحلیل تخصیص دارایی`.

---

## 2. Investor tools

**Code:** `src/api/endpoints/investor_tools.py`  
**UI:** `frontend-nextjs/src/app/investing/page.tsx`  
**Store:** `data/investor_tools.json` (gitignored `data/`; Compose mounts `./data:/app/data` on `api`)

Prefix: `/api/investor-tools`. Auth is **optional** (`get_optional_user`). Missing/invalid Bearer → `user_id = "default"`. All anonymous clients share one ledger.

### Endpoints

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/watchlists` | List for current user key |
| `POST` | `/watchlists` | Max **5** lists; `name` 1–60 chars; `symbols` ≤ 100, de-duped + sorted. `201` |
| `PUT` | `/watchlists/{id}` | `404` `واچ‌لیست پیدا نشد` |
| `DELETE` | `/watchlists/{id}` | `204` or `404` |
| `GET` | `/screener` | Query: `category`, `min_change`, `max_change`, `query`, `sort` (`change_desc` \| `change_asc` \| `price_desc`) |
| `GET` | `/paper` | Creates `{ cash: 100_000_000, orders: [], journal: [] }` on first read |
| `POST` | `/paper/orders` | Buy fails `422` if `qty*price+fee > cash`. Sell **does not** check inventory; cash increases. `201` |
| `GET` | `/events` | Hard-coded unverified placeholder; `source_status` says a verified provider is required |
| `GET` | `/dividends` | `{ items, summary.received, summary.expected }` |
| `POST` | `/dividends` | `currency` `IRT` \| `IRR`; `status` `received` \| `expected`. `201` |
| `DELETE` | `/dividends/{id}` | `204` or `404` `ثبت سود نقدی پیدا نشد` |

Screener example:

```bash
curl -s 'http://localhost:8000/api/investor-tools/screener?category=gold&sort=change_desc'
```

Screener keeps only `iran_market` overview rows with `available: true` (tgju + Nobitex). TEDPIX is not in that overview list. Cache TTL for overview is **60s** in `iran_market.py`.

Paper order example:

```bash
curl -s -X POST http://localhost:8000/api/investor-tools/paper/orders \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC-IRT","side":"buy","quantity":0.01,"price":5000000000,"thesis":"test"}'
```

### UI wiring and gaps

- Page exists at **`/investing`**. It is **not** in `navigation-wrapper.tsx` and **not** in `app/api/search/route.ts` `PLATFORM_INDEX`.
- Browser calls are **same-origin** (`fetch('/api/investor-tools/screener')`, …). There is no `frontend-nextjs/src/app/api/investor-tools/**` route and `next.config.js` has **no rewrite** to FastAPI. Those calls 404 on `:3003`.
- The page then client-filters screener rows again (`query` / `category`) instead of passing backend query params.
- Calendar tab does **not** call `GET /events`. It only shows a static “no official events until a verified source” alert.
- Paper form does not send `fee`, `stop_loss`, `take_profit`, or `tags` (API accepts them).

### Constraints

- Persistence is a **single JSON file**, not SQLAlchemy / Alembic. Concurrent writes can clobber. Corrupt JSON → empty in-memory default (watchlists/paper/dividends reset).
- Local `python3 start.py` needs a writable `data/` directory at the process cwd (repo root).
- Compose survives restarts via `./data:/app/data`. A container without that mount loses ledgers.
- Shared `"default"` user: do not treat watchlists or paper cash as private without a Bearer token.
- Paper sell can print money (no position check). Mode is always `"paper"` — nothing is sent to a broker or ZarinPal.
- Events are never live market data. Do not document them as a calendar feed.

---

## Troubleshooting

| Symptom | Cause | What to check |
|---------|--------|----------------|
| Copilot card: `خطا در دریافت تحلیل…` | POST never reached FastAPI | `NEXT_PUBLIC_API_URL` (local `:8000`, Compose `:8011`); backend `curl` the endpoint |
| Copilot hidden | No net-long Iran lots | `localStorage.iran_portfolio_v1`; add a buy in **پرتفولیو** |
| `/investing` empty + “دریافت داده‌ها ممکن نشد” | Next.js 404 on `/api/investor-tools/*` | Call FastAPI directly, or add a BFF/rewrite (not present today) |
| Watchlists vanish after restart | `data/` not mounted / not writable | Compose volume; `data/` is gitignored |
| Screener empty | All overview items `available: false` | tgju / Nobitex egress; 60s cache |
| Two browsers share paper cash | Both anonymous → `"default"` | Send `Authorization: Bearer <jwt>` |
| Calendar always empty | UI never fetches `/events` | Expected; even the API is a manual placeholder |

---

## Related

- Iran market + TEDPIX placeholder: `src/api/endpoints/iran_market.py` (covered in draft PR #30)
- Account / wallet / ZarinPal: draft PR #28 (`docs/ACCOUNT_PLATFORM.md` when merged)
- Session / BFF / `getBackendUrl`: draft PR #29 (`docs/FRONTEND_SESSION.md` when merged)
