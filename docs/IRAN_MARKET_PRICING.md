# Iran market pricing contract

Public interface for Iran-market quotes after `0aee22b` (2026-10-19). Verified against `src/api/endpoints/iran_market.py`, `tests/test_iran_market_contract.py`, `iran-market-overview.tsx`, `use-iran-ticker.ts`, `news-ticker.tsx`, `overview-dashboard.tsx` (`BlueTickerBar`), and `price_alerts.py`.

Wallet, ZarinPal, and subscriptions stay **toman**. This page is only the quote API and the UIs that print those numbers.

---

## Intent

| This API does | This API does not |
|---------------|-------------------|
| Return tgju + Nobitex quotes with a single unit: **rial (`IRR`)** | Convert gold/coin/crypto to toman because the symbol ends in `-IRT` |
| Mark a symbol `available: false` when a provider fails | Invent sample prices (the old market-tab mock seed is gone) |
| Cache the last successful payload for **60s** | Serve TEDPIX from tgju/Nobitex (ticker placeholder only) |

`-IRT` in a **symbol** is a Findash asset key. `currency` on the **payload** is the unit of `price`. As of this commit every overview row sets `"currency": "IRR"`.

---

## Architecture

```
tgju.org  ──► _tgju_price()     USD/EUR/GBP, gold, coins
Nobitex   ──► _nobitex_stats()  BTC/ETH/USDT/BNB/TRX  (dstCurrency=rls)
                 │
                 ▼
        GET /api/iran-market/overview   (60s in-process cache)
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   /prices   /ticker   price_alerts.evaluate
        │        │
        ▼        ▼
 IranMarketOverview   BlueTickerBar / NewsTicker
 (getBackendUrl)      (useIranTicker → :8000)
```

No auth. Router prefix `/api/iran-market`, registered from `src/main_refactored.py`.

---

## Endpoints

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/iran-market/overview` | Full list. `cached_at` is unix seconds of the **first** fill of this cache window. |
| GET | `/api/iran-market/prices?symbols=` | Filter of overview. Omit `symbols` → same as overview. |
| GET | `/api/iran-market/ticker` | Five slots: `USD-IRR`, `GOLD18-IRT`, `COIN-IRT`, `BTC-IRT`, `TEDPIX`. |
| GET | `/api/iran-market/assets` | Static `AssetsConfig` metadata — not live prices. |

### Overview item shape

```json
{
  "symbol": "BTC-IRT",
  "label": "بیت‌کوین",
  "icon": "₿",
  "category": "crypto",
  "price": 1100000,
  "currency": "IRR",
  "change_pct": 10,
  "up": true,
  "available": true
}
```

Provider miss: `price` / `change_pct` / `up` are `null`, `available` is `false`. The row is still present. Contract test: `tests/test_iran_market_contract.py`.

TEDPIX is **not** in overview. `/ticker` appends `{ available: false, price: null }` and **omits** `currency`.

### Example

```bash
# Local start.py / make dev
curl -s http://localhost:8000/api/iran-market/overview | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print({i["symbol"]: (i["price"], i["currency"], i["available"]) for i in d["items"]})'
```

Compose host port is **8011**.

---

## UI consumers

| Surface | Code | Unit shown |
|---------|------|------------|
| Market tab | `IranMarketOverview` → `GET {getBackendUrl()}/api/iran-market/overview` every 30s | `IRR` → «ریال», `IRT` → «تومان». Today the API only sends `IRR`. |
| Blue bar | `BlueTickerBar` → `useIranTicker` → `/ticker` every 60s | **Number only** — no ریال/تومان suffix |
| Nav marquee | `NewsTicker` | Hardcoded «ریال» on available ticker rows |
| Price alerts | `evaluate_price_alerts` compares `rule.target_price` to overview `price` | Raw rial. A toman target will not match. |

`IranMarketOverview` no longer seeds `MOCK_MARKET_ITEMS`. A 200 with zero `available` items is treated as failure. On error it keeps the last good list (empty on first load) and shows the amber line «ارتباط با بازار برقرار نشد…».

---

## Constraints

- **Do not divide Nobitex `latest` by 10.** The API keeps rial (`tests/test_iran_market_contract.py`).
- **Do not run `formatTomanValue` on quote `price`.** That helper is for the **mock portfolio** cards (`portfolio-content.tsx`): stored amounts are toman; the IRT/IRR toggle multiplies by 10 for display. Quotes are already rial.
- Gold/coin symbols (`GOLD18-IRT`, `COIN-IRT`, …) still have `currency: "IRR"`. The `-IRT` suffix is the asset id, not the unit.
- Wallet deposits and ZarinPal `amount_toman` are still toman; the gateway is `amount * 10` rial. See [ACCOUNT_PLATFORM.md](ACCOUNT_PLATFORM.md).
- Cache is **process memory**. Two API workers do not share it. TTL is 60s even though the file header comment still says “30-second”.
- `IranMarketOverview` and signup use `getBackendUrl()` (`:8000`). `useIranTicker` uses `NEXT_PUBLIC_API_URL \|\| http://localhost:8000`. Compose still publishes the API on **`:8011`** — set `NEXT_PUBLIC_API_URL` or the market tab and ticker miss the host mapping.

---

## Troubleshooting

| Symptom | Cause | What to do |
|---------|--------|------------|
| Market tab empty + amber alert | Overview 4xx/5xx, timeout (15s), or no `available` item | `curl` `/overview`; check tgju/Nobitex egress; do **not** expect mock cards |
| Ticker «—» on local uvicorn | `NEXT_PUBLIC_API_URL` still points at `:8011` | `.env.local` → `http://localhost:8000` |
| Crypto looks 10× too large vs a toman UI | Price is rial by contract | Label ریال or divide only in the display layer |
| Alert never fires on a “toman” target | Evaluator uses raw overview rial | Store `target_price` in rial |
| TEDPIX always unavailable | Placeholder in `/ticker` | Expected |
| Gold card says ریال but symbol is `GOLD18-IRT` | Payload `currency` is `IRR` | Expected after 2026-10-19 |

---

## Related

- Session / ports: [FRONTEND_SESSION.md](FRONTEND_SESSION.md)
- Public dashboard: [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md)
- Alerts + wallet units: [ACCOUNT_PLATFORM.md](ACCOUNT_PLATFORM.md)
- Copilot / screener (screener reads this overview): [INVESTING_TOOLS.md](INVESTING_TOOLS.md)
