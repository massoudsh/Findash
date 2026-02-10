# Option Trading Page — Implementation Flow

This document describes how to build the **Option Trading** feature so it matches the reference UI: ETH/USDC funding rates, implied APR chart, order book (short/long rates), and order entry panel.

---

## 1. Scope & Placement

| Item | Decision |
|------|----------|
| **Route** | `/options` — this is the main Option Trading feature. |
| **Navigation** | Existing "Options" in Trading sidebar → points to `/options`. |
| **Layout** | Full-width terminal: top bar → stats row → main area (chart | orderbook | order entry). |

---

## 2. Page Structure (Top → Bottom)

Build the page in this order so layout and data flow stay clear.

### Step 1: Top asset bar
- **Left:** Pair selector (e.g. ETHUSDC) with dropdown, star, info.
- **Right:** Maturity text, e.g. `25 days (Matures 27 Feb 2026)` with dropdown.
- **Data:** Symbol and maturity can be state; later from API (e.g. `/api/market-data` or options metadata).

### Step 2: Key metrics row
- One row, horizontal: **Implied APR**, **Mark APR**, **Underlying APR**, **Notional OI**, **24h Volume**, **Next Settlement** (countdown).
- **Data source:**  
  - Funding / APR: `GET /api/comprehensive/funding-rate/{symbol}/refresh` and `/funding-rate/{symbol}/analysis`.  
  - Map: `funding_rate` / `funding_rate_annualized` → Implied APR; add mock or separate endpoint for Mark APR, Underlying APR, OI, Volume, next funding time.
- **Updates:** Poll every 30–60s or use WebSocket later.

### Step 3: Main content (3 columns)
- **Left (≈50%):** Chart panel.  
- **Center (≈25%):** Order book.  
- **Right (≈25%):** Order entry.

### Step 4: Left panel — Chart
- **Tabs:** "APR Chart" (default), "My PnL".
- **APR Chart:** Timeframe buttons (5m, 1H, 1D, 1W). Small legend: Implied APR (OHLC-style), Underlying APR.
- **Chart:** Line chart: Implied APR and Underlying APR over time. Y-axis: percentage.
- **Data:** Historical funding/APR from `GET /api/comprehensive/funding-rate/{symbol}/analysis` (use history if exposed); otherwise mock array by timeframe.
- **My PnL:** Placeholder or link to portfolio/PnL view.

### Step 5: Center panel — Order book
- **Tabs:** "Orderbook", "Market Trades".
- **Orderbook:**  
  - **Short rate (top):** Table with Implied APR (%) and Size (ETH YU); rows sorted by rate; optional depth selector (e.g. 0.1%).  
  - **Spread:** e.g. "0.1% Spread", "Incent. Range: 5.16% - 5.62%", optional "Incentivized Range" checkbox.  
  - **Long rate (bottom):** Same columns.
- **Data:** Mock arrays for short/long rates and sizes; later replace with `GET /api/...` order book or funding-depth endpoint.

### Step 6: Right panel — Order entry
- **Banner:** "Maker Order Rewards Live!" (static or feature flag).
- **Row 1:** Position mode — "Cross", "2x", "One-way" (state only for now).
- **Row 2:** Order type — "Market", "Limit".
- **Row 3:** Direction — "Long Rates" (Pay Fixed, Rcv. Underlying) / "Short Rates" (Pay Underlying, Rcv. Fixed); green/red.
- **Account:** My Notional Size (0 YU), Available to Trade (0 ETH).
- **Input:** Notional Size (number + YU), optional % slider.
- **Options:** "Reduce Only" checkbox.
- **Summary:** Liquidation Implied APR, Margin Required, Fees, Slippage (Est / Max). Use "NA" and "0" until backend is ready.

### Step 7: Footer
- **Left:** Status (e.g. "Online"), Gas (e.g. "$0.02").  
- **Right:** Docs, Support, Terms, Policy, Help & Support (links).

---

## 3. Data Flow Summary

```
User selects symbol (e.g. ETHUSDC)
    → GET /api/comprehensive/funding-rate/{symbol}/refresh   (current rate, next funding time)
    → GET /api/comprehensive/funding-rate/{symbol}/analysis (history, analysis)
    → Optional: GET /api/market-data/{symbol}               (price, volume, OI if available)

Metrics row + chart
    ← funding_rate_annualized, next_funding_time, historical series

Order book
    ← Mock or GET /api/.../orderbook or /funding-depth (when implemented)

Order entry (submit)
    → POST /api/.../order (when implemented); payload: symbol, side long/short, notional, type, etc.
```

---

## 4. File Structure

- **Page:** `frontend-nextjs/src/app/options/page.tsx` — layout wrapper; can keep a simple header + single main component.
- **Feature component:** `frontend-nextjs/src/components/options/option-trading-terminal.tsx` — entire terminal (asset bar, metrics, 3-column main, footer).
- **Subcomponents (optional):**  
  - `option-trading-chart.tsx` — APR chart + tabs + timeframes.  
  - `option-trading-orderbook.tsx` — Short/long rate tables + spread.  
  - `option-trading-order-entry.tsx` — Right panel form.

Use existing UI: `Card`, `Button`, `Input`, `Tabs`, `Slider`, `Checkbox`/`Switch`, `Label`.

---

## 5. Implementation Order

1. **Layout only:** Terminal component with placeholder text for each block (asset bar, metrics, chart, order book, order entry, footer). No API calls.
2. **Metrics row:** Wire symbol state → `fetch(/api/comprehensive/funding-rate/{symbol}/refresh)` and display; add countdown from `next_funding_time`.
3. **Chart:** Mock time series for 5m/1H/1D/1W; then replace with analysis/history from API if available.
4. **Order book:** Static mock data; later swap to API.
5. **Order entry:** Local state only; "Place Order" can toast or disable until backend exists.
6. **Polish:** Dark theme, typography, spacing, and status/gas footer to match reference image.

---

## 6. Backend Hooks (Existing)

- `GET /api/comprehensive/funding-rate/{symbol}` — may return cached or "fetching".
- `GET /api/comprehensive/funding-rate/{symbol}/refresh` — current rate, next funding time, annualized.
- `GET /api/comprehensive/funding-rate/{symbol}/analysis` — current + historical analysis.

Use these for Implied APR, Next Settlement countdown, and (if structure allows) chart history. Add Mark APR, Underlying APR, OI, 24h Volume via mock or new endpoints as needed.

---

## 7. Acceptance Checklist

- [x] Top bar: pair selector + maturity.
- [x] One row: Implied APR, Mark APR, Underlying APR, Notional OI, 24h Volume, Next Settlement.
- [x] Left: APR Chart (with timeframes) + My PnL tab.
- [x] Center: Orderbook (short rate, spread, long rate) + Market Trades tab.
- [x] Right: Order entry (rewards banner, Cross/2x/One-way, Market/Limit, Long/Short Rates, notional, summary).
- [x] Footer: Online, Gas, links.
- [x] Funding rate API used for symbol-driven metrics (GET `/api/funding-rate/{symbol}/refresh`).
- [x] Option trading entry point: **Options** in nav → `/options` → this terminal.

## 8. Implemented Files

- **Page:** `frontend-nextjs/src/app/options/page.tsx` — renders `OptionTradingTerminal`.
- **Terminal:** `frontend-nextjs/src/components/options/option-trading-terminal.tsx` — full layout, metrics from funding API, mock order book, countdown, chart, order entry.

Once order book and order submission backends exist, swap mock data for API calls and add POST order endpoint.
