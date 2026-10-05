# App flow – Octopus Trading Platform (Findash)

How the app is structured and how key flows work.

---

## 1. App entry and layout

```mermaid
flowchart TB
    subgraph Entry["App entry"]
        A[Browser: /] --> B[Next.js root layout]
        B --> C[ErrorBoundary + LocaleProvider]
        C --> D[SessionProviderWrapper]
        D --> E[NavigationWrapper]
        E --> F[Page content]
    end
    E --> G[Left sidebar]
    E --> H[Right sidebar]
    E --> I[Header: search, theme, notifications]
```

- **Root:** `app/layout.tsx` wraps all pages with `NavigationWrapper`, session, locale, and toaster.
- **Navigation:** `NavigationWrapper` renders left sidebar, right sidebar, and header; `children` is the current page.

---

## 2. Navigation → pages

```mermaid
flowchart LR
    subgraph Left["Left sidebar – Trading & Portfolio"]
        L1[/dashboard]
        L2[/realtime]
        L3[/options]
        L4[/trades]
        L5[/trading-bots]
        L6[/backtesting]
        L7[/portfolio]
        L8[/strategies]
        L9[/risk]
    end
    subgraph Right["Right sidebar – Analysis & Tools"]
        R1[/technical]
        R2[/fundamental-data]
        R3[/macro]
        R4[/on-chain]
        R5[/social]
        R6[/ai-models]
        R7[/data-explorer]
        R8[/visualization]
        R9[/reports]
        R10[/api-playground]
        R11[/notifications]
        R12[/admin]
    end
```

| Route | Page / content |
|-------|-----------------|
| `/` | Landing (Persian marketing page; CTA to `/dashboard`) |
| `/dashboard` | Public sample dashboard — tabs: overview, portfolio, market, trades, analytics, help. See [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md). |
| `/auth/signin` | CTA to the sample dashboard (no email/password form) |
| `/options` | Options: **Trade** tab (terminal) + **Strategies** tab (options strategy library) |
| `/strategies` | Strategies: list, create, details, mini-charts; “Options Strategies” link |
| `/trades` | Trading center (order entry, open orders) |
| `/trading-bots` | Trading bots list and control |
| `/backtesting` | Backtest config and results |
| `/portfolio` | Portfolio view |
| `/risk` | Risk assessment |
| Others | Technical, Fundamental, Macro, On-chain, Social, AI Models, Data Explorer, Visualization, Reports, API Playground, Notifications, Admin |

---

## 3. Create-strategy flow (end-to-end)

```mermaid
sequenceDiagram
    participant U as User
    participant P as Strategies page
    participant API as Frontend api.ts
    participant B as FastAPI backend

    U->>P: Click "New Strategy"
    P->>P: Open modal (NewStrategyForm)
    U->>P: Fill form, Submit
    P->>API: createStrategy(payload)
    API->>B: POST /strategies/
    B->>B: Append to in-memory list
    B->>API: 200 + created strategy
    API->>P: Resolve with data
    P->>P: Toast "Strategy Created", close modal
    P->>API: getStrategies()
    API->>B: GET /strategies/
    B->>API: List of strategies
    API->>P: response.data
    P->>P: setStrategies(data), re-render list
```

- **Frontend:** `StrategiesContent` → `NewStrategyForm` submit → `createStrategy(strategyData)` from `lib/services/api.ts`.
- **Backend:** `POST /strategies/` handled by `strategies_crud.py`; stores in memory and returns the new strategy.
- **Refresh:** After create, `handleNewStrategySuccess()` calls `getStrategies()` and `setStrategies(response.data)` so the new strategy appears in the list.

---

## 4. Options flow (Trade vs Strategies)

```mermaid
flowchart TB
    O[/options] --> Tabs{Tabs}
    Tabs --> Trade[Trade tab]
    Tabs --> Strat[Strategies tab]
    Trade --> OT[OptionTradingTerminal]
    Strat --> OST[OptionsStrategiesTab]
    OST --> Cards[Strategy cards: Long Call, Iron Condor, etc.]
    Cards --> Deploy[Deploy / Open in Terminal]
    Deploy --> Tabs
```

- **Options page:** Two tabs – **Trade** (terminal) and **Strategies** (options strategy library).
- **Strategies tab:** Renders strategy cards; “Terminal” / “Deploy” can switch to the Trade tab or open the terminal for execution.

---

## 5. Dashboard data flow

```mermaid
flowchart TB
    SignIn["/auth/signin — CTA only"] --> Dash["/dashboard"]
    Landing["/"] --> Dash
    Dash --> Ticker[BlueTickerBar]
    Ticker --> LiveAPI["GET /api/iran-market/ticker — live, TEDPIX placeholder"]
    Dash --> Tabs[tab= overview / portfolio / market / trades / analytics / help]
    Tabs --> Overview[OverviewDashboard — hardcoded cards]
    Tabs --> Port[PortfolioContent — MOCK_PORTFOLIOS]
    Tabs --> Mkt[IranMarketOverview — mock, then overview API]
```

- **Public by design:** no session is required. Header copy: «داشبورد عمومی با داده نمونه».
- **Live:** ticker only (`useIranTicker` → FastAPI `/api/iran-market/ticker`). TEDPIX is not sourced from tgju/Nobitex.
- **Sample:** overview stats/positions/activity, portfolio holdings, and the market-tab seed/fallback.
- Tab query: `?tab=portfolio` etc. `overview` is omitted from the URL.
- Runbook: [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md).

---

## 6. API base URL and backend

- Client ticker/market/signup fall back to **`NEXT_PUBLIC_API_URL` or `http://localhost:8011`**.
- Some BFF helpers (`lib/backend-url.ts`) fall back to **`http://localhost:8000`**.
- Local `python3 start.py` / `make dev` bind FastAPI to **8000**. Docker Compose publishes the API at **8011**.
- Backend app: `src/main_refactored.py`.

---

## 7. Quick reference – where things live

| What | Where |
|------|--------|
| Layout + sidebars | `frontend-nextjs/src/components/navigation/navigation-wrapper.tsx` |
| Dashboard | `frontend-nextjs/src/app/dashboard/page.tsx` + `components/dashboard/overview-dashboard.tsx` |
| Sign-in CTA | `frontend-nextjs/src/app/auth/signin/page.tsx` |
| Strategies list + create | `frontend-nextjs/src/components/strategies/strategies-content.tsx` |
| Strategies API (frontend) | `frontend-nextjs/src/lib/services/api.ts` → `getStrategies`, `createStrategy` |
| Strategies API (backend) | `src/api/endpoints/strategies_crud.py` → GET/POST `/strategies/` |
| Options page | `frontend-nextjs/src/app/options/page.tsx` (tabs: Trade, Strategies) |
| Iran allocation copilot | `components/portfolio/allocation-copilot.tsx` → `POST /api/copilot/allocation-analysis` |
| Investing tools page | `app/investing/page.tsx` (not in sidebar; same-origin fetch has no BFF) |
| App layout | `frontend-nextjs/src/app/layout.tsx` |

---

## 8. Investing tools (allocation + `/investing`)

See **[INVESTING_TOOLS.md](./INVESTING_TOOLS.md)** for endpoints, examples, and pitfalls.

- **Portfolio tab:** `/portfolio` → `/dashboard?tab=portfolio` → `IranPortfolioSection` (localStorage) + allocation copilot (FastAPI via `getBackendUrl()`).
- **`/investing`:** watchlists, screener, paper, dividends. Browser calls `/api/investor-tools/*` on the Next.js origin; those routes are implemented only on FastAPI today.
