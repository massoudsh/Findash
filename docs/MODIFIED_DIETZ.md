# Modified Dietz return calculator

Client-only period return on `/dashboard?tab=portfolio`. Verified against `frontend-nextjs/src/lib/modified-dietz.ts`, `modified-dietz-calculator.tsx`, `portfolio-content.tsx`, and `modified-dietz.test.cjs` (2026-10-12).

This is **not** the allocation copilot (`docs/INVESTING_TOOLS.md`) and **not** FastAPI portfolio PnL.

---

## Intent

| It is | It is not |
|-------|-----------|
| A cash-flow-weighted **Modified Dietz** period return from numbers the user types | Not mark-to-market, not annualized, not TWR |
| Independent of wallet, Postgres, ZarinPal, and `localStorage` holdings | Not wired to `IranPortfolioSection` or the mock portfolio cards below it |
| Educational / spreadsheet-style | Not investment advice; values vanish on navigation |

---

## Architecture

```
/dashboard?tab=portfolio
  PortfolioContent
    ├─ IranPortfolioSection     localStorage iran_portfolio_v1  → copilot API
    ├─ ModifiedDietzCalculator  form state only  → modifiedDietz()
    └─ MOCK_PORTFOLIOS / MOCK_POSITIONS   display only
```

No FastAPI route. No Next.js BFF. `modifiedDietz()` runs in the browser.

| Piece | Path |
|-------|------|
| Formula | `frontend-nextjs/src/lib/modified-dietz.ts` |
| UI | `frontend-nextjs/src/components/portfolio/modified-dietz-calculator.tsx` |
| Mount | `portfolio-content.tsx` (directly under `IranPortfolioSection`) |
| Tests | `frontend-nextjs/src/lib/modified-dietz.test.cjs` (`node:test` + TypeScript transpile) |

---

## Formula

```
gain            = V₁ − V₀ − Σ CF
weightedCapital = V₀ + Σ (CF × w)
w               = (end − flowTime) / (end − start)
R               = gain / weightedCapital          (null if weightedCapital ≤ 0)
```

- **Positive CF** = external contribution (deposit). **Negative CF** = external withdrawal.
- Internal buys/sells are **not** external flows (UI copy).
- Weight at period **start** is **1**; at period **end** is **0**.
- UI dates are HTML `type="date"` (`YYYY-MM-DD`). `Date.parse` treats that form as UTC midnight, which matches the on-screen note that dates are start-of-day.

`returnPercent` is `R * 100`, or `null` — **not** `0` — when weighted capital is non-positive.

---

## Library contract

```ts
modifiedDietz(beginningValue, endingValue, start, end, cashFlows)
→ { gain, weightedCapital, returnPercent }
```

`cashFlows`: `{ amount: number; timestamp: number }[]` (epoch ms, same units as `start` / `end`).

Throws (Persian messages) when:

- Any of `beginningValue`, `endingValue`, `start`, `end` is not finite
- `end <= start`
- Beginning or ending value is **negative** (`0` is allowed)
- A flow amount/timestamp is not finite, or the timestamp is **outside** `[start, end]`
- `gain` / `weightedCapital` / `returnPercent` overflow to non-finite

UI extra: if any added flow row is missing amount or date, it throws `مبلغ و تاریخ همهٔ جریان‌های نقدی را تکمیل کنید.` before calling the library. Calculation starts only once beginning, ending, start, and end are all filled.

---

## Worked examples (from unit tests)

Period in the tests is `start=0`, `end=100`. Mid-period flow uses `timestamp=50` → `w = 0.5`.

| V₀ | V₁ | Flows | gain | weightedCapital | returnPercent |
|----|----|-------|------|-----------------|---------------|
| 1000 | 1100 | none | 100 | 1000 | **10** |
| 1000 | 1650 | +500 at t=50 | 150 | 1250 | **12** |
| 1000 | 640 | −400 at t=50 | 40 | 800 | **5** |
| 1000 | 1700 | +500 at t=0 (w=1), +200 at t=100 (w=0) | 0 | 1500 | **0** |
| 1000 | 950 | +200 at t=25, −100 at t=75 (unsorted) | −150 | 1125 | **≈ −13.333…** |
| 0 | 0 | none | 0 | 0 | **`null`** |
| 100 | 0 | −200 at t=0 | 100 | −100 | **`null`** |
| 0 | 110 | +100 at t=0 | 10 | 100 | **10** |

The last row is the “initial funding” case: a zero start value plus a contribution at the open can still produce a return.

UI displays `returnPercent === null` as `قابل محاسبه نیست؛ سرمایهٔ موزون باید مثبت باشد.`

---

## Constraints

- **One currency.** The card says values include cash and assets and must share a unit. There is no FX step.
- **Not annualized.** The result is the period rate only.
- **Not persisted.** React `useState` only; leaving the tab or page drops inputs.
- **Not the mock table.** Dietz does not read `MOCK_PORTFOLIOS` / `MOCK_POSITIONS` or `iran_portfolio_v1`.
- **Gregorian date inputs.** Labels are «شروع/پایان دوره (میلادی)»; this is not Jalali.
- **No npm script.** Tests are not in `package.json`. From `frontend-nextjs/` after `npm install` (needs the `typescript` devDependency):

```bash
cd frontend-nextjs
node --test src/lib/modified-dietz.test.cjs
```

---

## Troubleshooting

| Symptom | Cause |
|---------|--------|
| Card never shows a rate | Beginning, ending, start, and end must all be set; incomplete flow rows block the call |
| `قابل محاسبه نیست` | `weightedCapital <= 0` (empty book, or withdrawals larger than beginning + remaining weighted inflows) |
| Throw about dates | End not after start, or a flow date outside the period (`min`/`max` on the date inputs match start/end) |
| Rate disagrees with mock “سود/زیان کل” | Mock cards use `(current_value − initial_capital) / initial_capital` with **no** cash-flow timing |
| Expected copilot / live quotes | Copilot is a separate card; Dietz never calls FastAPI |

---

## Related

- Allocation copilot (HHI, localStorage lots): [INVESTING_TOOLS.md](INVESTING_TOOLS.md)
- Public sample dashboard / portfolio tab: [PUBLIC_DEMO_DASHBOARD.md](PUBLIC_DEMO_DASHBOARD.md)
- Session / BFF / dashboard composition: [FRONTEND_SESSION.md](FRONTEND_SESSION.md)
