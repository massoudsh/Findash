# Modified Dietz calculator

Client-only **Modified Dietz** period return on `/dashboard?tab=portfolio`. It is not wired to the wallet, mock holdings, or the allocation copilot.

Full runbook (formula, unit-test examples, constraints): [docs/MODIFIED_DIETZ.md](https://github.com/massoudsh/Findash/blob/main/docs/MODIFIED_DIETZ.md).

## Start here

| If you want to… | Go to |
|-----------------|--------|
| Compute a cash-flow-weighted period return from typed values | Portfolio tab → card «بازده با روش Modified Dietz» |
| Score concentration of Iran lots | [[Investing Tools]] (`POST /api/copilot/allocation-analysis`) |
| See which dashboard numbers are live vs mock | [[Public Demo Dashboard]] |

## Do not assume

- The calculator does **not** read `localStorage` (`iran_portfolio_v1`) or `MOCK_PORTFOLIOS`.
- The result is **not** annualized. `returnPercent` is `null` when weighted capital is ≤ 0 — not a 0% return.
- There is no FastAPI route. Tests: `cd frontend-nextjs && node --test src/lib/modified-dietz.test.cjs` (needs `typescript`).

## Next

- [[Frontend]] — dashboard tabs
- [[Investing Tools]] — copilot / `/investing`
- [[Home]]
