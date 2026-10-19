# Iran Market Pricing

Quote contract for `/api/iran-market/*`: every overview `price` is **rial (`IRR`)**, including symbols that end in `-IRT`. Provider failure returns `available: false`, not sample prices.

Full runbook (endpoints, UI units, alert pitfall, `formatTomanValue`): **[docs/IRAN_MARKET_PRICING.md](../docs/IRAN_MARKET_PRICING.md)** in the repo.

## Quick facts

| Item | Value |
|------|--------|
| Overview / prices / ticker | No auth; 60s in-process cache |
| Nobitex crypto `latest` | Rial — do not divide by 10 |
| Market tab | Live overview only; mock seed removed |
| Blue ticker | Number only (no unit suffix) |
| News ticker | Labels «ریال» |
| TEDPIX | Placeholder (`available: false`) |
| Alerts | `target_price` compared to rial overview prices |
| Wallet / ZarinPal | Still toman (separate from quotes) |

## Next

- [[Public Demo Dashboard]] — dashboard tabs and ports
- [[Account Platform]] — alerts, wallet, payments
- [[Investing Tools]] — screener uses overview
- [[API Reference]]
- [[Home]]
