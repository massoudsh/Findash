# API Documentation

FastAPI routers live in `src/api/endpoints/` and are registered in `src/main_refactored.py` (not `src/main.py`).

Interactive docs when the API is running: `/docs`, `/redoc`, `/openapi.json`.

Account, payment, KYC, alerts, and risk-policy workflows: [docs/ACCOUNT_PLATFORM.md](../../docs/ACCOUNT_PLATFORM.md).

## Adding new endpoints

1. Create `src/api/endpoints/my_router.py` with an `APIRouter`.
2. Import and `app.include_router(...)` in `src/main_refactored.py`.
3. Prefer existing auth deps: `get_current_active_user`, `require_admin`, `require_active_subscription`.

## Account-platform routers (2026-09)

| File | Prefix | Notes |
|------|--------|--------|
| `payment_zarinpal.py` | `/api/payment/zarinpal` | Shared `create_order()` + purpose dispatch |
| `wallet.py` | `/api/wallet` | IRT ledger; deposit does not credit until verify |
| `subscriptions.py` | `/api/subscriptions` | Server-side plan prices; `require_active_subscription` |
| `kyc.py` | `/api/kyc` | National-code form + admin review |
| `price_alerts.py` | `/api/alerts` | One-shot rules vs Iran-market overview |
| `risk_policy.py` | `/api/risk-policy` | Drawdown / concentration; optional bot stop |
| `admin_panel.py` | `/api/admin` | Users + audit log |
| `pdf_reports.py` | `/api/reports` | Persian portfolio PDF |
| `otp_auth.py` | `/api/auth` | `/send-otp`, `/verify-otp` (in-memory store) |
| `trading_bots.py` | `/api/trading-bots` | `POST /{id}/start` is subscription-gated |

## Other routers

LLM (`llm.py` / `llm_simple.py`), portfolio (`portfolio_api.py`), market data (`unified_market_data.py`, `iran_market.py`), and remaining feature routers are also included from `src/main_refactored.py`. Treat `/docs` as the live catalog.
