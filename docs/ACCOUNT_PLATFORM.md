# Account platform — payments, wallet, subscriptions, KYC, alerts, risk

Operational runbook for the account and commerce surfaces added in September 2026 (issues #12–#14, #18–#22). All paths below are verified against the FastAPI routers registered in `src/main_refactored.py`.

**Ports:** Docker compose maps the API to `http://localhost:8011` (container `:8000`). Local `uvicorn` / `python3 start.py` typically serves `:8000`. Frontend Docker and `npm run dev` use `http://localhost:3003`. Interactive OpenAPI: `/docs`.

---

## Intent

These modules turn Findash from a market-data dashboard into an Iran-market account platform:

| Concern | What it actually does |
|---------|------------------------|
| Payments | One ZarinPal order pipeline with a `purpose` that decides side effects |
| Wallet | IRT ledger; deposits credit only after verify; withdrawals lock funds |
| Subscriptions | Server-priced plans; activate/renew only after verify; gate bot start |
| KYC | National-code form + admin review — no live registry lookup |
| Alerts | One-shot price rules vs `/api/iran-market/overview`, Celery every 60s |
| Risk policy | Daily drawdown + concentration vs real portfolio snapshots, Celery every 5m |
| Admin | Real users, roles, and audit log (not mock data) |
| Reports | Persian RTL PDF of the caller’s own portfolio |

---

## Architecture

```
Browser / Next.js BFF
        │  JWT Bearer
        ▼
FastAPI (src/main_refactored.py)
  ├─ payment_zarinpal.create_order()   ← shared by wallet + subscriptions
  ├─ _dispatch_payment_success()       ← wallet_topup | subscription | general
  ├─ wallet / subscriptions / kyc / alerts / risk-policy / admin / reports
  └─ notifications service (SMS / Web Push / in-app)
        │
        ├─ PostgreSQL (SQLAlchemy models + payment_orders)
        └─ Celery Beat → Redis → Worker
              notifications.evaluate_price_alerts   (60s)
              notifications.evaluate_risk_policies  (300s)
```

**Frontend BFF (Next.js Route Handlers)** currently proxy:

- `/api/subscriptions/{plans,me,subscribe}`
- `/api/risk-policy/{me,breaches}`
- `/api/admin/{users,users/[id],audit-log}`
- `/api/payment/zarinpal/create`

Wallet, KYC, alerts, and PDF reports are FastAPI-only; call `NEXT_PUBLIC_API_URL` with a JWT.

---

## 1. ZarinPal payments

**Code:** `src/api/endpoints/payment_zarinpal.py`

### Flow

1. Authenticated caller creates an order (`POST /api/payment/zarinpal/create` or a helper such as `/api/wallet/deposit` / `/api/subscriptions/subscribe`).
2. API stores `payment_orders` (`pending`) and returns `authority` + `redirect_url` (`https://www.zarinpal.com/pg/StartPay/{authority}`).
3. User pays on ZarinPal.
4. Gateway hits `GET /api/payment/zarinpal/callback?Status=&Authority=`.
5. Callback **always verifies** with ZarinPal (never trusts `Status` alone). Codes `100` and `101` (already verified) count as success.
6. On success: `status=paid`, then `_dispatch_payment_success`.
7. User is redirected to `{APP_BASE_URL}/payment/success?id=&ref=` or `/payment/failed?...`.

Default callback if none is passed: `{APP_BASE_URL}/payment/callback/zarinpal` (`APP_BASE_URL` defaults to `http://localhost:3003`).

### Endpoints

| Method | Path | Auth | Notes |
|--------|------|------|--------|
| POST | `/api/payment/zarinpal/create` | user | `purpose` may be `general` or `wallet_topup` only |
| GET | `/api/payment/zarinpal/callback` | none (gateway) | verify + dispatch + redirect |
| GET | `/api/payment/zarinpal/status/{order_id}` | owner | |
| GET | `/api/payment/zarinpal/history` | user | last 50 orders |

### Constraints

- Amounts are **toman**. ZarinPal is called with `amount * 10` (rial).
- Generic create minimum: **1,000 toman**. Wallet deposit/withdraw minimum: **10,000 toman**.
- `purpose=subscription` is **rejected** on `/create`. Price must come from `SubscriptionPlan` via `/api/subscriptions/subscribe`.
- Dispatch failure after a successful verify is logged and the user still lands on `/payment/success` — reconcile manually from `payment_orders`.
- Requires `ZARINPAL_MERCHANT_ID`. Missing merchant → HTTP 503.

### Example — generic payment

```bash
curl -X POST "$API/api/payment/zarinpal/create" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"amount_toman": 50000, "description": "شارژ آزمایشی", "purpose": "general"}'
```

---

## 2. IRT wallet

**Code:** `src/api/endpoints/wallet.py`  
**Currency:** `IRT` only (toman). Ledger fields: `balance`, `available`, `locked`, `pending`.

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/api/wallet/balances` | Create-on-read zero balance if missing |
| GET | `/api/wallet/transactions` | Newest first; `limit` capped at 200 |
| GET | `/api/wallet/bank-accounts` | Sheba masked as `IR...` + last 4 |
| POST | `/api/wallet/bank-accounts/link` | IBAN checksum (`IR` + 24 digits, mod-97) |
| DELETE | `/api/wallet/bank-accounts/{id}` | Owner only |
| POST | `/api/wallet/deposit` | ZarinPal order, `purpose=wallet_topup` — **does not credit yet** |
| POST | `/api/wallet/withdraw` | Locks `available` → `locked`, tx `pending` |

### Deposit vs withdraw

- **Deposit:** credit happens only in `_dispatch_payment_success` after verify (`type=deposit`, `status=completed`, `method=card`).
- **Withdraw:** no payout provider is wired. Funds stay locked until an operator settles (ZarinPal Payout / Jibit / manual). Linked Sheba starts `verified=false`.

### Example — deposit

```bash
curl -X POST "$API/api/wallet/deposit" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"amount_toman": 100000}'
# → { "authority", "redirect_url", "order_id" }
```

---

## 3. Subscriptions and trading-bot gate

**Code:** `src/api/endpoints/subscriptions.py`, gate used in `src/api/endpoints/trading_bots.py`

### Default plans (seeded if the table is empty)

| code | name_fa | price_toman | duration_days |
|------|---------|-------------|----------------|
| `basic` | پایه | 99,000 | 30 |
| `pro` | حرفه‌ای | 249,000 | 30 |
| `elite` | الیت | 499,000 | 30 |

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/subscriptions/plans` | public; seeds defaults if empty |
| POST | `/api/subscriptions/subscribe` | user; body `{ "plan_code", "callback_url?" }` |
| GET | `/api/subscriptions/me` | user; `active` is false when missing or `end_at` ≤ now |

On paid `purpose=subscription`:

- Active unexpired sub → extend `end_at` by `plan.duration_days` and switch `plan_id`.
- Otherwise → new `UserSubscription` (`status=active`).

### Bot gate (issue #14)

`POST /api/trading-bots/{bot_id}/start` depends on `require_active_subscription`:

- Admins skip the check.
- Others without an unexpired `active` row get **HTTP 402**.
- Create / stop / delete stay ungated so users can configure paper bots before buying.

```bash
curl -X POST "$API/api/subscriptions/subscribe" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"plan_code": "pro"}'
```

---

## 4. KYC (admin review)

**Code:** `src/api/endpoints/kyc.py`

No Finnotech/Jibit/Zibal call is implemented. `provider` / `provider_reference` are reserved.

Validation that **is** implemented:

- Iranian national-code checksum (reject all-same digits).
- Iranian mobile (`09…` / `+98` / `0098`) normalized to `09XXXXXXXXX`.
- Optional Shamsi birth date `1[23]YY-MM-DD`.

| Method | Path | Auth |
|--------|------|------|
| POST | `/api/kyc/submit` | user; verified profiles cannot resubmit |
| GET | `/api/kyc/me` | user; `not_submitted` if none |
| GET | `/api/kyc/pending` | admin |
| POST | `/api/kyc/{kyc_id}/review` | admin; `verified` \| `rejected` (reason required on reject) |

Review writes `audit_logs.action=kyc.reviewed` and an in-app notification (`category=kyc`). National code is masked in responses (`XXX***XX`).

---

## 5. Price alerts

**Code:** `src/api/endpoints/price_alerts.py`, `src/notifications/tasks.py`

Rules fire against **current** `GET /api/iran-market/overview` prices (tgju + Nobitex). Symbols must match overview keys, for example:

`USD-IRR`, `EUR-IRR`, `GBP-IRR`, `GOLD18-IRT`, `GOLD24-IRT`, `COIN-IRT`, `HALFCOIN-IRT`, `QUARTERCOIN-IRT`, `BTC-IRT`, `ETH-IRT`, `USDT-IRT`, `BNB-IRT`, `TRX-IRT`.

| Method | Path | Auth |
|--------|------|------|
| POST | `/api/alerts/rules` | user; `direction` `above`\|`below`; channels `in_app`\|`push`\|`sms` |
| GET | `/api/alerts/rules` | user; active only unless `include_triggered=true` |
| DELETE | `/api/alerts/rules/{id}` | owner |
| POST | `/api/alerts/push-subscribe` | user; VAPID keys |
| DELETE | `/api/alerts/push-subscribe?endpoint=` | owner |
| POST | `/api/alerts/evaluate` | admin; manual run |

On hit: rule is **deactivated** (`is_active=false`, `triggered_at` set) — one-shot. SMS uses `User.phone`. Push uses stored `push_subscriptions`. Missing `SMS_API_KEY` / `VAPID_PRIVATE_KEY` logs a dev fallback and does not crash.

Beat: `notifications.evaluate_price_alerts` every **60 seconds**.

```bash
curl -X POST "$API/api/alerts/rules" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-IRT","direction":"above","target_price":5000000000,"channels":["in_app","sms"]}'
```

---

## 6. Risk policy engine

**Code:** `src/api/endpoints/risk_policy.py`

Per-user policy (created on first `GET /me`). Defaults from the model: **5%** max daily drawdown, **30%** max single-position concentration, `action_on_breach=alert`, `enabled=true`.

Evaluation uses **real** rows:

- Drawdown: latest `PortfolioSnapshot.daily_pnl_percent` ≤ `-max_daily_drawdown_pct`
- Concentration: `Position.market_value / Portfolio.total_value * 100` for active positions

| Method | Path | Auth |
|--------|------|------|
| GET/PUT | `/api/risk-policy/me` | user; `action_on_breach` is `alert` or `stop_bots` |
| GET | `/api/risk-policy/breaches` | user |
| POST | `/api/risk-policy/me/evaluate` | user |
| POST | `/api/risk-policy/evaluate-all` | admin |

On a new breach: in-app `category=risk` notification. Same `rule` is recorded **once per user per calendar day**. `stop_bots` flips that user’s JSON-persisted bots (`src/api/bots_persistence.py`) from `active` → `stopped`.

Beat: `notifications.evaluate_risk_policies` every **300 seconds**.

---

## 7. Admin panel

**Code:** `src/api/endpoints/admin_panel.py` — `require_admin` on every route.

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/api/admin/users` | Real users + trade counts + summed `portfolio.total_value` |
| PATCH | `/api/admin/users/{id}` | `role` ∈ `{admin,trader,demo}` and/or `is_active` |
| GET | `/api/admin/audit-log` | Newest first; `limit` 1–500 (default 100) |

Admins **cannot** deactivate themselves or drop their own `admin` role. Successful user updates write `audit_logs.action=user.updated`.

---

## 8. Persian PDF reports

**Code:** `src/api/endpoints/pdf_reports.py`, `src/services/pdf_reports.py`

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/api/reports/portfolio.pdf` | `application/pdf`; optional `portfolio_id`, `include_trades` (default true, last 20) |
| GET | `/api/reports/portfolio/preview` | metadata + `download_url` (Jalali date) |

- Another user’s (or missing) portfolio → **404**, not 403.
- Insights are deterministic (concentration ≥ 30%, return vs `initial_cash`) — no LLM.
- Requires Vazirmatn at `/usr/share/fonts/truetype/vazirmatn/`. Missing font → **503**.
- RTL needs `arabic_reshaper` + `python-bidi` + a Persian TTF (declared in `requirements/requirements.txt`).

---

## 9. OTP (phone login)

**Code:** `src/api/endpoints/otp_auth.py` — prefix `/api/auth`

| Method | Path | Limits |
|--------|------|--------|
| POST | `/api/auth/send-otp` | 3 sends / 10 minutes / phone; 6-digit code, 5-minute TTL |
| POST | `/api/auth/verify-otp` | 3 wrong attempts then lock; returns JWT |

KaveNegar uses template `findash-otp` (`verify/lookup`). Without `SMS_PROVIDER=kavenegar` + `SMS_API_KEY`, the code is logged and the API still returns success (anti-enumeration).

**Pitfall:** codes live in **process memory** (`_OTP_STORE`). Multiple API workers or a restart lose pending OTPs. Redis is noted in-code as the production store but is not wired.

---

## Environment (names only)

Copy `.env.example`. Production fails closed if `ZARINPAL_MERCHANT_ID` is empty (`src/core/config.py`).

| Variable | Used by |
|----------|---------|
| `ZARINPAL_MERCHANT_ID` | Payment create/verify |
| `APP_BASE_URL` | Callback + success/fail redirects |
| `SMS_PROVIDER`, `SMS_API_KEY`, `SMS_SENDER_LINE` | OTP + alert SMS (KaveNegar) |
| `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_ADMIN_EMAIL` | Web Push |
| `CELERY_BROKER_URL` | Alert/risk beat |

Never commit real keys. SMS/push degrade to log-only when unset.

---

## Database

Schema lives in `src/database/models.py` plus `PaymentOrder` in `payment_zarinpal.py`. Migration `src/alembic/versions/20260919_admin_risk_subscription_schema.py` (`admin_risk_sub_001`) creates audit, notification, subscription, and risk-policy tables. Apply from repo root:

```bash
alembic upgrade head
```

| Table | Owner module |
|-------|----------------|
| `payment_orders` | ZarinPal |
| `wallet_balances`, `wallet_transactions`, `bank_accounts` | Wallet |
| `subscription_plans`, `user_subscriptions` | Subscriptions |
| `kyc_profiles` | KYC |
| `price_alert_rules`, `push_subscriptions` | Alerts |
| `risk_policies`, `risk_policy_breaches` | Risk policy |
| `audit_logs`, `notifications` | Admin / in-app |

---

## Operator runbook

### Payments “success but no credit / no sub”

1. Confirm `payment_orders.status=paid` and `purpose`.
2. Search API logs for `Payment success but dispatch failed`.
3. Re-apply the side effect (wallet credit or `UserSubscription`) from `amount_toman` / `purpose_ref`; do not re-verify blindly (code `101` is already-verified).

### Withdrawals stuck in `pending`

Expected until a payout provider is connected. Unlock/complete the row only after the Sheba transfer is actually sent.

### Alerts never fire

- Celery worker **and** beat must be up (`docker compose -f docker-compose-core.yml logs -f celery-worker celery-beat`).
- Symbol must exist in overview with a non-null `price`.
- Rule must still be `is_active`.
- SMS needs `User.phone`; push needs a stored subscription + VAPID.

### Risk policy never stops bots

- Policy `enabled=true` and `action_on_breach=stop_bots`.
- Latest snapshot / position values must actually breach the thresholds.
- Same rule is ignored for the rest of that calendar day.
- Bots are the JSON store used by `trading_bots.py`, not a separate broker.

### PDF 503

Install Vazirmatn on the API host (paths listed in `src/services/pdf_reports.py`) and the `arabic-reshaper` / `python-bidi` / `reportlab` packages.

### OTP works in logs but not on a second replica

In-memory store. Use a single API process locally, or move the store to Redis before horizontally scaling.

---

## Frontend pages

| Route | Role |
|-------|------|
| `/account`, `/account?tab=subscription`, `/account/subscription` | Profile / plan status |
| `/payment/checkout` | Plan picker |
| `/payment/callback/zarinpal`, `/payment/success`, `/payment/failed` | Gateway return |
| `/alerts` | Price-alert UI |
| `/risk/policy` | Risk-policy editor |
| `/admin`, `/audit-log` | Admin + audit (admin role) |
| `/auth/otp`, `/auth/phone` | Phone OTP |
| `/reports` | Report UI (PDF download via `/api/reports/portfolio.pdf`) |
