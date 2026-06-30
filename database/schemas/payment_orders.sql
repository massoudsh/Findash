-- جدول سفارش‌های پرداخت زرین‌پال
-- payment_orders table for ZarinPal integration

CREATE TABLE IF NOT EXISTS payment_orders (
    id               BIGSERIAL PRIMARY KEY,
    user_id          VARCHAR(128)  NOT NULL,
    gateway          VARCHAR(32)   NOT NULL DEFAULT 'zarinpal',
    authority        VARCHAR(128)  NOT NULL,
    amount_rial      BIGINT        NOT NULL,
    amount_toman     BIGINT        NOT NULL,
    description      TEXT,
    status           VARCHAR(16)   NOT NULL DEFAULT 'pending',
        -- pending | paid | failed | expired
    ref_id           VARCHAR(64),
    card_pan         VARCHAR(20),
    error_code       VARCHAR(32),
    callback_payload JSONB,
    verify_payload   JSONB,
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    paid_at          TIMESTAMPTZ,

    CONSTRAINT payment_orders_gateway_authority_key UNIQUE (gateway, authority)
);

CREATE INDEX IF NOT EXISTS payment_orders_user_id_idx  ON payment_orders (user_id);
CREATE INDEX IF NOT EXISTS payment_orders_status_idx   ON payment_orders (status);
CREATE INDEX IF NOT EXISTS payment_orders_created_idx  ON payment_orders (created_at DESC);

COMMENT ON TABLE  payment_orders                  IS 'تراکنش‌های درگاه پرداخت ایرانی';
COMMENT ON COLUMN payment_orders.amount_rial      IS 'مبلغ به ریال — همیشه ریال در DB ذخیره می‌شود';
COMMENT ON COLUMN payment_orders.amount_toman     IS 'مبلغ به تومان — برای نمایش';
COMMENT ON COLUMN payment_orders.authority        IS 'شناسه یکتای تراکنش از زرین‌پال';
COMMENT ON COLUMN payment_orders.ref_id           IS 'کد رهگیری بانکی پس از تأیید';
COMMENT ON COLUMN payment_orders.callback_payload IS 'داده خام callback از درگاه';
COMMENT ON COLUMN payment_orders.verify_payload   IS 'داده خام پاسخ verify از درگاه';
