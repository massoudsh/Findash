from sqlalchemy import Column, Integer, BigInteger, String, Text, Numeric, ForeignKey, TIMESTAMP, Float, DateTime, UniqueConstraint, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    is_verified = Column(Boolean, default=False)
    risk_tolerance = Column(String(20), default='moderate')  # conservative, moderate, aggressive
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default='trader')  # admin, trader, demo
    permissions = Column(JSON, default=list)
    last_login = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    portfolios = relationship('Portfolio', back_populates='user', cascade="all, delete-orphan")

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    initial_cash = Column(Numeric(15, 2), default=10000.00)
    current_cash = Column(Numeric(15, 2), default=10000.00)
    total_value = Column(Numeric(15, 2), default=0.00)
    is_active = Column(Boolean, default=True)
    risk_level = Column(String(20), default='moderate')  # low, moderate, high
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', back_populates='portfolios')
    positions = relationship('Position', back_populates='portfolio', cascade="all, delete-orphan")
    trades = relationship('Trade', back_populates='portfolio', cascade="all, delete-orphan")
    snapshots = relationship('PortfolioSnapshot', back_populates='portfolio', cascade="all, delete-orphan")
    risk_metrics = relationship('RiskMetrics', back_populates='portfolio', uselist=False, cascade="all, delete-orphan")

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(32), nullable=False)
    quantity = Column(Numeric(15, 6), nullable=False)
    average_price = Column(Numeric(15, 6), nullable=False)
    current_price = Column(Numeric(15, 6), default=0.000000)
    market_value = Column(Numeric(15, 2), default=0.00)
    unrealized_pnl = Column(Numeric(15, 2), default=0.00)
    position_type = Column(String(10), default='long')  # long, short
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    portfolio = relationship('Portfolio', back_populates='positions')
    __table_args__ = (UniqueConstraint('portfolio_id', 'symbol', name='unique_portfolio_symbol'),)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(32), nullable=False)
    trade_type = Column(String(4), nullable=False)  # BUY, SELL
    quantity = Column(Numeric(15, 6), nullable=False)
    price = Column(Numeric(15, 6), nullable=False)
    total_amount = Column(Numeric(15, 2), nullable=False)
    fees = Column(Numeric(10, 2), default=0.00)
    trade_date = Column(TIMESTAMP, server_default=func.now())
    settlement_date = Column(TIMESTAMP, nullable=True)
    status = Column(String(20), default='pending')  # pending, executed, cancelled, failed
    order_id = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    portfolio = relationship('Portfolio', back_populates='trades')

class PortfolioSnapshot(Base):
    __tablename__ = 'portfolio_snapshots'
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False)
    snapshot_date = Column(TIMESTAMP, server_default=func.now())
    total_value = Column(Numeric(15, 2), nullable=False)
    cash_value = Column(Numeric(15, 2), default=0.00)
    positions_value = Column(Numeric(15, 2), default=0.00)
    daily_pnl = Column(Numeric(15, 2), default=0.00)
    daily_pnl_percent = Column(Numeric(8, 4), default=0.0000)
    portfolio = relationship('Portfolio', back_populates='snapshots')
    __table_args__ = (UniqueConstraint('portfolio_id', 'snapshot_date', name='unique_portfolio_snapshot'),)

class RiskMetrics(Base):
    __tablename__ = 'risk_metrics'
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'), nullable=False, unique=True)
    value_at_risk_1d = Column(Numeric(15, 2), nullable=True)
    value_at_risk_1w = Column(Numeric(15, 2), nullable=True)
    value_at_risk_1m = Column(Numeric(15, 2), nullable=True)
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    beta = Column(Numeric(8, 4), nullable=True)
    volatility = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=True)
    concentration_risk = Column(String(20), default='medium')  # low, medium, high
    last_updated = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    portfolio = relationship('Portfolio', back_populates='risk_metrics')

class MarketData(Base):
    __tablename__ = 'financial_time_series'
    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False)
    symbol = Column(String(32), nullable=False)
    price = Column(Float)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Integer)
    exchange = Column(String(32))
    __table_args__ = (UniqueConstraint('time', 'symbol', name='_time_symbol_uc'),)

class NewsArticle(Base):
    __tablename__ = 'news_articles'
    id = Column(Integer, primary_key=True)
    title = Column(String(512), nullable=False)
    summary = Column(Text)
    url = Column(String(1024))
    published_at = Column(DateTime)
    source = Column(String(128))
    symbol = Column(String(32))

class RedditSentiment(Base):
    __tablename__ = 'reddit_sentiment'
    id = Column(Integer, primary_key=True)
    subreddit = Column(String(128), nullable=False)
    positive = Column(Integer)
    negative = Column(Integer)
    neutral = Column(Integer)
    top_keywords = Column(String(256))  # Comma-separated
    sample_size = Column(Integer)
    analyzed_at = Column(DateTime)

class AlertRule(Base):
    __tablename__ = 'alert_rules'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(64), nullable=False)
    metric = Column(String(64), nullable=False)
    operator = Column(String(8), nullable=False)  # >, <, =, >=, <=
    threshold = Column(Float, nullable=False)
    duration = Column(String(16), nullable=False)
    notifications = Column(JSON, nullable=False, default={})  # {email: bool, sms: bool, push: bool, webhook: bool}
    severity = Column(String(16), nullable=False, default='medium')
    enabled = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', backref='alert_rules')

# Phase 3: Agent Monitoring Models
class AgentStatus(Base):
    __tablename__ = 'agent_status'
    id = Column(String(32), primary_key=True)  # e.g., 'M1', 'M2'
    name = Column(String(128), nullable=False)
    status = Column(String(20), default='idle')  # active, idle, error, training
    uptime = Column(Integer, default=0)  # seconds
    tasks_completed = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_execution_time = Column(Float, default=0.0)  # milliseconds
    current_task = Column(Text, nullable=True)
    last_activity = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())

class AgentLog(Base):
    __tablename__ = 'agent_logs'
    id = Column(String(64), primary_key=True)
    agent_id = Column(String(32), ForeignKey('agent_status.id'), nullable=False)
    level = Column(String(20), nullable=False)  # info, warning, error, success
    message = Column(Text, nullable=False)
    context = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)  # milliseconds
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    agent = relationship('AgentStatus', backref='logs')

class AgentDecision(Base):
    __tablename__ = 'agent_decisions'
    id = Column(String(64), primary_key=True)
    agent_id = Column(String(32), ForeignKey('agent_status.id'), nullable=False)
    symbol = Column(String(32), nullable=False)
    action = Column(String(10), nullable=False)  # buy, sell, hold
    confidence = Column(Float, nullable=False)
    reasoning = Column(JSON, nullable=False)  # List of strings
    executed = Column(Boolean, default=False)
    result = Column(JSON, nullable=True)  # {price, pnl, status}
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    agent = relationship('AgentStatus', backref='decisions')

# Phase 3: Wallet & Funding Models
class WalletBalance(Base):
    __tablename__ = 'wallet_balances'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    currency = Column(String(10), nullable=False)  # USD, BTC, ETH
    balance = Column(Numeric(20, 8), default=0.0)
    available = Column(Numeric(20, 8), default=0.0)
    locked = Column(Numeric(20, 8), default=0.0)
    pending = Column(Numeric(20, 8), default=0.0)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', backref='wallet_balances')
    __table_args__ = (UniqueConstraint('user_id', 'currency', name='unique_user_currency'),)

class WalletTransaction(Base):
    __tablename__ = 'wallet_transactions'
    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(20), nullable=False)  # deposit, withdrawal, transfer, fee
    amount = Column(Numeric(20, 8), nullable=False)
    currency = Column(String(10), nullable=False)
    status = Column(String(20), default='pending')  # pending, completed, failed, cancelled
    method = Column(String(20), nullable=False)  # bank, crypto, card, wire
    description = Column(Text, nullable=True)
    reference = Column(String(128), nullable=True, unique=True)
    fees = Column(Numeric(10, 2), nullable=True)
    bank_account_id = Column(Integer, ForeignKey('bank_accounts.id'), nullable=True)
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    user = relationship('User', backref='wallet_transactions')
    bank_account = relationship('BankAccount', backref='transactions')

class BankAccount(Base):
    """حساب بانکی/شبا کاربر برای برداشت از کیف پول (issue #21).

    برداشت واقعی وجه از طریق درگاه‌های Payout ایرانی (مانند زرین‌پال Payout یا جیبیت)
    نیاز به قرارداد جداگانه با ارائه‌دهنده دارد؛ اینجا فقط ثبت/تأیید شماره شبا انجام می‌شود
    و رکورد withdrawal با وضعیت pending برای پردازش دستی/بعدی ایجاد می‌گردد.
    """
    __tablename__ = 'bank_accounts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(128), nullable=False)
    sheba_number = Column(String(26), nullable=False)     # IRxxxxxxxxxxxxxxxxxxxxxxxx
    card_number_last4 = Column(String(4), nullable=True)
    bank_name = Column(String(128), nullable=False)
    type = Column(String(20), nullable=False, default='sheba')  # sheba
    verified = Column(Boolean, default=False)
    last_used = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', backref='bank_accounts')

# Phase 3: Security Models
class APIKey(Base):
    __tablename__ = 'api_keys'
    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(128), nullable=False)
    key_hash = Column(String(256), nullable=False)  # Hashed API key
    key_preview = Column(String(64), nullable=False)  # e.g., 'sk_live_***xyz789'
    permissions = Column(JSON, nullable=False)  # List of permissions
    ip_whitelist = Column(JSON, nullable=True)  # List of IPs
    status = Column(String(20), default='active')  # active, revoked, expired
    last_used = Column(TIMESTAMP, nullable=True)
    expires_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    user = relationship('User', backref='api_keys')

class UserSession(Base):
    __tablename__ = 'user_sessions'
    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    device = Column(String(128), nullable=False)
    browser = Column(String(128), nullable=False)
    location = Column(String(128), nullable=True)
    ip_address = Column(String(45), nullable=False)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    status = Column(String(20), default='active')  # active, expired, revoked
    login_time = Column(TIMESTAMP, server_default=func.now())
    last_activity = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    expires_at = Column(TIMESTAMP, nullable=True)
    user = relationship('User', backref='sessions')

class TradingPermission(Base):
    __tablename__ = 'trading_permissions'
    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, default=False)
    restrictions = Column(JSON, nullable=True)  # {max_order_size, allowed_symbols, etc.}
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', backref='trading_permissions')

# ─────────────────────────────────────────────
# Admin panel (issue #12): real audit log + generic in-app notifications
# ─────────────────────────────────────────────

class AuditLog(Base):
    """هر رویداد مهم (تغییر نقش، مسدودسازی، ورود ادمین، ...) اینجا ثبت می‌شود."""
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    actor_user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    action = Column(String(64), nullable=False)          # e.g. 'user.role_changed'
    target_type = Column(String(32), nullable=True)       # e.g. 'user', 'subscription'
    target_id = Column(String(64), nullable=True)
    detail = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    actor = relationship('User', foreign_keys=[actor_user_id])


class Notification(Base):
    """اعلان درون‌برنامه‌ای عمومی — یادآوری اشتراک، هشدار قیمت، وضعیت KYC و ..."""
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    category = Column(String(32), nullable=False)   # subscription | price_alert | kyc | admin | system
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)
    user = relationship('User', backref='notifications')

# ─────────────────────────────────────────────
# Subscription plan management (issue #13)
# ─────────────────────────────────────────────

class SubscriptionPlan(Base):
    __tablename__ = 'subscription_plans'
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)   # 'free' | 'pro_monthly' | 'pro_yearly'
    name_fa = Column(String(128), nullable=False)
    price_toman = Column(Integer, nullable=False, default=0)
    duration_days = Column(Integer, nullable=False, default=30)
    features = Column(JSON, nullable=True)   # لیست entitlement های واقعی این پلن
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())


class UserSubscription(Base):
    __tablename__ = 'user_subscriptions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    plan_id = Column(Integer, ForeignKey('subscription_plans.id'), nullable=False)
    status = Column(String(16), nullable=False, default='active')  # active | expired | cancelled
    start_at = Column(TIMESTAMP, server_default=func.now())
    end_at = Column(TIMESTAMP, nullable=False)
    auto_renew = Column(Boolean, default=False)
    reminder_sent_at = Column(TIMESTAMP, nullable=True)
    last_payment_order_id = Column(BigInteger, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    user = relationship('User', backref='subscriptions')
    plan = relationship('SubscriptionPlan')

# ─────────────────────────────────────────────
# Risk Policy Engine (issue #22)
# ─────────────────────────────────────────────

class RiskPolicy(Base):
    """سیاست ریسک قابل‌تنظیم هر کاربر برای توقف/هشدار خودکار ربات‌های معاملاتی."""
    __tablename__ = 'risk_policies'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    max_daily_drawdown_pct = Column(Numeric(5, 2), nullable=False, default=5.0)     # حداکثر افت روزانه مجاز
    max_position_concentration_pct = Column(Numeric(5, 2), nullable=False, default=30.0)  # حداکثر تمرکز روی یک دارایی
    action_on_breach = Column(String(16), nullable=False, default='alert')  # alert | stop_bots
    enabled = Column(Boolean, default=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    user = relationship('User', backref='risk_policy', uselist=False)


class RiskPolicyBreach(Base):
    """تاریخچه‌ی هر بار نقض سیاست ریسک — برای پنل و اعلان."""
    __tablename__ = 'risk_policy_breaches'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    rule = Column(String(64), nullable=False)   # 'max_daily_drawdown' | 'max_position_concentration'
    value = Column(Numeric(10, 4), nullable=False)
    threshold = Column(Numeric(10, 4), nullable=False)
    action_taken = Column(String(16), nullable=False)  # alert | stop_bots
    created_at = Column(TIMESTAMP, server_default=func.now())

# ─────────────────────────────────────────────
# Price alerts via Push/SMS (issue #19)
# ─────────────────────────────────────────────

class PriceAlertRule(Base):
    __tablename__ = 'price_alert_rules'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)   # above | below
    target_price = Column(Numeric(20, 4), nullable=False)
    channels = Column(JSON, nullable=False, default=lambda: ['in_app'])  # in_app, push, sms
    note = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    triggered_at = Column(TIMESTAMP, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    user = relationship('User', backref='price_alert_rules')


class PushSubscription(Base):
    """اشتراک Web Push مرورگر کاربر (طبق استاندارد VAPID)."""
    __tablename__ = 'push_subscriptions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    endpoint = Column(Text, nullable=False, unique=True)
    p256dh_key = Column(String(255), nullable=False)
    auth_key = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    user = relationship('User', backref='push_subscriptions')

# ─────────────────────────────────────────────
# KYC — احراز هویت مالی (issue #20)
# نکته مهم: طبق خودِ issue، استعلام هویت واقعی نیازمند انتخاب یک ارائه‌دهنده
# مجاز (Finnotech/Jibit/Zibal و ...) و تأیید حقوقی/رگولاتوری است که تصمیم
# محصول/کسب‌وکار است، نه یک تصمیم فنی. این مدل فقط زیرساخت فرم + وضعیت را
# آماده می‌کند؛ فیلد provider_reference برای زمانی است که آن تصمیم گرفته شد.
# ─────────────────────────────────────────────

class KYCProfile(Base):
    __tablename__ = 'kyc_profiles'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    national_code = Column(String(10), nullable=False)
    full_name = Column(String(255), nullable=False)
    birth_date_shamsi = Column(String(10), nullable=True)   # YYYY-MM-DD شمسی
    mobile_number = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default='pending_review')
    # pending_review | verified | rejected  (تا زمانی که provider واقعی وصل نشده، فقط pending_review/rejected ممکن است)
    rejection_reason = Column(Text, nullable=True)
    provider = Column(String(32), nullable=True)      # نام سرویس احراز هویت، وقتی انتخاب شد
    provider_reference = Column(String(128), nullable=True)
    submitted_at = Column(TIMESTAMP, server_default=func.now())
    reviewed_at = Column(TIMESTAMP, nullable=True)
    user = relationship('User', backref='kyc_profile', uselist=False) 