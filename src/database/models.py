from sqlalchemy import Column, Integer, String, Text, Numeric, ForeignKey, TIMESTAMP, Float, DateTime, UniqueConstraint, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    phone = Column(String(20), nullable=True)
    is_verified = Column(Boolean, default=False)
    risk_tolerance = Column(String(20), default='moderate')  # conservative, moderate, aggressive
    is_active = Column(Boolean, default=True)
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