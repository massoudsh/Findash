"""
Database models for Quantum Trading Matrix
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
import uuid
from datetime import datetime

Base = declarative_base()

class OptionType(PyEnum):
    CALL = "call"
    PUT = "put"

class OrderStatus(PyEnum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class RiskLevel(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    risk_tolerance = Column(Enum(RiskLevel), default=RiskLevel.MEDIUM)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    option_positions = relationship("OptionPosition", back_populates="user")
    risk_reports = relationship("RiskReport", back_populates="user")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    initial_capital = Column(Float, default=100000.0)
    current_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=100000.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    option_positions = relationship("OptionPosition", back_populates="portfolio")
    portfolio_metrics = relationship("PortfolioMetrics", back_populates="portfolio")

class OptionPosition(Base):
    __tablename__ = "option_positions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    
    symbol = Column(String, nullable=False, index=True)
    option_type = Column(Enum(OptionType), nullable=False)
    strike_price = Column(Float, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    quantity = Column(Integer, nullable=False)  # positive for long, negative for short
    
    # Pricing data
    premium_paid = Column(Float, nullable=False)
    current_price = Column(Float)
    underlying_price = Column(Float, nullable=False)
    implied_volatility = Column(Float, default=0.2)
    risk_free_rate = Column(Float, default=0.05)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Metadata
    order_status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True))
    pnl = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="option_positions")
    portfolio = relationship("Portfolio", back_populates="option_positions")

class PortfolioMetrics(Base):
    __tablename__ = "portfolio_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    
    # Portfolio-level Greeks
    total_delta = Column(Float, default=0.0)
    total_gamma = Column(Float, default=0.0)
    total_theta = Column(Float, default=0.0)
    total_vega = Column(Float, default=0.0)
    total_rho = Column(Float, default=0.0)
    
    # Risk metrics
    var_95 = Column(Float)  # Value at Risk 95%
    var_99 = Column(Float)  # Value at Risk 99%
    expected_shortfall = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    beta = Column(Float)
    
    # Concentration metrics
    concentration_index = Column(Float)  # Herfindahl-Hirschman Index
    largest_position_weight = Column(Float)
    
    # Performance metrics
    total_return = Column(Float)
    daily_return = Column(Float)
    volatility = Column(Float)
    
    # Timestamp
    calculation_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="portfolio_metrics")

class RiskReport(Base):
    __tablename__ = "risk_reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(String, ForeignKey("portfolios.id"))
    
    report_type = Column(String, nullable=False)  # daily, weekly, monthly, stress_test
    risk_level = Column(Enum(RiskLevel), nullable=False)
    
    # Report content
    summary = Column(Text)
    recommendations = Column(Text)
    alerts = Column(Text)  # JSON string of alerts
    scenario_analysis = Column(Text)  # JSON string of scenario results
    
    # Risk scores
    overall_risk_score = Column(Float)
    liquidity_risk_score = Column(Float)
    concentration_risk_score = Column(Float)
    market_risk_score = Column(Float)
    
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="risk_reports")

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=False, index=True)
    
    # Price data
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    
    # Additional metrics
    implied_volatility = Column(Float)
    beta = Column(Float)
    market_cap = Column(Float)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CorrelationMatrix(Base):
    __tablename__ = "correlation_matrices"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbols = Column(Text, nullable=False)  # JSON array of symbols
    correlation_data = Column(Text, nullable=False)  # JSON correlation matrix
    period = Column(String, default="1y")  # time period for correlation
    calculation_date = Column(DateTime(timezone=True), server_default=func.now())

class TradingSignal(Base):
    __tablename__ = "trading_signals"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=False, index=True)
    signal_type = Column(String, nullable=False)  # buy, sell, hold
    strategy_name = Column(String, nullable=False)
    confidence = Column(Float)  # 0-1 confidence score
    
    # Signal details
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    reasoning = Column(Text)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    key_name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    
    # Permissions
    can_read = Column(Boolean, default=True)
    can_write = Column(Boolean, default=False)
    can_trade = Column(Boolean, default=False)
    
    # Usage tracking
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    action = Column(String, nullable=False)
    resource_type = Column(String)  # portfolio, position, user, etc.
    resource_id = Column(String)
    
    # Details
    old_values = Column(Text)  # JSON of old values
    new_values = Column(Text)  # JSON of new values
    ip_address = Column(String)
    user_agent = Column(String)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now()) 