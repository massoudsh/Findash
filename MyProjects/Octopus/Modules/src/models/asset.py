"""
SQLAlchemy models for Iranian Assets
TimescaleDB hypertable for price_history
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, Enum as SAEnum, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

# Assumes Base is imported from your db config
# from src.database import Base
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass


class AssetCategory(str, enum.Enum):
    GOLD = "gold"
    SILVER = "silver"
    CURRENCY = "currency"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"


class Asset(Base):
    """Master list of trackable assets"""
    __tablename__ = "assets"

    id          = Column(Integer, primary_key=True, index=True)
    symbol      = Column(String(20), unique=True, nullable=False, index=True)
    name_fa     = Column(String(100), nullable=False)
    name_en     = Column(String(100), nullable=False)
    category    = Column(SAEnum(AssetCategory), nullable=False, index=True)
    unit        = Column(String(20), default="تومان")
    source_key  = Column(String(100))          # tgju endpoint key
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime, server_default=func.now())

    prices      = relationship("AssetPriceSnapshot", back_populates="asset")
    histories   = relationship("AssetPriceHistory", back_populates="asset")


class AssetPriceSnapshot(Base):
    """
    Latest price snapshot — updated every ~60 seconds via Celery beat.
    One row per asset (upsert on symbol).
    """
    __tablename__ = "asset_price_snapshots"

    id                 = Column(Integer, primary_key=True)
    asset_id           = Column(Integer, ForeignKey("assets.id"), nullable=False)
    symbol             = Column(String(20), nullable=False, unique=True, index=True)
    price              = Column(Float, nullable=False)    # native unit (rial/toman/usd)
    price_toman        = Column(Float, nullable=False)    # always in toman
    change_24h         = Column(Float, default=0)
    change_percent_24h = Column(Float, default=0)
    high_24h           = Column(Float)
    low_24h            = Column(Float)
    source             = Column(String(50))
    updated_at         = Column(DateTime, server_default=func.now(), onupdate=func.now())

    asset = relationship("Asset", back_populates="prices")

    __table_args__ = (
        Index("ix_snapshot_updated", "updated_at"),
    )


class AssetPriceHistory(Base):
    """
    OHLCV history — TimescaleDB hypertable partitioned by timestamp.
    Migration note: Run `SELECT create_hypertable('asset_price_history', 'timestamp');`
    """
    __tablename__ = "asset_price_history"

    id        = Column(Integer, primary_key=True)
    asset_id  = Column(Integer, ForeignKey("assets.id"), nullable=False)
    symbol    = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float, nullable=False)
    volume    = Column(Float, default=0)
    interval  = Column(String(5), default="1d")  # 1h, 4h, 1d

    asset = relationship("Asset", back_populates="histories")

    __table_args__ = (
        Index("ix_history_symbol_ts", "symbol", "timestamp"),
    )


class UserPortfolioAsset(Base):
    """User's personal asset holdings"""
    __tablename__ = "portfolio_assets"

    id                  = Column(Integer, primary_key=True)
    user_id             = Column(Integer, nullable=False, index=True)   # FK to users table
    symbol              = Column(String(20), nullable=False)
    quantity            = Column(Float, nullable=False)
    buy_price           = Column(Float, nullable=False)      # price at purchase in toman
    buy_date            = Column(DateTime, nullable=False)
    notes               = Column(Text)
    created_at          = Column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_portfolio_user_symbol", "user_id", "symbol"),
    )


# ─── Seed Data ────────────────────────────────────────────────────────────────

ASSET_SEEDS = [
    # ── طلا ──────────────────────────────────────────────────
    {"symbol": "XAU18",       "name_fa": "طلای ۱۸ عیار (هر گرم)",   "name_en": "Gold 18K (per gram)",     "category": "gold",        "source_key": "geram18"},
    {"symbol": "XAU24",       "name_fa": "طلای ۲۴ عیار (هر گرم)",   "name_en": "Gold 24K (per gram)",     "category": "gold",        "source_key": "geram24"},
    {"symbol": "COIN_FULL",   "name_fa": "سکه بهار آزادی",           "name_en": "Bahar Azadi Coin",        "category": "gold",        "source_key": "sekeb"},
    {"symbol": "COIN_HALF",   "name_fa": "نیم‌سکه",                  "name_en": "Half Coin",               "category": "gold",        "source_key": "nim"},
    {"symbol": "COIN_QUARTER","name_fa": "ربع‌سکه",                  "name_en": "Quarter Coin",            "category": "gold",        "source_key": "rob"},
    {"symbol": "COIN_OLD",    "name_fa": "سکه قدیم",                 "name_en": "Old Coin",                "category": "gold",        "source_key": "sekeq"},
    {"symbol": "MESGHAL",     "name_fa": "مثقال طلا",                 "name_en": "Gold Mithqal",            "category": "gold",        "source_key": "mesghal"},
    # ── نقره ────────────────────────────────────────────────
    {"symbol": "XAG",         "name_fa": "نقره (هر گرم)",            "name_en": "Silver (per gram)",       "category": "silver",      "source_key": "silver"},
    # ── ارز ────────────────────────────────────────────────
    {"symbol": "USD",         "name_fa": "دلار آمریکا",              "name_en": "US Dollar",               "category": "currency",    "source_key": "price_dollar_rl"},
    {"symbol": "EUR",         "name_fa": "یورو",                      "name_en": "Euro",                    "category": "currency",    "source_key": "price_eur"},
    {"symbol": "AED",         "name_fa": "درهم امارات",              "name_en": "UAE Dirham",              "category": "currency",    "source_key": "price_aed"},
    {"symbol": "GBP",         "name_fa": "پوند انگلیس",             "name_en": "British Pound",           "category": "currency",    "source_key": "price_gbp"},
    # ── مسکن ───────────────────────────────────────────────
    {"symbol": "RE_TEHRAN",   "name_fa": "شاخص مسکن تهران",         "name_en": "Tehran Real Estate Index","category": "real_estate", "source_key": "real_estate_tehran"},
    # ── کریپتو ─────────────────────────────────────────────
    {"symbol": "BTC",         "name_fa": "بیت‌کوین",                 "name_en": "Bitcoin",                 "category": "crypto",      "source_key": "crypto_bitcoin"},
    {"symbol": "ETH",         "name_fa": "اتریوم",                   "name_en": "Ethereum",                "category": "crypto",      "source_key": "crypto_ethereum"},
    {"symbol": "USDT",        "name_fa": "تتر",                      "name_en": "Tether",                  "category": "crypto",      "source_key": "crypto_tether"},
]
