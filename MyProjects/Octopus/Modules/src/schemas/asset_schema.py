"""
Pydantic schemas for Iranian Assets API
Covers: gold, silver, currency, real-estate, crypto
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AssetCategory(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    CURRENCY = "currency"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"


class AssetBase(BaseModel):
    symbol: str
    name_fa: str
    name_en: str
    category: AssetCategory
    unit: str = "تومان"


class AssetPrice(BaseModel):
    symbol: str
    price: float
    price_toman: float
    change_24h: float          # مقدار تغییر
    change_percent_24h: float  # درصد تغییر
    high_24h: float
    low_24h: float
    updated_at: str


class AssetDetail(AssetBase):
    price: float
    price_toman: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    updated_at: str
    source: str


class AssetHistory(BaseModel):
    symbol: str
    interval: str              # 1d, 1w, 1m, 3m, 1y
    data: list[dict]           # [{timestamp, open, high, low, close, volume}]


class AssetListResponse(BaseModel):
    assets: list[AssetDetail]
    usd_to_toman: float
    last_updated: str


class PortfolioAssetCreate(BaseModel):
    symbol: str
    quantity: float
    buy_price: float
    buy_date: str
    notes: Optional[str] = None


class PortfolioAsset(PortfolioAssetCreate):
    id: int
    current_price: float
    current_value: float
    profit_loss: float
    profit_loss_percent: float

    class Config:
        from_attributes = True
