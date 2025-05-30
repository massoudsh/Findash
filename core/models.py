"""Data validation models using Pydantic."""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator, HttpUrl
from decimal import Decimal
from enum import Enum

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class MarketData(BaseModel):
    """Market data validation model."""
    symbol: str = Field(..., min_length=1, max_length=10)
    price: Decimal = Field(..., gt=0)
    volume: int = Field(..., ge=0)
    timestamp: datetime
    exchange: str
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate stock symbol format."""
        if not v.isalpha():
            raise ValueError("Symbol must contain only letters")
        return v.upper()

class TradingSignal(BaseModel):
    """Trading signal validation model."""
    symbol: str
    side: OrderSide
    strength: float = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    indicators: Dict[str, float]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class NewsArticle(BaseModel):
    """News article validation model."""
    title: str = Field(..., min_length=1)
    content: str
    url: HttpUrl
    source: str
    published_at: datetime
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    
    @validator('content')
    def validate_content(cls, v):
        """Validate article content."""
        if len(v.split()) < 10:
            raise ValueError("Content too short")
        return v

class SocialMetrics(BaseModel):
    """Social media metrics validation model."""
    platform: str
    mentions: int = Field(..., ge=0)
    sentiment: float = Field(..., ge=-1, le=1)
    engagement: int = Field(..., ge=0)
    timestamp: datetime
    trending_tags: List[str] = Field(default_factory=list)

class PortfolioPosition(BaseModel):
    """Portfolio position validation model."""
    symbol: str
    quantity: Decimal = Field(..., ge=0)
    average_price: Decimal = Field(..., gt=0)
    current_price: Decimal = Field(..., gt=0)
    market_value: Decimal = Field(..., ge=0)
    unrealized_pl: Decimal
    
    @validator('market_value')
    def validate_market_value(cls, v, values):
        """Validate market value calculation."""
        if 'quantity' in values and 'current_price' in values:
            expected = values['quantity'] * values['current_price']
            if abs(v - expected) > Decimal('0.01'):
                raise ValueError("Market value doesn't match quantity * price")
        return v 