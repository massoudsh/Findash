from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TradeBase(BaseModel):
    symbol: str
    trade_type: str
    quantity: float
    price: float
    notes: Optional[str] = None

class TradeCreate(TradeBase):
    pass

class TradeRead(TradeBase):
    id: int
    portfolio_id: int
    trade_time: datetime

    class Config:
        orm_mode = True 