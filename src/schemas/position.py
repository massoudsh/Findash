from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PositionBase(BaseModel):
    symbol: str
    quantity: float
    average_price: float

class PositionCreate(PositionBase):
    pass

class PositionRead(PositionBase):
    id: int
    portfolio_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 