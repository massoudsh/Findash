"""Portfolio Snapshot schemas for Quantum Trading Matrix™"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class PortfolioSnapshotBase(BaseModel):
    """Base portfolio snapshot schema"""
    portfolio_id: int
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    cash_balance: float = Field(..., ge=0, description="Cash balance")
    positions_value: float = Field(..., ge=0, description="Total positions value")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    realized_pnl: float = Field(default=0.0, description="Realized P&L")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PortfolioSnapshotCreate(PortfolioSnapshotBase):
    """Schema for creating portfolio snapshots"""
    pass


class PortfolioSnapshotUpdate(BaseModel):
    """Schema for updating portfolio snapshots"""
    total_value: Optional[float] = Field(None, gt=0)
    cash_balance: Optional[float] = Field(None, ge=0)
    positions_value: Optional[float] = Field(None, ge=0)
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PortfolioSnapshotRead(PortfolioSnapshotBase):
    """Schema for reading portfolio snapshots"""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True 