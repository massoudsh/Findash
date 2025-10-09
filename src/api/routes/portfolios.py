"""
Portfolio Management API Routes for FastAPI service
Handles portfolio operations, positions, and performance tracking
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import random

router = APIRouter()

# Pydantic models
class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    initial_capital: float = Field(..., gt=0)
    risk_tolerance: str = Field(..., pattern="^(conservative|moderate|aggressive)$")

class PortfolioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    initial_capital: float
    current_value: float
    cash_balance: float
    total_return: float
    total_return_percent: float
    risk_tolerance: str
    created_at: str
    updated_at: str

class Position(BaseModel):
    symbol: str
    quantity: float
    average_cost: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float

class PortfolioDetail(BaseModel):
    portfolio: PortfolioResponse
    positions: List[Position]
    allocation: Dict[str, float]
    performance_metrics: Dict[str, float]

@router.get("/", response_model=List[PortfolioResponse])
async def list_portfolios():
    """Get list of all portfolios"""
    
    # In production, fetch from database
    # For now, return sample portfolios
    portfolios = [
        PortfolioResponse(
            id="portfolio_1",
            name="Growth Portfolio", 
            description="Focused on high-growth technology stocks",
            initial_capital=100000.0,
            current_value=112500.0,
            cash_balance=15000.0,
            total_return=12500.0,
            total_return_percent=12.5,
            risk_tolerance="aggressive",
            created_at="2024-01-15T10:00:00Z",
            updated_at=datetime.utcnow().isoformat()
        ),
        PortfolioResponse(
            id="portfolio_2",
            name="Conservative Income",
            description="Dividend-focused with lower volatility",
            initial_capital=250000.0,
            current_value=267500.0,
            cash_balance=35000.0,
            total_return=17500.0,
            total_return_percent=7.0,
            risk_tolerance="conservative",
            created_at="2024-02-01T14:30:00Z",
            updated_at=datetime.utcnow().isoformat()
        )
    ]
    
    return portfolios

@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(portfolio_data: PortfolioCreate):
    """Create a new portfolio"""
    
    portfolio_id = f"portfolio_{int(datetime.utcnow().timestamp())}"
    
    # In production, save to database
    new_portfolio = PortfolioResponse(
        id=portfolio_id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        initial_capital=portfolio_data.initial_capital,
        current_value=portfolio_data.initial_capital,  # Initially same as capital
        cash_balance=portfolio_data.initial_capital,   # All cash initially
        total_return=0.0,
        total_return_percent=0.0,
        risk_tolerance=portfolio_data.risk_tolerance,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    return new_portfolio

@router.get("/{portfolio_id}", response_model=PortfolioDetail)
async def get_portfolio_detail(portfolio_id: str):
    """Get detailed information about a specific portfolio"""
    
    # In production, fetch from database
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Sample portfolio data
    portfolio = PortfolioResponse(
        id=portfolio_id,
        name="Sample Portfolio",
        description="Sample portfolio for demonstration",
        initial_capital=100000.0,
        current_value=115000.0,
        cash_balance=25000.0,
        total_return=15000.0,
        total_return_percent=15.0,
        risk_tolerance="moderate",
        created_at="2024-01-15T10:00:00Z",
        updated_at=datetime.utcnow().isoformat()
    )
    
    # Sample positions
    positions = [
        Position(
            symbol="AAPL",
            quantity=100,
            average_cost=150.0,
            market_value=17500.0,
            unrealized_pnl=2500.0,
            unrealized_pnl_percent=16.67,
            weight=15.22
        ),
        Position(
            symbol="TSLA",
            quantity=50,
            average_cost=200.0,
            market_value=12000.0,
            unrealized_pnl=2000.0,
            unrealized_pnl_percent=20.0,
            weight=10.43
        ),
        Position(
            symbol="MSFT",
            quantity=200,
            average_cost=250.0,
            market_value=55000.0,
            unrealized_pnl=5000.0,
            unrealized_pnl_percent=10.0,
            weight=47.83
        )
    ]
    
    # Sample allocation
    allocation = {
        "Technology": 73.0,
        "Cash": 21.7,
        "Other": 5.3
    }
    
    # Sample performance metrics
    performance_metrics = {
        "sharpe_ratio": 1.25,
        "beta": 1.15,
        "alpha": 0.03,
        "max_drawdown": -0.08,
        "volatility": 0.18,
        "var_95": -0.025,
        "sortino_ratio": 1.45
    }
    
    return PortfolioDetail(
        portfolio=portfolio,
        positions=positions,
        allocation=allocation,
        performance_metrics=performance_metrics
    )

@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(portfolio_id: str, portfolio_data: PortfolioCreate):
    """Update portfolio information"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # In production, update in database
    updated_portfolio = PortfolioResponse(
        id=portfolio_id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        initial_capital=portfolio_data.initial_capital,
        current_value=portfolio_data.initial_capital * 1.1,  # Sample 10% growth
        cash_balance=portfolio_data.initial_capital * 0.2,   # 20% cash
        total_return=portfolio_data.initial_capital * 0.1,
        total_return_percent=10.0,
        risk_tolerance=portfolio_data.risk_tolerance,
        created_at="2024-01-15T10:00:00Z",
        updated_at=datetime.utcnow().isoformat()
    )
    
    return updated_portfolio

@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: str):
    """Delete a portfolio"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # In production, delete from database
    return {"message": f"Portfolio {portfolio_id} deleted successfully"}

@router.get("/{portfolio_id}/positions", response_model=List[Position])
async def get_portfolio_positions(portfolio_id: str):
    """Get all positions in a portfolio"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Sample positions
    positions = [
        Position(
            symbol="AAPL",
            quantity=100,
            average_cost=150.0,
            market_value=17500.0,
            unrealized_pnl=2500.0,
            unrealized_pnl_percent=16.67,
            weight=35.0
        ),
        Position(
            symbol="GOOGL",
            quantity=25,
            average_cost=2800.0,
            market_value=32500.0,
            unrealized_pnl=2500.0,
            unrealized_pnl_percent=8.33,
            weight=65.0
        )
    ]
    
    return positions

@router.get("/{portfolio_id}/performance", response_model=Dict[str, Any])
async def get_portfolio_performance(
    portfolio_id: str,
    period: str = Query("1M", pattern="^(1D|1W|1M|3M|6M|1Y|YTD|ALL)$")
):
    """Get portfolio performance metrics for a specific period"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Generate sample performance data based on period
    periods = {
        "1D": {"return": 0.5, "volatility": 0.12},
        "1W": {"return": 2.1, "volatility": 0.15},
        "1M": {"return": 8.3, "volatility": 0.18},
        "3M": {"return": 15.7, "volatility": 0.22},
        "6M": {"return": 23.4, "volatility": 0.25},
        "1Y": {"return": 35.2, "volatility": 0.28},
        "YTD": {"return": 18.9, "volatility": 0.20},
        "ALL": {"return": 45.6, "volatility": 0.24}
    }
    
    period_data = periods.get(period, periods["1M"])
    
    return {
        "period": period,
        "total_return": period_data["return"],
        "volatility": period_data["volatility"],
        "sharpe_ratio": period_data["return"] / period_data["volatility"],
        "max_drawdown": -random.uniform(0.05, 0.15),
        "beta": random.uniform(0.8, 1.3),
        "alpha": random.uniform(-0.02, 0.05),
        "var_95": -random.uniform(0.02, 0.04),
        "benchmark_return": period_data["return"] * 0.8,  # Assume outperformance
        "excess_return": period_data["return"] * 0.2,
        "win_rate": random.uniform(0.55, 0.75),
        "profit_factor": random.uniform(1.2, 1.8),
        "calmar_ratio": random.uniform(1.0, 2.5)
    }

@router.get("/{portfolio_id}/allocation", response_model=Dict[str, Any])
async def get_portfolio_allocation(portfolio_id: str):
    """Get portfolio asset allocation breakdown"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return {
        "by_sector": {
            "Technology": 45.2,
            "Healthcare": 18.7,
            "Financial": 15.3,
            "Consumer Discretionary": 12.1,
            "Cash": 8.7
        },
        "by_asset_class": {
            "Equities": 85.5,
            "Fixed Income": 5.8,
            "Cash": 8.7
        },
        "by_geography": {
            "US": 78.2,
            "International Developed": 13.1,
            "Emerging Markets": 8.7
        },
        "by_market_cap": {
            "Large Cap": 65.4,
            "Mid Cap": 20.8,
            "Small Cap": 13.8
        },
        "top_holdings": [
            {"symbol": "AAPL", "weight": 8.5},
            {"symbol": "MSFT", "weight": 7.2},
            {"symbol": "GOOGL", "weight": 6.8},
            {"symbol": "AMZN", "weight": 5.9},
            {"symbol": "TSLA", "weight": 4.6}
        ]
    }

@router.get("/{portfolio_id}/risk-metrics", response_model=Dict[str, Any])
async def get_portfolio_risk_metrics(portfolio_id: str):
    """Get comprehensive risk metrics for the portfolio"""
    
    if not portfolio_id.startswith("portfolio_"):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return {
        "value_at_risk": {
            "var_95_1d": -0.024,
            "var_99_1d": -0.035,
            "var_95_10d": -0.076,
            "expected_shortfall_95": -0.031
        },
        "volatility_metrics": {
            "annualized_volatility": 0.185,
            "downside_deviation": 0.132,
            "upside_capture": 1.05,
            "downside_capture": 0.92
        },
        "correlation_metrics": {
            "correlation_with_sp500": 0.85,
            "correlation_with_nasdaq": 0.91,
            "tracking_error": 0.045
        },
        "concentration_risk": {
            "herfindahl_index": 0.12,
            "top_5_concentration": 0.38,
            "effective_number_of_stocks": 25.7
        },
        "liquidity_metrics": {
            "portfolio_liquidity_score": 8.5,
            "days_to_liquidate": 2.3,
            "bid_ask_impact": 0.008
        },
        "stress_test_scenarios": {
            "market_crash_2008": -0.42,
            "covid_crash_2020": -0.28,
            "dot_com_bubble_2000": -0.35,
            "interest_rate_shock": -0.15
        }
    }

@router.get("/health", response_model=Dict[str, Any])
async def portfolio_health():
    """Health check for portfolio service"""
    return {
        "status": "healthy",
        "service": "portfolios",
        "timestamp": datetime.utcnow().isoformat(),
        "total_portfolios": 147,  # Sample count
        "active_positions": 1245,
        "total_aum": 25600000.0,  # Assets under management
        "last_updated": datetime.utcnow().isoformat()
    } 