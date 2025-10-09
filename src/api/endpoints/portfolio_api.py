from fastapi import APIRouter
from typing import List, Dict

portfolio_router = APIRouter()

# --- Mock Data (to be replaced by database calls) ---

mock_portfolios = [
  { "id": 1, "name": "Growth Stocks", "description": "High-risk, high-reward tech stocks.", "initial_capital": 100000, "current_value": 125000, "created_at": "2022-01-15T09:30:00Z" },
  { "id": 2, "name": "Dividend Income", "description": "Stable, income-generating assets.", "initial_capital": 250000, "current_value": 265000, "created_at": "2021-06-20T14:00:00Z" },
]

mock_positions: Dict[int, List[Dict]] = {
  1: [
    { "id": 101, "symbol": "AAPL", "quantity": 50, "average_price": 150, "current_price": 175, "market_value": 8750, "unrealized_pnl": 1250 },
    { "id": 102, "symbol": "GOOGL", "quantity": 20, "average_price": 2800, "current_price": 2950, "market_value": 59000, "unrealized_pnl": 3000 },
    { "id": 103, "symbol": "TSLA", "quantity": 15, "average_price": 800, "current_price": 750, "market_value": 11250, "unrealized_pnl": -750 },
  ],
  2: [
    { "id": 201, "symbol": "JNJ", "quantity": 100, "average_price": 160, "current_price": 170, "market_value": 17000, "unrealized_pnl": 1000 },
    { "id": 202, "symbol": "PG", "quantity": 150, "average_price": 140, "current_price": 145, "market_value": 21750, "unrealized_pnl": 750 },
    { "id": 203, "symbol": "KO", "quantity": 200, "average_price": 55, "current_price": 60, "market_value": 12000, "unrealized_pnl": 1000 },
  ]
}

# --- API Endpoints ---

@portfolio_router.get("/")
def get_all_portfolios():
    """
    Retrieves a list of all portfolios.
    """
    return mock_portfolios

@portfolio_router.get("/{portfolio_id}/positions")
def get_portfolio_positions(portfolio_id: int):
    """
    Retrieves all positions for a specific portfolio.
    """
    return mock_positions.get(portfolio_id, []) 