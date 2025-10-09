"""
Trading Operations API Routes for FastAPI service
Handles order management, trade execution, and trading operations
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum
import random

router = APIRouter()

# Enums
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    portfolio_id: str

class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    average_fill_price: Optional[float]
    time_in_force: TimeInForce
    status: OrderStatus
    portfolio_id: str
    created_at: str
    updated_at: str

class Trade(BaseModel):
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    executed_at: str
    portfolio_id: str

class TradingStats(BaseModel):
    total_orders: int
    filled_orders: int
    cancelled_orders: int
    rejected_orders: int
    total_volume: float
    win_rate: float
    avg_hold_time_hours: float
    total_commission: float

@router.post("/orders", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    """Place a new trading order"""
    
    # Validate order
    if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
        raise HTTPException(status_code=400, detail="Price required for limit orders")
    
    if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
        raise HTTPException(status_code=400, detail="Stop price required for stop orders")
    
    # Generate order ID
    order_id = f"order_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
    
    # In production, this would:
    # 1. Validate account/portfolio
    # 2. Check buying power/positions
    # 3. Submit to broker/exchange
    # 4. Store in database
    
    # Simulate order placement
    status = OrderStatus.PENDING
    if order.order_type == OrderType.MARKET:
        # Market orders typically fill immediately
        status = random.choice([OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED])
    
    filled_qty = 0.0
    avg_fill_price = None
    
    if status == OrderStatus.FILLED:
        filled_qty = order.quantity
        avg_fill_price = order.price or 150.0  # Sample fill price
    elif status == OrderStatus.PARTIALLY_FILLED:
        filled_qty = order.quantity * random.uniform(0.3, 0.8)
        avg_fill_price = order.price or 150.0
    
    response = OrderResponse(
        order_id=order_id,
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        filled_quantity=filled_qty,
        remaining_quantity=order.quantity - filled_qty,
        price=order.price,
        stop_price=order.stop_price,
        average_fill_price=avg_fill_price,
        time_in_force=order.time_in_force,
        status=status,
        portfolio_id=order.portfolio_id,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    return response

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    portfolio_id: Optional[str] = Query(None),
    status: Optional[OrderStatus] = Query(None),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """Get trading orders with optional filters"""
    
    # In production, fetch from database with filters
    # Generate sample orders
    orders = []
    
    for i in range(min(limit, 10)):  # Generate up to 10 sample orders
        order_status = status or random.choice(list(OrderStatus))
        order_symbol = symbol or random.choice(["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"])
        
        filled_qty = 0.0
        avg_fill_price = None
        
        if order_status == OrderStatus.FILLED:
            filled_qty = 100.0
            avg_fill_price = 150.0 + random.uniform(-10, 10)
        elif order_status == OrderStatus.PARTIALLY_FILLED:
            filled_qty = random.uniform(30, 80)
            avg_fill_price = 150.0 + random.uniform(-10, 10)
        
        orders.append(OrderResponse(
            order_id=f"order_{int(datetime.utcnow().timestamp())}_{i}",
            symbol=order_symbol,
            side=random.choice(list(OrderSide)),
            order_type=random.choice(list(OrderType)),
            quantity=100.0,
            filled_quantity=filled_qty,
            remaining_quantity=100.0 - filled_qty,
            price=150.0 + random.uniform(-5, 5),
            stop_price=None,
            average_fill_price=avg_fill_price,
            time_in_force=TimeInForce.DAY,
            status=order_status,
            portfolio_id=portfolio_id or f"portfolio_{i}",
            created_at=(datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat(),
            updated_at=datetime.utcnow().isoformat()
        ))
    
    return orders

@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str):
    """Get details of a specific order"""
    
    if not order_id.startswith("order_"):
        raise HTTPException(status_code=404, detail="Order not found")
    
    # In production, fetch from database
    return OrderResponse(
        order_id=order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        filled_quantity=100.0,
        remaining_quantity=0.0,
        price=150.0,
        stop_price=None,
        average_fill_price=149.85,
        time_in_force=TimeInForce.DAY,
        status=OrderStatus.FILLED,
        portfolio_id="portfolio_123",
        created_at="2024-12-01T10:30:00Z",
        updated_at=datetime.utcnow().isoformat()
    )

@router.put("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel a pending order"""
    
    if not order_id.startswith("order_"):
        raise HTTPException(status_code=404, detail="Order not found")
    
    # In production, this would:
    # 1. Check if order can be cancelled
    # 2. Submit cancellation to broker
    # 3. Update order status in database
    
    return {
        "order_id": order_id,
        "status": "cancelled",
        "message": "Order cancellation submitted",
        "cancelled_at": datetime.utcnow().isoformat()
    }

@router.get("/trades", response_model=List[Trade])
async def get_trades(
    portfolio_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """Get executed trades with optional filters"""
    
    # Generate sample trades
    trades = []
    
    for i in range(min(limit, 15)):  # Generate up to 15 sample trades
        trade_symbol = symbol or random.choice(["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"])
        
        trades.append(Trade(
            trade_id=f"trade_{int(datetime.utcnow().timestamp())}_{i}",
            order_id=f"order_{int(datetime.utcnow().timestamp())}_{i}",
            symbol=trade_symbol,
            side=random.choice(list(OrderSide)),
            quantity=random.randint(10, 200),
            price=150.0 + random.uniform(-20, 20),
            commission=random.uniform(1.0, 10.0),
            executed_at=(datetime.utcnow() - timedelta(hours=random.randint(1, 168))).isoformat(),
            portfolio_id=portfolio_id or f"portfolio_{i % 5}"
        ))
    
    return trades

@router.get("/positions", response_model=List[Dict[str, Any]])
async def get_positions(portfolio_id: Optional[str] = Query(None)):
    """Get current positions"""
    
    # Generate sample positions
    positions = []
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    for symbol in symbols:
        quantity = random.randint(10, 500)
        avg_cost = 150.0 + random.uniform(-50, 100)
        current_price = avg_cost + random.uniform(-20, 30)
        market_value = quantity * current_price
        unrealized_pnl = quantity * (current_price - avg_cost)
        
        positions.append({
            "symbol": symbol,
            "quantity": quantity,
            "average_cost": round(avg_cost, 2),
            "current_price": round(current_price, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_percent": round((unrealized_pnl / (quantity * avg_cost)) * 100, 2),
            "day_change": round(random.uniform(-5, 5), 2),
            "day_change_percent": round(random.uniform(-3, 3), 2),
            "portfolio_id": portfolio_id or "portfolio_default"
        })
    
    return positions

@router.get("/stats", response_model=TradingStats)
async def get_trading_stats(
    portfolio_id: Optional[str] = Query(None),
    period: str = Query("1M", pattern="^(1D|1W|1M|3M|6M|1Y|ALL)$")
):
    """Get trading statistics for a period"""
    
    # Generate sample stats based on period
    base_orders = {"1D": 5, "1W": 25, "1M": 100, "3M": 300, "6M": 600, "1Y": 1200, "ALL": 2500}
    total = base_orders.get(period, 100)
    
    return TradingStats(
        total_orders=total,
        filled_orders=int(total * 0.85),
        cancelled_orders=int(total * 0.12),
        rejected_orders=int(total * 0.03),
        total_volume=total * random.uniform(50, 200),
        win_rate=random.uniform(0.55, 0.75),
        avg_hold_time_hours=random.uniform(24, 168),
        total_commission=total * random.uniform(2, 8)
    )

@router.post("/bracket-order", response_model=Dict[str, Any])
async def place_bracket_order(
    symbol: str,
    side: OrderSide, 
    quantity: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    portfolio_id: str
):
    """Place a bracket order (entry + stop loss + take profit)"""
    
    # Generate IDs for the bracket
    parent_order_id = f"bracket_parent_{int(datetime.utcnow().timestamp())}"
    stop_order_id = f"bracket_stop_{int(datetime.utcnow().timestamp())}"
    profit_order_id = f"bracket_profit_{int(datetime.utcnow().timestamp())}"
    
    return {
        "bracket_id": f"bracket_{int(datetime.utcnow().timestamp())}",
        "parent_order_id": parent_order_id,
        "stop_loss_order_id": stop_order_id,
        "take_profit_order_id": profit_order_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "portfolio_id": portfolio_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat()
    }

@router.get("/order-book/{symbol}", response_model=Dict[str, Any])
async def get_order_book(symbol: str, depth: int = Query(10, le=50)):
    """Get order book data for a symbol"""
    
    # Generate sample order book
    base_price = 150.0
    
    bids = []
    asks = []
    
    for i in range(depth):
        bid_price = base_price - (i + 1) * 0.01
        ask_price = base_price + (i + 1) * 0.01
        
        bids.append({
            "price": round(bid_price, 2),
            "size": random.randint(100, 10000),
            "orders": random.randint(1, 50)
        })
        
        asks.append({
            "price": round(ask_price, 2), 
            "size": random.randint(100, 10000),
            "orders": random.randint(1, 50)
        })
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "bids": bids,
        "asks": asks,
        "spread": round(asks[0]["price"] - bids[0]["price"], 2),
        "mid_price": round((asks[0]["price"] + bids[0]["price"]) / 2, 2)
    }

@router.get("/health", response_model=Dict[str, Any])
async def trading_health():
    """Health check for trading service"""
    return {
        "status": "healthy",
        "service": "trading",
        "timestamp": datetime.utcnow().isoformat(),
        "pending_orders": 23,
        "orders_today": 156,
        "trading_enabled": True,
        "market_status": "open",
        "last_trade": (datetime.utcnow() - timedelta(seconds=30)).isoformat()
    } 