"""
Real-Time Execution Manager
Advanced Order Management and Execution System

This module handles:
- Intelligent order routing
- Execution algorithms (TWAP, VWAP, Implementation Shortfall)
- Market impact minimization
- Order lifecycle management
- Slippage analysis
- Fill quality assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
from decimal import Decimal, ROUND_HALF_UP
import json

from ..core.cache import CacheManager
from ..database.postgres_connection import get_db
from ..realtime.websockets import WebSocketManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

@dataclass
class OrderRequest:
    """Order request specification"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Algorithm parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Risk parameters
    max_participation_rate: float = 0.20  # Max 20% of volume
    max_market_impact: float = 0.005  # Max 0.5% impact
    urgency: float = 0.5  # 0 = patient, 1 = aggressive

@dataclass
class Fill:
    """Individual fill record"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    venue: str = "primary"

@dataclass
class Order:
    """Complete order with execution tracking"""
    order_id: str
    request: OrderRequest
    status: OrderStatus
    submitted_time: datetime
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    
    # Execution metrics
    slippage: float = 0.0
    market_impact: float = 0.0
    implementation_shortfall: float = 0.0
    
    # Timing
    first_fill_time: Optional[datetime] = None
    last_fill_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None

class ExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, order: Order, market_data: Dict) -> List[OrderRequest]:
        """Generate child orders for execution"""
        raise NotImplementedError

class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price algorithm"""
    
    def __init__(self):
        super().__init__("TWAP")
    
    async def execute(self, order: Order, market_data: Dict) -> List[OrderRequest]:
        """Split order across time intervals"""
        
        params = order.request.algorithm_params
        duration_minutes = params.get('duration_minutes', 60)
        num_slices = params.get('num_slices', 12)
        
        slice_size = order.remaining_quantity / num_slices
        slice_interval = duration_minutes / num_slices
        
        child_orders = []
        
        for i in range(num_slices):
            child_order = OrderRequest(
                symbol=order.request.symbol,
                side=order.request.side,
                quantity=slice_size,
                order_type=OrderType.LIMIT,
                price=market_data.get('mid_price'),  # Use mid price
                time_in_force=TimeInForce.IOC
            )
            child_orders.append(child_order)
        
        return child_orders

class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm"""
    
    def __init__(self):
        super().__init__("VWAP")
    
    async def execute(self, order: Order, market_data: Dict) -> List[OrderRequest]:
        """Split order based on historical volume profile"""
        
        params = order.request.algorithm_params
        volume_profile = params.get('volume_profile', [])
        
        if not volume_profile:
            # Default uniform distribution
            volume_profile = [1.0 / 12] * 12  # 12 intervals
        
        child_orders = []
        
        for i, volume_weight in enumerate(volume_profile):
            slice_size = order.remaining_quantity * volume_weight
            
            if slice_size > 0:
                child_order = OrderRequest(
                    symbol=order.request.symbol,
                    side=order.request.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    price=market_data.get('mid_price'),
                    time_in_force=TimeInForce.IOC
                )
                child_orders.append(child_order)
        
        return child_orders

class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall algorithm (Almgren-Chriss)"""
    
    def __init__(self):
        super().__init__("Implementation Shortfall")
    
    async def execute(self, order: Order, market_data: Dict) -> List[OrderRequest]:
        """Optimize trade-off between market impact and timing risk"""
        
        params = order.request.algorithm_params
        
        # Market impact parameters
        temporary_impact = params.get('temporary_impact', 0.01)
        permanent_impact = params.get('permanent_impact', 0.005)
        volatility = params.get('volatility', 0.02)
        
        # Risk aversion parameter
        risk_aversion = params.get('risk_aversion', 1e-6)
        
        # Calculate optimal trajectory
        duration_minutes = params.get('duration_minutes', 60)
        num_slices = params.get('num_slices', 12)
        
        # Simplified optimal execution rate (exponential decay)
        decay_rate = np.sqrt(risk_aversion * volatility**2 / temporary_impact)
        
        child_orders = []
        remaining = order.remaining_quantity
        
        for i in range(num_slices):
            t = i / num_slices
            
            # Optimal execution rate
            execution_rate = decay_rate * np.exp(-decay_rate * t)
            slice_size = min(remaining * execution_rate, remaining / (num_slices - i))
            
            if slice_size > 0:
                child_order = OrderRequest(
                    symbol=order.request.symbol,
                    side=order.request.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    price=market_data.get('mid_price'),
                    time_in_force=TimeInForce.IOC
                )
                child_orders.append(child_order)
                remaining -= slice_size
        
        return child_orders

class ExecutionManager:
    """Advanced Order Management and Execution System"""
    
    def __init__(self, cache_manager: CacheManager, websocket_manager: WebSocketManager):
        self.cache_manager = cache_manager
        self.websocket_manager = websocket_manager
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        
        # Execution algorithms
        self.algorithms = {
            OrderType.TWAP: TWAPAlgorithm(),
            OrderType.VWAP: VWAPAlgorithm(),
            "implementation_shortfall": ImplementationShortfallAlgorithm()
        }
        
        # Market data
        self.market_data: Dict[str, Dict] = {}
        
        # Execution metrics
        self.daily_metrics = {
            'total_trades': 0,
            'total_volume': 0.0,
            'average_slippage': 0.0,
            'average_market_impact': 0.0,
            'fill_rate': 0.0
        }
        
        # Risk limits
        self.max_order_size = 1000000  # $1M max order
        self.max_daily_volume = 10000000  # $10M daily limit
        self.max_position_concentration = 0.1  # 10% max position
    
    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit new order for execution"""
        
        try:
            # Validate order
            validation_result = await self._validate_order(order_request)
            if not validation_result['valid']:
                raise ValueError(f"Order validation failed: {validation_result['reason']}")
            
            # Create order object
            order = Order(
                order_id=str(uuid.uuid4()),
                request=order_request,
                status=OrderStatus.PENDING,
                submitted_time=datetime.utcnow(),
                remaining_quantity=order_request.quantity
            )
            
            # Store order
            self.active_orders[order.order_id] = order
            
            # Start execution process
            asyncio.create_task(self._execute_order(order))
            
            # Log order submission
            logger.info(f"Order submitted: {order.order_id} - {order_request.symbol} {order_request.side.value} {order_request.quantity}")
            
            # Cache order
            await self.cache_manager.set(
                f"order:{order.order_id}",
                order.__dict__,
                expire=86400  # 24 hours
            )
            
            return order.order_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise
    
    async def _validate_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Comprehensive order validation"""
        
        # Size validation
        order_value = order_request.quantity * (order_request.price or 100)  # Estimate
        
        if order_value > self.max_order_size:
            return {
                'valid': False,
                'reason': f'Order size ({order_value:,.0f}) exceeds maximum ({self.max_order_size:,.0f})'
            }
        
        # Daily volume check
        daily_volume = await self._get_daily_volume()
        if daily_volume + order_value > self.max_daily_volume:
            return {
                'valid': False,
                'reason': f'Order would exceed daily volume limit'
            }
        
        # Symbol validation
        if not await self._is_tradeable_symbol(order_request.symbol):
            return {
                'valid': False,
                'reason': f'Symbol {order_request.symbol} is not tradeable'
            }
        
        # Price validation for limit orders
        if order_request.order_type == OrderType.LIMIT and not order_request.price:
            return {
                'valid': False,
                'reason': 'Limit orders must specify price'
            }
        
        return {'valid': True, 'reason': 'Order validation passed'}
    
    async def _execute_order(self, order: Order):
        """Execute order using appropriate algorithm"""
        
        try:
            order.status = OrderStatus.SUBMITTED
            
            # Get current market data
            market_data = await self._get_market_data(order.request.symbol)
            
            # Route to appropriate execution algorithm
            if order.request.order_type in [OrderType.MARKET, OrderType.LIMIT]:
                await self._execute_simple_order(order, market_data)
            elif order.request.order_type in self.algorithms:
                await self._execute_algorithmic_order(order, market_data)
            else:
                raise ValueError(f"Unsupported order type: {order.request.order_type}")
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            await self._finalize_order(order)
    
    async def _execute_simple_order(self, order: Order, market_data: Dict):
        """Execute market or limit order"""
        
        # Simulate order execution (in practice, route to exchange/broker)
        if order.request.order_type == OrderType.MARKET:
            fill_price = market_data.get('ask_price') if order.request.side == OrderSide.BUY else market_data.get('bid_price')
        else:
            fill_price = order.request.price
        
        # Simulate partial fills for large orders
        max_fill_size = market_data.get('available_size', order.remaining_quantity)
        fill_quantity = min(order.remaining_quantity, max_fill_size)
        
        if fill_quantity > 0:
            await self._process_fill(order, fill_quantity, fill_price)
    
    async def _execute_algorithmic_order(self, order: Order, market_data: Dict):
        """Execute order using algorithmic strategy"""
        
        algorithm = self.algorithms[order.request.order_type]
        child_orders = await algorithm.execute(order, market_data)
        
        # Execute child orders sequentially with delays
        for i, child_order in enumerate(child_orders):
            if order.remaining_quantity <= 0:
                break
                
            # Adjust child order size if needed
            child_order.quantity = min(child_order.quantity, order.remaining_quantity)
            
            # Simulate execution delay
            if i > 0:
                delay_seconds = order.request.algorithm_params.get('slice_interval_seconds', 30)
                await asyncio.sleep(delay_seconds)
            
            # Execute child order
            fill_price = await self._get_execution_price(child_order, market_data)
            fill_quantity = await self._simulate_fill(child_order, market_data)
            
            if fill_quantity > 0:
                await self._process_fill(order, fill_quantity, fill_price)
    
    async def _process_fill(self, order: Order, quantity: float, price: float):
        """Process order fill"""
        
        # Create fill record
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.request.symbol,
            side=order.request.side,
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow(),
            commission=self._calculate_commission(quantity, price)
        )
        
        # Update order
        order.fills.append(fill)
        order.filled_quantity += quantity
        order.remaining_quantity -= quantity
        
        # Update average fill price
        total_value = sum(f.quantity * f.price for f in order.fills)
        order.average_fill_price = total_value / order.filled_quantity
        
        # Update timing
        if not order.first_fill_time:
            order.first_fill_time = fill.timestamp
        order.last_fill_time = fill.timestamp
        
        # Update status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.completion_time = fill.timestamp
            await self._finalize_order(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Calculate execution metrics
        await self._calculate_execution_metrics(order)
        
        # Notify via WebSocket
        await self._notify_fill(fill)
        
        logger.info(f"Order {order.order_id} filled: {quantity} @ {price}")
    
    async def _calculate_execution_metrics(self, order: Order):
        """Calculate execution quality metrics"""
        
        if not order.fills:
            return
        
        # Get benchmark price (arrival price)
        arrival_price = await self._get_arrival_price(order)
        
        # Slippage calculation
        if order.request.side == OrderSide.BUY:
            order.slippage = (order.average_fill_price - arrival_price) / arrival_price
        else:
            order.slippage = (arrival_price - order.average_fill_price) / arrival_price
        
        # Market impact (simplified)
        order.market_impact = abs(order.slippage) * 0.5  # Assume 50% is market impact
        
        # Implementation shortfall
        order.implementation_shortfall = order.slippage + order.market_impact
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        # Simplified commission structure
        trade_value = quantity * price
        commission_rate = 0.0005  # 5 basis points
        min_commission = 1.0
        
        commission = max(trade_value * commission_rate, min_commission)
        return commission
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get current market data for symbol"""
        
        # Try cache first
        cache_key = f"market_data:{symbol}"
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            return cached_data
        
        # Simulate market data (in practice, get from market data feed)
        market_data = {
            'symbol': symbol,
            'bid_price': 100.0,
            'ask_price': 100.1,
            'mid_price': 100.05,
            'bid_size': 1000,
            'ask_size': 1000,
            'available_size': 500,
            'last_price': 100.02,
            'volume': 100000,
            'timestamp': datetime.utcnow()
        }
        
        # Cache for short time
        await self.cache_manager.set(cache_key, market_data, expire=1)
        
        return market_data
    
    async def _get_execution_price(self, order_request: OrderRequest, market_data: Dict) -> float:
        """Determine execution price for order"""
        
        if order_request.order_type == OrderType.MARKET:
            return market_data['ask_price'] if order_request.side == OrderSide.BUY else market_data['bid_price']
        elif order_request.order_type == OrderType.LIMIT:
            return order_request.price
        else:
            return market_data['mid_price']
    
    async def _simulate_fill(self, order_request: OrderRequest, market_data: Dict) -> float:
        """Simulate order fill based on market conditions"""
        
        # Simulate fill probability based on order aggressiveness
        available_liquidity = market_data.get('available_size', 1000)
        max_participation = available_liquidity * order_request.max_participation_rate
        
        # Fill quantity (simplified simulation)
        fill_quantity = min(
            order_request.quantity,
            max_participation,
            available_liquidity * np.random.uniform(0.5, 1.0)  # Random fill rate
        )
        
        return max(0, fill_quantity)
    
    async def _get_arrival_price(self, order: Order) -> float:
        """Get arrival price for benchmark calculation"""
        
        # Use cached arrival price or approximate
        cache_key = f"arrival_price:{order.order_id}"
        cached_price = await self.cache_manager.get(cache_key)
        
        if cached_price:
            return cached_price
        
        # Approximate with current mid price
        market_data = await self._get_market_data(order.request.symbol)
        arrival_price = market_data['mid_price']
        
        # Cache arrival price
        await self.cache_manager.set(cache_key, arrival_price, expire=86400)
        
        return arrival_price
    
    async def _notify_fill(self, fill: Fill):
        """Send fill notification via WebSocket"""
        
        notification = {
            'type': 'fill',
            'data': {
                'fill_id': fill.fill_id,
                'order_id': fill.order_id,
                'symbol': fill.symbol,
                'side': fill.side.value,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp.isoformat(),
                'commission': fill.commission
            }
        }
        
        await self.websocket_manager.broadcast_to_channel(
            'executions',
            json.dumps(notification)
        )
    
    async def _finalize_order(self, order: Order):
        """Finalize completed order"""
        
        # Move to completed orders
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        
        self.completed_orders[order.order_id] = order
        
        # Update daily metrics
        self.daily_metrics['total_trades'] += 1
        self.daily_metrics['total_volume'] += order.filled_quantity * order.average_fill_price
        
        # Store in database
        await self._store_order_in_db(order)
        
        logger.info(f"Order {order.order_id} finalized with status {order.status.value}")
    
    async def _store_order_in_db(self, order: Order):
        """Store order in database"""
        
        try:
            async with get_db_connection() as conn:
                # Store order
                await conn.execute("""
                    INSERT INTO orders (
                        order_id, symbol, side, quantity, order_type, status,
                        submitted_time, completion_time, filled_quantity,
                        average_fill_price, slippage, market_impact
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, (
                    order.order_id,
                    order.request.symbol,
                    order.request.side.value,
                    order.request.quantity,
                    order.request.order_type.value,
                    order.status.value,
                    order.submitted_time,
                    order.completion_time,
                    order.filled_quantity,
                    order.average_fill_price,
                    order.slippage,
                    order.market_impact
                ))
                
                # Store fills
                for fill in order.fills:
                    await conn.execute("""
                        INSERT INTO fills (
                            fill_id, order_id, symbol, side, quantity,
                            price, timestamp, commission
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, (
                        fill.fill_id,
                        fill.order_id,
                        fill.symbol,
                        fill.side.value,
                        fill.quantity,
                        fill.price,
                        fill.timestamp,
                        fill.commission
                    ))
        
        except Exception as e:
            logger.error(f"Error storing order in database: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.completion_time = datetime.utcnow()
        
        await self._finalize_order(order)
        
        logger.info(f"Order {order_id} cancelled")
        return True
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get current order status"""
        
        order = self.active_orders.get(order_id) or self.completed_orders.get(order_id)
        
        if not order:
            return None
        
        return {
            'order_id': order.order_id,
            'symbol': order.request.symbol,
            'side': order.request.side.value,
            'quantity': order.request.quantity,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'slippage': order.slippage,
            'market_impact': order.market_impact,
            'submitted_time': order.submitted_time.isoformat(),
            'completion_time': order.completion_time.isoformat() if order.completion_time else None
        }
    
    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        
        completed_orders = list(self.completed_orders.values())
        filled_orders = [o for o in completed_orders if o.status == OrderStatus.FILLED]
        
        if not filled_orders:
            return self.daily_metrics
        
        # Calculate averages
        avg_slippage = np.mean([o.slippage for o in filled_orders])
        avg_market_impact = np.mean([o.market_impact for o in filled_orders])
        fill_rate = len(filled_orders) / len(completed_orders)
        
        return {
            **self.daily_metrics,
            'average_slippage': avg_slippage,
            'average_market_impact': avg_market_impact,
            'fill_rate': fill_rate,
            'total_orders': len(completed_orders),
            'filled_orders': len(filled_orders)
        }
    
    async def _get_daily_volume(self) -> float:
        """Get today's trading volume"""
        return self.daily_metrics['total_volume']
    
    async def _is_tradeable_symbol(self, symbol: str) -> bool:
        """Check if symbol is tradeable"""
        # Simplified check - in practice, validate against supported symbols
        return True 