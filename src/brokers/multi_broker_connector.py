"""
Multi-Broker Connector
Unified interface for multiple trading platforms

This module handles:
- Multi-broker connectivity and authentication
- Order routing and execution optimization
- Real-time market data aggregation
- Position and account management across brokers
- Failover and redundancy
- Commission optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    TD_AMERITRADE = "td_ameritrade"
    CHARLES_SCHWAB = "charles_schwab"
    ROBINHOOD = "robinhood"
    WEBULL = "webull"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class AssetType(Enum):
    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"

@dataclass
class BrokerCredentials:
    """Broker authentication credentials"""
    broker_type: BrokerType
    api_key: str
    secret_key: str
    account_id: Optional[str] = None
    paper_trading: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderRequest:
    """Unified order request structure"""
    symbol: str
    quantity: float
    side: str  # "buy" or "sell"
    order_type: OrderType
    
    # Optional parameters
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    asset_type: AssetType = AssetType.STOCK
    
    # Execution preferences
    preferred_broker: Optional[BrokerType] = None
    max_commission: Optional[float] = None
    require_all_or_none: bool = False

@dataclass
class OrderResponse:
    """Unified order response"""
    order_id: str
    broker_order_id: str
    broker_type: BrokerType
    symbol: str
    quantity: float
    side: str
    order_type: OrderType
    status: OrderStatus
    
    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Error information
    error_message: Optional[str] = None

@dataclass
class Position:
    """Unified position representation"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    broker_type: BrokerType
    asset_type: AssetType = AssetType.STOCK

@dataclass
class AccountInfo:
    """Unified account information"""
    broker_type: BrokerType
    account_id: str
    buying_power: float
    total_value: float
    cash_balance: float
    day_trading_buying_power: Optional[float] = None
    pattern_day_trader: bool = False

class BrokerInterface(ABC):
    """Abstract base class for broker implementations"""
    
    def __init__(self, credentials: BrokerCredentials):
        self.credentials = credentials
        self.connected = False
        self.last_error: Optional[str] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        pass

class InteractiveBrokerConnector(BrokerInterface):
    """Interactive Brokers connector"""
    
    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.client = None
        self.next_order_id = 1
    
    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway"""
        try:
            # Mock connection for demo
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info("Connected to Interactive Brokers")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"IB connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from IB"""
        if self.connected:
            self.connected = False
            logger.info("Disconnected from Interactive Brokers")
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit order to IB"""
        if not self.connected:
            raise ConnectionError("Not connected to Interactive Brokers")
        
        # Generate order ID
        order_id = f"IB_{self.next_order_id}"
        self.next_order_id += 1
        
        # Mock order submission
        await asyncio.sleep(0.1)
        
        # Simulate order acceptance
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"IB_BROKER_{self.next_order_id}",
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            side=order_request.side,
            order_type=order_request.order_type,
            status=OrderStatus.SUBMITTED,
            commission=1.0  # IB commission
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel IB order"""
        if not self.connected:
            return False
        
        await asyncio.sleep(0.05)
        return True
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get IB order status"""
        # Mock order status
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"IB_BROKER_{order_id.split('_')[-1]}",
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            symbol="AAPL",
            quantity=100,
            side="buy",
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_fill_price=150.25,
            commission=1.0
        )
    
    async def get_positions(self) -> List[Position]:
        """Get IB positions"""
        # Mock positions
        return [
            Position(
                symbol="AAPL",
                quantity=100,
                avg_cost=148.50,
                market_value=15025.0,
                unrealized_pnl=175.0,
                broker_type=BrokerType.INTERACTIVE_BROKERS
            ),
            Position(
                symbol="MSFT",
                quantity=50,
                avg_cost=298.75,
                market_value=15000.0,
                unrealized_pnl=62.5,
                broker_type=BrokerType.INTERACTIVE_BROKERS
            )
        ]
    
    async def get_account_info(self) -> AccountInfo:
        """Get IB account info"""
        return AccountInfo(
            broker_type=BrokerType.INTERACTIVE_BROKERS,
            account_id=self.credentials.account_id or "IB123456",
            buying_power=50000.0,
            total_value=75000.0,
            cash_balance=25000.0,
            day_trading_buying_power=100000.0
        )
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get IB market data"""
        # Mock market data
        return {
            "symbol": symbol,
            "bid": 149.95,
            "ask": 150.05,
            "last": 150.00,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }

class AlpacaConnector(BrokerInterface):
    """Alpaca Markets connector"""
    
    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.api = None
    
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info("Connected to Alpaca")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Alpaca connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.connected:
            self.connected = False
            logger.info("Disconnected from Alpaca")
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit order to Alpaca"""
        if not self.connected:
            raise ConnectionError("Not connected to Alpaca")
        
        order_id = f"ALPACA_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        await asyncio.sleep(0.1)
        
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"ALP_BROKER_{order_id}",
            broker_type=BrokerType.ALPACA,
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            side=order_request.side,
            order_type=order_request.order_type,
            status=OrderStatus.SUBMITTED,
            commission=0.0  # Alpaca commission-free
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Alpaca order"""
        if not self.connected:
            return False
        
        await asyncio.sleep(0.05)
        return True
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get Alpaca order status"""
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"ALP_BROKER_{order_id}",
            broker_type=BrokerType.ALPACA,
            symbol="AAPL",
            quantity=100,
            side="buy",
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_fill_price=150.15,
            commission=0.0
        )
    
    async def get_positions(self) -> List[Position]:
        """Get Alpaca positions"""
        return [
            Position(
                symbol="TSLA",
                quantity=25,
                avg_cost=198.50,
                market_value=5000.0,
                unrealized_pnl=37.5,
                broker_type=BrokerType.ALPACA
            )
        ]
    
    async def get_account_info(self) -> AccountInfo:
        """Get Alpaca account info"""
        return AccountInfo(
            broker_type=BrokerType.ALPACA,
            account_id=self.credentials.account_id or "ALP123456",
            buying_power=25000.0,
            total_value=30000.0,
            cash_balance=20000.0,
            day_trading_buying_power=50000.0
        )
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get Alpaca market data"""
        return {
            "symbol": symbol,
            "bid": 149.98,
            "ask": 150.02,
            "last": 150.00,
            "volume": 800000,
            "timestamp": datetime.utcnow().isoformat()
        }

class TDAConnector(BrokerInterface):
    """TD Ameritrade connector"""
    
    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.client = None
    
    async def connect(self) -> bool:
        """Connect to TD Ameritrade"""
        try:
            await asyncio.sleep(0.1)
            self.connected = True
            logger.info("Connected to TD Ameritrade")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"TDA connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from TDA"""
        if self.connected:
            self.connected = False
            logger.info("Disconnected from TD Ameritrade")
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit order to TDA"""
        if not self.connected:
            raise ConnectionError("Not connected to TD Ameritrade")
        
        order_id = f"TDA_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        await asyncio.sleep(0.1)
        
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"TDA_BROKER_{order_id}",
            broker_type=BrokerType.TD_AMERITRADE,
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            side=order_request.side,
            order_type=order_request.order_type,
            status=OrderStatus.SUBMITTED,
            commission=0.65  # TDA commission
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel TDA order"""
        if not self.connected:
            return False
        
        await asyncio.sleep(0.05)
        return True
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get TDA order status"""
        return OrderResponse(
            order_id=order_id,
            broker_order_id=f"TDA_BROKER_{order_id}",
            broker_type=BrokerType.TD_AMERITRADE,
            symbol="AAPL",
            quantity=100,
            side="buy",
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_fill_price=150.20,
            commission=0.65
        )
    
    async def get_positions(self) -> List[Position]:
        """Get TDA positions"""
        return [
            Position(
                symbol="SPY",
                quantity=50,
                avg_cost=398.75,
                market_value=20000.0,
                unrealized_pnl=62.5,
                broker_type=BrokerType.TD_AMERITRADE
            )
        ]
    
    async def get_account_info(self) -> AccountInfo:
        """Get TDA account info"""
        return AccountInfo(
            broker_type=BrokerType.TD_AMERITRADE,
            account_id=self.credentials.account_id or "TDA123456",
            buying_power=40000.0,
            total_value=45000.0,
            cash_balance=15000.0,
            day_trading_buying_power=80000.0
        )
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get TDA market data"""
        return {
            "symbol": symbol,
            "bid": 149.97,
            "ask": 150.03,
            "last": 150.00,
            "volume": 1200000,
            "timestamp": datetime.utcnow().isoformat()
        }

class OrderRouter:
    """Intelligent order routing across brokers"""
    
    def __init__(self):
        self.broker_scores: Dict[BrokerType, float] = {}
        self.execution_history: List[OrderResponse] = []
    
    def calculate_broker_score(
        self,
        broker_type: BrokerType,
        order_request: OrderRequest,
        account_info: AccountInfo,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate broker execution score"""
        
        score = 0.0
        
        # Commission cost (lower is better)
        commission_scores = {
            BrokerType.ALPACA: 1.0,  # Commission-free
            BrokerType.INTERACTIVE_BROKERS: 0.8,  # Low commission
            BrokerType.ROBINHOOD: 1.0,  # Commission-free
            BrokerType.TD_AMERITRADE: 0.6,  # Higher commission
            BrokerType.CHARLES_SCHWAB: 0.7,
            BrokerType.WEBULL: 0.9
        }
        score += commission_scores.get(broker_type, 0.5) * 0.3
        
        # Liquidity and execution quality
        liquidity_scores = {
            BrokerType.INTERACTIVE_BROKERS: 1.0,  # Best execution
            BrokerType.TD_AMERITRADE: 0.9,
            BrokerType.CHARLES_SCHWAB: 0.85,
            BrokerType.ALPACA: 0.7,
            BrokerType.ROBINHOOD: 0.6,
            BrokerType.WEBULL: 0.65
        }
        score += liquidity_scores.get(broker_type, 0.5) * 0.4
        
        # Available buying power
        if account_info.buying_power > order_request.quantity * market_data.get("last", 100):
            score += 0.2
        
        # Asset type support
        asset_support = {
            BrokerType.INTERACTIVE_BROKERS: {AssetType.STOCK: 1.0, AssetType.OPTION: 1.0, AssetType.FUTURE: 1.0, AssetType.FOREX: 1.0},
            BrokerType.TD_AMERITRADE: {AssetType.STOCK: 1.0, AssetType.OPTION: 1.0, AssetType.FUTURE: 0.8, AssetType.FOREX: 0.7},
            BrokerType.ALPACA: {AssetType.STOCK: 1.0, AssetType.OPTION: 0.0, AssetType.FUTURE: 0.0, AssetType.FOREX: 0.0},
            BrokerType.CHARLES_SCHWAB: {AssetType.STOCK: 1.0, AssetType.OPTION: 1.0, AssetType.FUTURE: 0.8, AssetType.FOREX: 0.6},
            BrokerType.ROBINHOOD: {AssetType.STOCK: 1.0, AssetType.OPTION: 0.8, AssetType.FUTURE: 0.0, AssetType.FOREX: 0.0},
            BrokerType.WEBULL: {AssetType.STOCK: 1.0, AssetType.OPTION: 0.7, AssetType.FUTURE: 0.0, AssetType.FOREX: 0.0}
        }
        asset_score = asset_support.get(broker_type, {}).get(order_request.asset_type, 0.0)
        score += asset_score * 0.1
        
        return score
    
    async def route_order(
        self,
        order_request: OrderRequest,
        available_brokers: Dict[BrokerType, BrokerInterface],
        account_infos: Dict[BrokerType, AccountInfo]
    ) -> BrokerType:
        """Route order to optimal broker"""
        
        # If preferred broker specified and available
        if (order_request.preferred_broker and 
            order_request.preferred_broker in available_brokers):
            return order_request.preferred_broker
        
        best_broker = None
        best_score = -1.0
        
        for broker_type, broker in available_brokers.items():
            if not broker.connected:
                continue
            
            account_info = account_infos.get(broker_type)
            if not account_info:
                continue
            
            # Get market data for scoring
            try:
                market_data = await broker.get_market_data(order_request.symbol)
            except Exception:
                market_data = {"last": 100.0}  # Fallback
            
            score = self.calculate_broker_score(
                broker_type, order_request, account_info, market_data
            )
            
            if score > best_score:
                best_score = score
                best_broker = broker_type
        
        if best_broker is None:
            raise RuntimeError("No suitable broker found for order routing")
        
        return best_broker

class MultiBrokerConnector:
    """Main multi-broker connector class"""
    
    def __init__(self):
        self.brokers: Dict[BrokerType, BrokerInterface] = {}
        self.order_router = OrderRouter()
        self.active_orders: Dict[str, OrderResponse] = {}
        self.consolidated_positions: Dict[str, List[Position]] = {}
        
    def add_broker(self, credentials: BrokerCredentials):
        """Add broker to connector"""
        
        if credentials.broker_type == BrokerType.INTERACTIVE_BROKERS:
            broker = InteractiveBrokerConnector(credentials)
        elif credentials.broker_type == BrokerType.ALPACA:
            broker = AlpacaConnector(credentials)
        elif credentials.broker_type == BrokerType.TD_AMERITRADE:
            broker = TDAConnector(credentials)
        else:
            raise ValueError(f"Unsupported broker type: {credentials.broker_type}")
        
        self.brokers[credentials.broker_type] = broker
        logger.info(f"Added {credentials.broker_type.value} broker")
    
    async def connect_all_brokers(self) -> Dict[BrokerType, bool]:
        """Connect to all configured brokers"""
        
        connection_results = {}
        
        for broker_type, broker in self.brokers.items():
            try:
                success = await broker.connect()
                connection_results[broker_type] = success
                if success:
                    logger.info(f"Successfully connected to {broker_type.value}")
                else:
                    logger.warning(f"Failed to connect to {broker_type.value}")
            except Exception as e:
                connection_results[broker_type] = False
                logger.error(f"Connection error for {broker_type.value}: {e}")
        
        return connection_results
    
    async def disconnect_all_brokers(self):
        """Disconnect from all brokers"""
        
        for broker_type, broker in self.brokers.items():
            try:
                await broker.disconnect()
                logger.info(f"Disconnected from {broker_type.value}")
            except Exception as e:
                logger.error(f"Disconnect error for {broker_type.value}: {e}")
    
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit order with intelligent routing"""
        
        # Get account info for routing decision
        account_infos = {}
        for broker_type, broker in self.brokers.items():
            if broker.connected:
                try:
                    account_info = await broker.get_account_info()
                    account_infos[broker_type] = account_info
                except Exception as e:
                    logger.warning(f"Could not get account info for {broker_type.value}: {e}")
        
        # Route order to best broker
        connected_brokers = {
            bt: broker for bt, broker in self.brokers.items() 
            if broker.connected
        }
        
        if not connected_brokers:
            raise RuntimeError("No connected brokers available")
        
        selected_broker_type = await self.order_router.route_order(
            order_request, connected_brokers, account_infos
        )
        
        selected_broker = self.brokers[selected_broker_type]
        
        # Submit order
        order_response = await selected_broker.submit_order(order_request)
        
        # Track order
        self.active_orders[order_response.order_id] = order_response
        
        logger.info(f"Order {order_response.order_id} submitted to {selected_broker_type.value}")
        
        return order_response
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        broker = self.brokers[order.broker_type]
        
        if broker.connected:
            success = await broker.cancel_order(order.broker_order_id)
            if success:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Cancelled order {order_id}")
            return success
        
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Get order status"""
        
        if order_id not in self.active_orders:
            return None
        
        order = self.active_orders[order_id]
        broker = self.brokers[order.broker_type]
        
        if broker.connected:
            try:
                updated_order = await broker.get_order_status(order.broker_order_id)
                self.active_orders[order_id] = updated_order
                return updated_order
            except Exception as e:
                logger.error(f"Error getting order status: {e}")
        
        return order
    
    async def get_consolidated_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get consolidated positions across all brokers"""
        
        consolidated = {}
        
        for broker_type, broker in self.brokers.items():
            if not broker.connected:
                continue
            
            try:
                positions = await broker.get_positions()
                
                for position in positions:
                    symbol = position.symbol
                    
                    if symbol not in consolidated:
                        consolidated[symbol] = {
                            "total_quantity": 0.0,
                            "total_market_value": 0.0,
                            "total_unrealized_pnl": 0.0,
                            "weighted_avg_cost": 0.0,
                            "brokers": []
                        }
                    
                    consolidated[symbol]["total_quantity"] += position.quantity
                    consolidated[symbol]["total_market_value"] += position.market_value
                    consolidated[symbol]["total_unrealized_pnl"] += position.unrealized_pnl
                    
                    # Weighted average cost calculation
                    total_cost = consolidated[symbol]["weighted_avg_cost"] * (consolidated[symbol]["total_quantity"] - position.quantity)
                    total_cost += position.avg_cost * position.quantity
                    consolidated[symbol]["weighted_avg_cost"] = total_cost / consolidated[symbol]["total_quantity"]
                    
                    consolidated[symbol]["brokers"].append({
                        "broker": broker_type.value,
                        "quantity": position.quantity,
                        "avg_cost": position.avg_cost,
                        "market_value": position.market_value,
                        "unrealized_pnl": position.unrealized_pnl
                    })
                    
            except Exception as e:
                logger.error(f"Error getting positions from {broker_type.value}: {e}")
        
        return consolidated
    
    async def get_consolidated_account_info(self) -> Dict[str, Any]:
        """Get consolidated account information"""
        
        total_buying_power = 0.0
        total_value = 0.0
        total_cash = 0.0
        broker_accounts = []
        
        for broker_type, broker in self.brokers.items():
            if not broker.connected:
                continue
            
            try:
                account_info = await broker.get_account_info()
                
                total_buying_power += account_info.buying_power
                total_value += account_info.total_value
                total_cash += account_info.cash_balance
                
                broker_accounts.append({
                    "broker": broker_type.value,
                    "account_id": account_info.account_id,
                    "buying_power": account_info.buying_power,
                    "total_value": account_info.total_value,
                    "cash_balance": account_info.cash_balance,
                    "day_trading_buying_power": account_info.day_trading_buying_power,
                    "pattern_day_trader": account_info.pattern_day_trader
                })
                
            except Exception as e:
                logger.error(f"Error getting account info from {broker_type.value}: {e}")
        
        return {
            "total_buying_power": total_buying_power,
            "total_value": total_value,
            "total_cash": total_cash,
            "broker_accounts": broker_accounts,
            "connected_brokers": len([b for b in self.brokers.values() if b.connected])
        }
    
    async def get_best_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get best market data across brokers"""
        
        market_data_sources = []
        
        for broker_type, broker in self.brokers.items():
            if not broker.connected:
                continue
            
            try:
                data = await broker.get_market_data(symbol)
                data["source"] = broker_type.value
                market_data_sources.append(data)
            except Exception as e:
                logger.warning(f"Could not get market data from {broker_type.value}: {e}")
        
        if not market_data_sources:
            return {"error": "No market data available"}
        
        # Aggregate best bid/ask
        best_bid = max(data.get("bid", 0) for data in market_data_sources)
        best_ask = min(data.get("ask", float('inf')) for data in market_data_sources if data.get("ask", float('inf')) < float('inf'))
        
        # Use most recent timestamp
        latest_data = max(market_data_sources, key=lambda x: x.get("timestamp", ""))
        
        return {
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "last": latest_data.get("last"),
            "volume": max(data.get("volume", 0) for data in market_data_sources),
            "timestamp": latest_data.get("timestamp"),
            "spread": best_ask - best_bid if best_ask < float('inf') else None,
            "sources": [data["source"] for data in market_data_sources]
        }
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get status of all brokers"""
        
        status = {
            "total_brokers": len(self.brokers),
            "connected_brokers": len([b for b in self.brokers.values() if b.connected]),
            "active_orders": len(self.active_orders),
            "brokers": []
        }
        
        for broker_type, broker in self.brokers.items():
            broker_status = {
                "broker": broker_type.value,
                "connected": broker.connected,
                "last_error": broker.last_error,
                "paper_trading": broker.credentials.paper_trading
            }
            status["brokers"].append(broker_status)
        
        return status 