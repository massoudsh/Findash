'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PaperTradingAgentPanel } from '@/components/agents/paper-trading-agent-panel';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert } from '@/components/ui/alert';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  BarChart3,
  Activity,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  Zap,
  RefreshCw,
  Download,
  Upload,
  Settings,
  Eye,
  Filter,
  Search,
  Plus,
  Minus,
  ArrowUpRight,
  ArrowDownRight,
  PieChart,
  Calendar
} from 'lucide-react';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  marketValue: number;
  entryTime: string;
  entryFee: number;
}

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  placedAt: string;
  filledAt?: string;
  filledPrice?: number;
  fee: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  value: number;
  fee: number;
  pnl?: number;
  pnlPercent?: number;
  executedAt: string;
}

interface AccountStats {
  totalValue: number;
  cash: number;
  dayPnL: number;
  dayPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  buyingPower: number;
  marginUsed: number;
  positionsValue: number;
  totalTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  sharpeRatio: number;
}

export default function PaperTradingPage() {
  const [selectedTab, setSelectedTab] = useState('trading');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop' | 'stop_limit'>('market');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState<number>(1);
  const [price, setPrice] = useState<number>(0);
  const [stopPrice, setStopPrice] = useState<number>(0);
  const [timeInForce, setTimeInForce] = useState<'GTC' | 'IOC' | 'FOK' | 'DAY'>('GTC');

  // Sample data - in real app this would come from APIs
  const [accountStats, setAccountStats] = useState<AccountStats>({
    totalValue: 100000,
    cash: 45230.50,
    dayPnL: 1250.75,
    dayPnLPercent: 1.26,
    totalPnL: 8450.30,
    totalPnLPercent: 8.45,
    buyingPower: 90000,
    marginUsed: 0,
    positionsValue: 54769.50,
    totalTrades: 47,
    winRate: 68.1,
    avgWin: 420.50,
    avgLoss: -185.30,
    maxDrawdown: -2150.80,
    sharpeRatio: 1.85
  });

  const [positions, setPositions] = useState<Position[]>([
    {
      id: 'pos1',
      symbol: 'AAPL',
      side: 'long',
      quantity: 100,
      entryPrice: 175.23,
      currentPrice: 178.45,
      unrealizedPnL: 322.00,
      unrealizedPnLPercent: 1.84,
      marketValue: 17845.00,
      entryTime: '2024-01-20T09:30:00Z',
      entryFee: 1.00
    },
    {
      id: 'pos2',
      symbol: 'NVDA',
      side: 'long',
      quantity: 50,
      entryPrice: 485.67,
      currentPrice: 492.15,
      unrealizedPnL: 324.00,
      unrealizedPnLPercent: 1.33,
      marketValue: 24607.50,
      entryTime: '2024-01-19T14:15:00Z',
      entryFee: 1.00
    },
    {
      id: 'pos3',
      symbol: 'BTC-USD',
      side: 'long',
      quantity: 0.25,
      entryPrice: 42850.25,
      currentPrice: 43200.00,
      unrealizedPnL: 87.44,
      unrealizedPnLPercent: 0.82,
      marketValue: 10800.00,
      entryTime: '2024-01-18T16:45:00Z',
      entryFee: 2.50
    }
  ]);

  const [orders, setOrders] = useState<Order[]>([
    {
      id: 'ord1',
      symbol: 'TSLA',
      side: 'buy',
      type: 'limit',
      quantity: 20,
      price: 245.00,
      status: 'pending',
      timeInForce: 'GTC',
      placedAt: '2024-01-20T10:15:00Z',
      fee: 1.00
    },
    {
      id: 'ord2',
      symbol: 'MSFT',
      side: 'sell',
      type: 'stop',
      quantity: 30,
      stopPrice: 315.00,
      status: 'pending',
      timeInForce: 'GTC',
      placedAt: '2024-01-20T09:45:00Z',
      fee: 1.00
    }
  ]);

  const [trades, setTrades] = useState<Trade[]>([
    {
      id: 'trd1',
      symbol: 'AAPL',
      side: 'buy',
      quantity: 100,
      price: 175.23,
      value: 17523.00,
      fee: 1.00,
      executedAt: '2024-01-20T09:30:15Z'
    },
    {
      id: 'trd2',
      symbol: 'NVDA',
      side: 'buy',
      quantity: 50,
      price: 485.67,
      value: 24283.50,
      fee: 1.00,
      executedAt: '2024-01-19T14:15:32Z'
    },
    {
      id: 'trd3',
      symbol: 'GOOGL',
      side: 'sell',
      quantity: 40,
      price: 128.90,
      value: 5156.00,
      fee: 1.00,
      pnl: 450.75,
      pnlPercent: 9.6,
      executedAt: '2024-01-18T11:22:45Z'
    }
  ]);

  const watchlist = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BTC-USD', 'ETH-USD', 'GLD', 'SPY'];

  const [marketData, setMarketData] = useState<{[key: string]: {price: number, change: number, changePercent: number}}>({
    'AAPL': { price: 178.45, change: 3.22, changePercent: 1.84 },
    'TSLA': { price: 248.50, change: -3.21, changePercent: -1.27 },
    'MSFT': { price: 320.45, change: -1.23, changePercent: -0.38 },
    'GOOGL': { price: 125.67, change: 0.89, changePercent: 0.71 },
    'AMZN': { price: 142.18, change: 1.87, changePercent: 1.33 },
    'NVDA': { price: 492.15, change: 6.48, changePercent: 1.33 },
    'BTC-USD': { price: 43200.00, change: 349.75, changePercent: 0.82 },
    'ETH-USD': { price: 2485.60, change: -85.40, changePercent: -3.32 },
    'GLD': { price: 185.45, change: 2.15, changePercent: 1.17 },
    'SPY': { price: 445.78, change: 3.21, changePercent: 0.73 }
  });

  const handlePlaceOrder = () => {
    const newOrder: Order = {
      id: `ord${Date.now()}`,
      symbol: selectedSymbol,
      side: orderSide,
      type: orderType,
      quantity: quantity,
      price: orderType === 'market' ? undefined : price,
      stopPrice: (orderType === 'stop' || orderType === 'stop_limit') ? stopPrice : undefined,
      status: 'pending',
      timeInForce: timeInForce,
      placedAt: new Date().toISOString(),
      fee: 1.00
    };

    setOrders(prev => [newOrder, ...prev]);
    
    // Reset form
    setQuantity(1);
    setPrice(0);
    setStopPrice(0);
  };

  const handleCancelOrder = (orderId: string) => {
    setOrders(prev => prev.map(order => 
      order.id === orderId ? { ...order, status: 'cancelled' as const } : order
    ));
  };

  const handleClosePosition = (positionId: string) => {
    const position = positions.find(p => p.id === positionId);
    if (position) {
      // Create closing trade
      const closingTrade: Trade = {
        id: `trd${Date.now()}`,
        symbol: position.symbol,
        side: position.side === 'long' ? 'sell' : 'buy',
        quantity: position.quantity,
        price: position.currentPrice,
        value: position.marketValue,
        fee: 1.00,
        pnl: position.unrealizedPnL,
        pnlPercent: position.unrealizedPnLPercent,
        executedAt: new Date().toISOString()
      };

      setTrades(prev => [closingTrade, ...prev]);
      setPositions(prev => prev.filter(p => p.id !== positionId));
      
      // Update account stats
      setAccountStats(prev => ({
        ...prev,
        cash: prev.cash + position.marketValue,
        positionsValue: prev.positionsValue - position.marketValue,
        totalPnL: prev.totalPnL + position.unrealizedPnL,
        totalTrades: prev.totalTrades + 1
      }));
    }
  };

  const getOrderStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'cancelled': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'rejected': return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
      <div className="min-w-0 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Paper Trading</h1>
        <p className="text-muted-foreground">
          Practice trading with real market data in a risk-free environment
        </p>
      </div>

      {/* Account Summary */}
      <div className="grid gap-4 md:grid-cols-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Value</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(accountStats.totalValue)}</div>
            <p className="text-xs text-muted-foreground">
              Portfolio value
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Day P&L</CardTitle>
            {accountStats.dayPnL >= 0 ? <TrendingUp className="h-4 w-4 text-green-500" /> : <TrendingDown className="h-4 w-4 text-red-500" />}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${accountStats.dayPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatCurrency(accountStats.dayPnL)}
            </div>
            <p className={`text-xs ${accountStats.dayPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercent(accountStats.dayPnLPercent)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${accountStats.totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatCurrency(accountStats.totalPnL)}
            </div>
            <p className={`text-xs ${accountStats.totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercent(accountStats.totalPnLPercent)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cash</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(accountStats.cash)}</div>
            <p className="text-xs text-muted-foreground">
              Available cash
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{accountStats.winRate}%</div>
            <p className="text-xs text-muted-foreground">
              {accountStats.totalTrades} total trades
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{accountStats.sharpeRatio}</div>
            <p className="text-xs text-muted-foreground">
              Risk-adjusted return
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="positions">Positions</TabsTrigger>
          <TabsTrigger value="orders">Orders</TabsTrigger>
          <TabsTrigger value="trades">Trade History</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="trading" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Order Entry */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Place Order
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="symbol">Symbol</Label>
                  <select 
                    value={selectedSymbol} 
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    className="w-full p-2 border rounded-md"
                  >
                    {watchlist.map(symbol => (
                      <option key={symbol} value={symbol}>{symbol}</option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <Button 
                    variant={orderSide === 'buy' ? 'default' : 'outline'} 
                    onClick={() => setOrderSide('buy')}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    BUY
                  </Button>
                  <Button 
                    variant={orderSide === 'sell' ? 'default' : 'outline'} 
                    onClick={() => setOrderSide('sell')}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    SELL
                  </Button>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="orderType">Order Type</Label>
                  <select 
                    value={orderType} 
                    onChange={(e) => setOrderType(e.target.value as any)}
                    className="w-full p-2 border rounded-md"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                    <option value="stop_limit">Stop Limit</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="quantity">Quantity</Label>
                  <Input
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(Number(e.target.value))}
                    min="1"
                  />
                </div>

                {(orderType === 'limit' || orderType === 'stop_limit') && (
                  <div className="space-y-2">
                    <Label htmlFor="price">Limit Price</Label>
                    <Input
                      type="number"
                      value={price}
                      onChange={(e) => setPrice(Number(e.target.value))}
                      step="0.01"
                    />
                  </div>
                )}

                {(orderType === 'stop' || orderType === 'stop_limit') && (
                  <div className="space-y-2">
                    <Label htmlFor="stopPrice">Stop Price</Label>
                    <Input
                      type="number"
                      value={stopPrice}
                      onChange={(e) => setStopPrice(Number(e.target.value))}
                      step="0.01"
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="timeInForce">Time in Force</Label>
                  <select 
                    value={timeInForce} 
                    onChange={(e) => setTimeInForce(e.target.value as any)}
                    className="w-full p-2 border rounded-md"
                  >
                    <option value="GTC">Good Till Cancelled</option>
                    <option value="DAY">Day Only</option>
                    <option value="IOC">Immediate or Cancel</option>
                    <option value="FOK">Fill or Kill</option>
                  </select>
                </div>

                {selectedSymbol && marketData[selectedSymbol] && (
                  <div className="p-3 bg-gray-50 rounded-md">
                    <div className="flex justify-between">
                      <span className="font-medium">{selectedSymbol}</span>
                      <span className="font-bold">{formatCurrency(marketData[selectedSymbol].price)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className={marketData[selectedSymbol].change >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {formatCurrency(marketData[selectedSymbol].change)}
                      </span>
                      <span className={marketData[selectedSymbol].change >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {formatPercent(marketData[selectedSymbol].changePercent)}
                      </span>
                    </div>
                  </div>
                )}

                <Button onClick={handlePlaceOrder} className="w-full" size="lg">
                  Place {orderSide.toUpperCase()} Order
                </Button>
              </CardContent>
            </Card>

            {/* Market Watch */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Market Watch
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {watchlist.map(symbol => {
                    const data = marketData[symbol];
                    return (
                      <div 
                        key={symbol} 
                        className="flex justify-between items-center p-2 hover:bg-gray-50 rounded cursor-pointer"
                        onClick={() => setSelectedSymbol(symbol)}
                      >
                        <div className="font-medium">{symbol}</div>
                        <div className="text-right">
                          <div className="font-bold">{formatCurrency(data.price)}</div>
                          <div className={`text-sm ${data.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatPercent(data.changePercent)}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="positions" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="w-5 h-5" />
                Open Positions
              </CardTitle>
            </CardHeader>
            <CardContent>
              {positions.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No open positions
                </div>
              ) : (
                <div className="space-y-4">
                  {positions.map(position => (
                    <div key={position.id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="font-bold text-lg">{position.symbol}</h3>
                          <Badge variant={position.side === 'long' ? 'default' : 'destructive'}>
                            {position.side.toUpperCase()} {position.quantity}
                          </Badge>
                        </div>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleClosePosition(position.id)}
                        >
                          Close Position
                        </Button>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-gray-500">Entry Price</div>
                          <div className="font-medium">{formatCurrency(position.entryPrice)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Current Price</div>
                          <div className="font-medium">{formatCurrency(position.currentPrice)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Market Value</div>
                          <div className="font-medium">{formatCurrency(position.marketValue)}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Unrealized P&L</div>
                          <div className={`font-medium ${position.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatCurrency(position.unrealizedPnL)} ({formatPercent(position.unrealizedPnLPercent)})
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="orders" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Order Management
              </CardTitle>
            </CardHeader>
            <CardContent>
              {orders.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No active orders
                </div>
              ) : (
                <div className="space-y-4">
                  {orders.map(order => (
                    <div key={order.id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-3">
                          {getOrderStatusIcon(order.status)}
                          <div>
                            <h3 className="font-bold">{order.symbol}</h3>
                            <div className="text-sm text-gray-500">
                              {order.side.toUpperCase()} {order.quantity} @ {order.type.toUpperCase()}
                            </div>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Badge variant={order.status === 'pending' ? 'secondary' : order.status === 'filled' ? 'default' : 'destructive'}>
                            {order.status.toUpperCase()}
                          </Badge>
                          {order.status === 'pending' && (
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => handleCancelOrder(order.id)}
                            >
                              Cancel
                            </Button>
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-gray-500">Type</div>
                          <div className="font-medium">{order.type.toUpperCase()}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Price</div>
                          <div className="font-medium">
                            {order.price ? formatCurrency(order.price) : 'Market'}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500">Time in Force</div>
                          <div className="font-medium">{order.timeInForce}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Placed At</div>
                          <div className="font-medium">{new Date(order.placedAt).toLocaleTimeString()}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trades" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Trade History
              </CardTitle>
            </CardHeader>
            <CardContent>
              {trades.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No trades executed
                </div>
              ) : (
                <div className="space-y-4">
                  {trades.map(trade => (
                    <div key={trade.id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-3">
                        <div>
                          <h3 className="font-bold">{trade.symbol}</h3>
                          <div className="flex items-center gap-2">
                            {trade.side === 'buy' ? 
                              <ArrowUpRight className="w-4 h-4 text-green-500" /> : 
                              <ArrowDownRight className="w-4 h-4 text-red-500" />
                            }
                            <span className="text-sm">
                              {trade.side.toUpperCase()} {trade.quantity} @ {formatCurrency(trade.price)}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">{formatCurrency(trade.value)}</div>
                          {trade.pnl && (
                            <div className={`text-sm ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {formatCurrency(trade.pnl)} ({formatPercent(trade.pnlPercent || 0)})
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex justify-between text-sm text-gray-500">
                        <span>Fee: {formatCurrency(trade.fee)}</span>
                        <span>{new Date(trade.executedAt).toLocaleString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-500">Total Trades</div>
                    <div className="text-2xl font-bold">{accountStats.totalTrades}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Win Rate</div>
                    <div className="text-2xl font-bold text-green-600">{accountStats.winRate}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Avg Win</div>
                    <div className="text-lg font-bold text-green-600">{formatCurrency(accountStats.avgWin)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Avg Loss</div>
                    <div className="text-lg font-bold text-red-600">{formatCurrency(accountStats.avgLoss)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Max Drawdown</div>
                    <div className="text-lg font-bold text-red-600">{formatCurrency(accountStats.maxDrawdown)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Sharpe Ratio</div>
                    <div className="text-lg font-bold">{accountStats.sharpeRatio}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Account Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button className="w-full" variant="outline">
                  <Download className="w-4 h-4 mr-2" />
                  Export Trade History
                </Button>
                <Button className="w-full" variant="outline">
                  <Upload className="w-4 h-4 mr-2" />
                  Import Portfolio
                </Button>
                <Button className="w-full" variant="outline">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Reset Account
                </Button>
                <Button className="w-full" variant="outline">
                  <Settings className="w-4 h-4 mr-2" />
                  Trading Settings
                </Button>
              </CardContent>
            </Card>
          </div>

          <Alert>
            <AlertCircle className="h-4 w-4" />
            <div>
              <h4 className="font-semibold">Paper Trading Disclaimer</h4>
              <p className="text-sm">
                This is a simulated trading environment. All trades are virtual and no real money is involved. 
                Market data may be delayed. Past performance does not guarantee future results.
              </p>
            </div>
          </Alert>
        </TabsContent>
      </Tabs>
      </div>
      <aside className="hidden xl:block min-h-[360px]">
        <PaperTradingAgentPanel />
      </aside>
    </div>
  );
} 