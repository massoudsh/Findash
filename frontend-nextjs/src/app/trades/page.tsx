'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
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
  Shield,
  Zap,
  RefreshCw,
  Download,
  Settings,
  Eye,
  Lock,
  Unlock,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  PieChart,
  Wifi,
  WifiOff,
  Bell,
  Target,
  Power,
  Pause,
  Play,
  StopCircle,
  Plus,
  Minus,
  MoreVertical,
  History,
  LineChart,
  Gauge
} from 'lucide-react';

// Interfaces
interface LivePosition {
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
  stopLoss?: number;
  takeProfit?: number;
  dayPnL: number;
  fees: number;
}

interface LiveOrder {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected' | 'partially_filled';
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  placedAt: string;
  filledQuantity: number;
  remainingQuantity: number;
  avgFillPrice?: number;
  fees: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  value: number;
  fees: number;
  pnl?: number;
  pnlPercent?: number;
  executed_at: string;
  status: string;
  portfolio_id?: number;
}

interface AccountInfo {
  accountId: string;
  totalValue: number;
  cash: number;
  buyingPower: number;
  dayPnL: number;
  dayPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  marginUsed: number;
  marginAvailable: number;
  positionsValue: number;
  lastUpdated: string;
}

interface ConnectionStatus {
  isConnected: boolean;
  brokerConnection: 'connected' | 'disconnected' | 'connecting';
  dataFeed: 'live' | 'delayed' | 'offline';
  lastHeartbeat: string;
  latency: number;
}

export default function UnifiedTradingPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop' | 'stop_limit'>('limit');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [quantity, setQuantity] = useState<number>(1);
  const [price, setPrice] = useState<number>(0);
  const [stopPrice, setStopPrice] = useState<number>(0);
  const [stopLoss, setStopLoss] = useState<number>(0);
  const [takeProfit, setTakeProfit] = useState<number>(0);
  const [timeInForce, setTimeInForce] = useState<'GTC' | 'IOC' | 'FOK' | 'DAY'>('GTC');
  const [tradingEnabled, setTradingEnabled] = useState(false);
  const [confirmationRequired, setConfirmationRequired] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  // Mock data
  const [connectionStatus] = useState<ConnectionStatus>({
    isConnected: true,
    brokerConnection: 'connected',
    dataFeed: 'live',
    lastHeartbeat: new Date().toISOString(),
    latency: 12
  });

  const [accountInfo] = useState<AccountInfo>({
    accountId: 'ACC-12345',
    totalValue: 50000,
    cash: 15230.50,
    buyingPower: 45000,
    dayPnL: -125.75,
    dayPnLPercent: -0.25,
    totalPnL: 2450.30,
    totalPnLPercent: 5.14,
    marginUsed: 5000,
    marginAvailable: 40000,
    positionsValue: 34769.50,
    lastUpdated: new Date().toISOString()
  });

  const [positions] = useState<LivePosition[]>([
    {
      id: 'pos1',
      symbol: 'AAPL',
      side: 'long',
      quantity: 50,
      entryPrice: 178.23,
      currentPrice: 177.45,
      unrealizedPnL: -39.00,
      unrealizedPnLPercent: -0.44,
      marketValue: 8872.50,
      entryTime: '2024-01-20T09:30:00Z',
      stopLoss: 170.00,
      takeProfit: 185.00,
      dayPnL: -39.00,
      fees: 1.00
    },
    {
      id: 'pos2',
      symbol: 'TSLA',
      side: 'long',
      quantity: 25,
      entryPrice: 248.67,
      currentPrice: 251.15,
      unrealizedPnL: 62.00,
      unrealizedPnLPercent: 1.00,
      marketValue: 6278.75,
      entryTime: '2024-01-20T10:15:00Z',
      stopLoss: 240.00,
      takeProfit: 260.00,
      dayPnL: 62.00,
      fees: 1.25
    }
  ]);

  const [liveOrders] = useState<LiveOrder[]>([
    {
      id: 'ord1',
      symbol: 'NVDA',
      side: 'buy',
      type: 'limit',
      quantity: 10,
      price: 720.00,
      status: 'pending',
      timeInForce: 'GTC',
      placedAt: '2024-01-20T11:30:00Z',
      filledQuantity: 0,
      remainingQuantity: 10,
      fees: 0
    }
  ]);

  const [tradeHistory] = useState<Trade[]>([
    {
      id: 'trade1',
      symbol: 'AAPL',
      side: 'buy',
      quantity: 50,
      price: 178.23,
      value: 8911.50,
      fees: 1.00,
      executed_at: '2024-01-20T09:30:00Z',
      status: 'executed'
    },
    {
      id: 'trade2',
      symbol: 'TSLA',
      side: 'buy',
      quantity: 25,
      price: 248.67,
      value: 6216.75,
      fees: 1.25,
      executed_at: '2024-01-20T10:15:00Z',
      status: 'executed'
    }
  ]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('fa-IR');
  };

  const getConnectionStatusIcon = () => {
    return connectionStatus.isConnected ? (
      <Wifi className="h-4 w-4 text-green-500" />
    ) : (
      <WifiOff className="h-4 w-4 text-red-500" />
    );
  };

  const handlePlaceOrder = async () => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      console.log('Order placed:', {
        symbol: selectedSymbol,
        side: orderSide,
        type: orderType,
        quantity,
        price: orderType !== 'market' ? price : undefined,
        stopPrice: orderType.includes('stop') ? stopPrice : undefined,
        timeInForce
      });
    } catch (error) {
      console.error('Failed to place order:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              مرکز فرماندهی
            </h1>
            <p className="text-gray-400 text-lg">معاملات زنده، مدیریت سفارش‌ها و تاریخچه معاملات</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              {getConnectionStatusIcon()}
              <span className="text-sm text-gray-400">
                {connectionStatus.brokerConnection === 'connected' ? 'متصل' : connectionStatus.brokerConnection === 'connecting' ? 'در حال اتصال' : 'قطع'} • {connectionStatus.latency} میلی‌ثانیه
              </span>
            </div>
            <Badge variant={tradingEnabled ? "default" : "secondary"}>
              {tradingEnabled ? "معاملات زنده" : "معاملات آزمایشی"}
            </Badge>
          </div>
        </div>
      </div>

      {/* Account Overview — responsive like dashboard/options */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4 md:gap-5 lg:gap-6 mb-6 sm:mb-8">
        <Card className="glass-card min-w-0">
          <CardContent className="p-4 sm:p-5 md:p-6">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 rounded-lg bg-blue-500/20 shrink-0">
                <DollarSign className="h-5 w-5 sm:h-6 sm:w-6 text-blue-400" />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm text-muted-foreground">ارزش کل</p>
                <p className="text-lg sm:text-xl md:text-2xl font-bold text-blue-400 truncate">{formatCurrency(accountInfo.totalValue)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card min-w-0">
          <CardContent className="p-4 sm:p-5 md:p-6">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 rounded-lg bg-green-500/20 shrink-0">
                <TrendingUp className="h-5 w-5 sm:h-6 sm:w-6 text-green-400" />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm text-muted-foreground">سود/زیان روزانه</p>
                <p className={`text-lg sm:text-xl md:text-2xl font-bold truncate ${accountInfo.dayPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(accountInfo.dayPnL)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card min-w-0">
          <CardContent className="p-4 sm:p-5 md:p-6">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 rounded-lg bg-purple-500/20 shrink-0">
                <BarChart3 className="h-5 w-5 sm:h-6 sm:w-6 text-purple-400" />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm text-muted-foreground">قدرت خرید</p>
                <p className="text-lg sm:text-xl md:text-2xl font-bold text-purple-400 truncate">{formatCurrency(accountInfo.buyingPower)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card min-w-0">
          <CardContent className="p-4 sm:p-5 md:p-6">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 rounded-lg bg-yellow-500/20 shrink-0">
                <PieChart className="h-5 w-5 sm:h-6 sm:w-6 text-yellow-400" />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm text-muted-foreground">پوزیشن‌ها</p>
                <p className="text-lg sm:text-xl md:text-2xl font-bold text-yellow-400">{positions.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card min-w-0">
          <CardContent className="p-4 sm:p-5 md:p-6">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 rounded-lg bg-red-500/20 shrink-0">
                <Activity className="h-5 w-5 sm:h-6 sm:w-6 text-red-400" />
              </div>
              <div className="min-w-0">
                <p className="text-xs sm:text-sm text-muted-foreground">سفارش‌های باز</p>
                <p className="text-lg sm:text-xl md:text-2xl font-bold text-red-400">{liveOrders.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Trading Interface */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4 sm:space-y-6">
        <TabsList className="grid w-full grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1 glass-card p-1 overflow-x-auto">
          <TabsTrigger value="overview" className="text-xs sm:text-sm whitespace-nowrap">نمای کلی</TabsTrigger>
          <TabsTrigger value="order-entry" className="text-xs sm:text-sm whitespace-nowrap">ثبت سفارش</TabsTrigger>
          <TabsTrigger value="positions" className="text-xs sm:text-sm whitespace-nowrap">پوزیشن‌ها</TabsTrigger>
          <TabsTrigger value="orders" className="text-xs sm:text-sm whitespace-nowrap">سفارش‌ها</TabsTrigger>
          <TabsTrigger value="history" className="text-xs sm:text-sm whitespace-nowrap">تاریخچه</TabsTrigger>
          <TabsTrigger value="funding-rates" className="text-xs sm:text-sm whitespace-nowrap">نرخ فاندینگ</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4 sm:space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            {/* Quick Order Entry */}
            <Card className="glass-card min-w-0">
              <CardHeader className="p-4 sm:p-6 pb-2">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <Zap className="h-5 w-5 text-yellow-400 shrink-0" />
                  سفارش سریع
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 p-4 sm:p-6 pt-0">
                <div>
                  <Label>نماد</Label>
                  <Input
                    value={selectedSymbol}
                    onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                    placeholder="مثلاً AAPL"
                    className="mt-1"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label>تعداد</Label>
                    <Input
                      type="number"
                      value={quantity}
                      onChange={(e) => setQuantity(Number(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>قیمت</Label>
                    <Input
                      type="number"
                      value={price}
                      onChange={(e) => setPrice(Number(e.target.value))}
                      placeholder="بازار"
                      className="mt-1"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => {setOrderSide('buy'); handlePlaceOrder();}}
                    disabled={isLoading}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    خرید
                  </Button>
                  <Button
                    onClick={() => {setOrderSide('sell'); handlePlaceOrder();}}
                    disabled={isLoading}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    فروش
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Active Positions Summary */}
            <Card className="glass-card min-w-0">
              <CardHeader className="p-4 sm:p-6 pb-2">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <Target className="h-5 w-5 text-blue-400 shrink-0" />
                  پوزیشن‌های فعال
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 sm:p-6 pt-0">
                <div className="space-y-3">
                  {positions.slice(0, 3).map((position) => (
                    <div key={position.id} className="flex items-center justify-between p-3 border border-gray-700 rounded-lg">
                      <div>
                        <p className="font-semibold">{position.symbol}</p>
                        <p className="text-sm text-gray-400">{position.quantity} سهم</p>
                      </div>
                      <div className="text-right">
                        <p className={`font-bold ${position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatCurrency(position.unrealizedPnL)}
                        </p>
                        <p className="text-sm text-gray-400">{formatPercent(position.unrealizedPnLPercent)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card className="glass-card min-w-0">
              <CardHeader className="p-4 sm:p-6 pb-2">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <History className="h-5 w-5 text-purple-400 shrink-0" />
                  فعالیت‌های اخیر
                </CardTitle>
              </CardHeader>
              <CardContent className="p-4 sm:p-6 pt-0">
                <div className="space-y-3">
                  {tradeHistory.slice(0, 3).map((trade) => (
                    <div key={trade.id} className="flex items-center justify-between p-3 border border-gray-700 rounded-lg">
                      <div>
                        <p className="font-semibold">{trade.symbol}</p>
                        <p className="text-sm text-gray-400">
                          {trade.side === 'buy' ? 'خرید' : 'فروش'} {trade.quantity}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold">{formatCurrency(trade.price)}</p>
                        <p className="text-sm text-gray-400">{formatDate(trade.executed_at)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Order Entry Tab */}
        <TabsContent value="order-entry" className="space-y-6">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-6">
            <Card className="glass-card min-w-0">
              <CardHeader className="p-4 sm:p-6 pb-2">
                <CardTitle className="text-base sm:text-lg">ثبت سفارش پیشرفته</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6 p-4 sm:p-6 pt-0">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <Label>نماد</Label>
                    <Input
                      value={selectedSymbol}
                      onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                      placeholder="مثلاً AAPL"
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>نوع سفارش</Label>
                    <Select value={orderType} onValueChange={(value: any) => setOrderType(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="market">بازار</SelectItem>
                        <SelectItem value="limit">محدود</SelectItem>
                        <SelectItem value="stop">استاپ</SelectItem>
                        <SelectItem value="stop_limit">استاپ محدود</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>جهت</Label>
                    <Select value={orderSide} onValueChange={(value: any) => setOrderSide(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="buy">خرید</SelectItem>
                        <SelectItem value="sell">فروش</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>تعداد</Label>
                    <Input
                      type="number"
                      value={quantity}
                      onChange={(e) => setQuantity(Number(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                </div>

                {orderType !== 'market' && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>قیمت محدود</Label>
                      <Input
                        type="number"
                        value={price}
                        onChange={(e) => setPrice(Number(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    {orderType.includes('stop') && (
                      <div>
                        <Label>قیمت استاپ</Label>
                        <Input
                          type="number"
                          value={stopPrice}
                          onChange={(e) => setStopPrice(Number(e.target.value))}
                          className="mt-1"
                        />
                      </div>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>اعتبار زمانی</Label>
                    <Select value={timeInForce} onValueChange={(value: any) => setTimeInForce(value)}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="GTC">معتبر تا لغو</SelectItem>
                        <SelectItem value="IOC">فوری یا لغو</SelectItem>
                        <SelectItem value="FOK">کامل یا هیچ</SelectItem>
                        <SelectItem value="DAY">سفارش روزانه</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-end">
                    <Button
                      onClick={handlePlaceOrder}
                      disabled={isLoading}
                      className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
                    >
                      {isLoading ? 'در حال ثبت...' : 'ثبت سفارش'}
                    </Button>
                  </div>
                </div>

                <div className="border-t border-gray-700 pt-4 space-y-4">
                  <h4 className="font-semibold">مدیریت ریسک</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>حد ضرر</Label>
                      <Input
                        type="number"
                        value={stopLoss}
                        onChange={(e) => setStopLoss(Number(e.target.value))}
                        placeholder="اختیاری"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>حد سود</Label>
                      <Input
                        type="number"
                        value={takeProfit}
                        onChange={(e) => setTakeProfit(Number(e.target.value))}
                        placeholder="اختیاری"
                        className="mt-1"
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card min-w-0">
              <CardHeader className="p-4 sm:p-6 pb-2">
                <CardTitle className="text-base sm:text-lg">پیش‌نمایش و تنظیمات سفارش</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="p-4 border border-gray-700 rounded-lg space-y-2">
                  <h4 className="font-semibold">خلاصه سفارش</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span>نماد:</span>
                      <span className="font-mono">{selectedSymbol}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>جهت:</span>
                      <span className={orderSide === 'buy' ? 'text-green-400' : 'text-red-400'}>
                        {orderSide === 'buy' ? 'خرید' : 'فروش'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>نوع:</span>
                      <span className="capitalize">{orderType.replace('_', ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>تعداد:</span>
                      <span>{quantity}</span>
                    </div>
                    {orderType !== 'market' && (
                      <div className="flex justify-between">
                        <span>قیمت:</span>
                        <span>{formatCurrency(price)}</span>
                      </div>
                    )}
                    <div className="flex justify-between border-t border-gray-700 pt-2">
                      <span>ارزش تخمینی:</span>
                      <span className="font-bold">
                        {formatCurrency((orderType === 'market' ? 150 : price) * quantity)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold">تنظیمات معاملاتی</h4>
                  <div className="flex items-center justify-between">
                    <Label>معاملات زنده</Label>
                    <Switch
                      checked={tradingEnabled}
                      onCheckedChange={setTradingEnabled}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>نیاز به تأیید</Label>
                    <Switch
                      checked={confirmationRequired}
                      onCheckedChange={setConfirmationRequired}
                    />
                  </div>
                </div>

                {!tradingEnabled && (
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      حالت معاملات آزمایشی فعال است. سفارش‌ها به‌صورت شبیه‌سازی‌شده اجرا می‌شوند.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Positions Tab */}
        <TabsContent value="positions" className="space-y-4 sm:space-y-6">
          <Card className="glass-card min-w-0 overflow-hidden">
            <CardHeader className="p-4 sm:p-6 pb-2">
              <CardTitle className="flex items-center justify-between">
                <span>پوزیشن‌های فعال</span>
                <Button variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  بازخوانی
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3">نماد</th>
                      <th className="text-left py-3">جهت</th>
                      <th className="text-right py-3">تعداد</th>
                      <th className="text-right py-3">قیمت ورود</th>
                      <th className="text-right py-3">قیمت فعلی</th>
                      <th className="text-right py-3">ارزش بازار</th>
                      <th className="text-right py-3">سود/زیان محقق‌نشده</th>
                      <th className="text-right py-3">سود/زیان روزانه</th>
                      <th className="text-center py-3">عملیات</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position) => (
                      <tr key={position.id} className="border-b border-gray-700/50">
                        <td className="py-3 font-semibold">{position.symbol}</td>
                        <td className="py-3">
                          <Badge variant={position.side === 'long' ? 'default' : 'secondary'}>
                            {position.side === 'long' ? 'خرید (Long)' : 'فروش (Short)'}
                          </Badge>
                        </td>
                        <td className="text-right py-3">{position.quantity}</td>
                        <td className="text-right py-3">{formatCurrency(position.entryPrice)}</td>
                        <td className="text-right py-3">{formatCurrency(position.currentPrice)}</td>
                        <td className="text-right py-3">{formatCurrency(position.marketValue)}</td>
                        <td className={`text-right py-3 font-bold ${position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatCurrency(position.unrealizedPnL)}
                          <br />
                          <span className="text-sm">({formatPercent(position.unrealizedPnLPercent)})</span>
                        </td>
                        <td className={`text-right py-3 font-bold ${position.dayPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatCurrency(position.dayPnL)}
                        </td>
                        <td className="text-center py-3">
                          <Button variant="outline" size="sm">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Orders Tab */}
        <TabsContent value="orders" className="space-y-4 sm:space-y-6">
          <Card className="glass-card min-w-0 overflow-hidden">
            <CardHeader className="p-4 sm:p-6 pb-2">
              <CardTitle className="flex flex-wrap items-center justify-between gap-2 text-base sm:text-lg">
                <span>سفارش‌های باز</span>
                <Button variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  بازخوانی
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3">نماد</th>
                      <th className="text-left py-3">جهت</th>
                      <th className="text-left py-3">نوع</th>
                      <th className="text-right py-3">تعداد</th>
                      <th className="text-right py-3">قیمت</th>
                      <th className="text-left py-3">وضعیت</th>
                      <th className="text-left py-3">اعتبار زمانی</th>
                      <th className="text-left py-3">زمان ثبت</th>
                      <th className="text-center py-3">عملیات</th>
                    </tr>
                  </thead>
                  <tbody>
                    {liveOrders.map((order) => (
                      <tr key={order.id} className="border-b border-gray-700/50">
                        <td className="py-3 font-semibold">{order.symbol}</td>
                        <td className="py-3">
                          <Badge variant={order.side === 'buy' ? 'default' : 'secondary'}>
                            {order.side === 'buy' ? 'خرید' : 'فروش'}
                          </Badge>
                        </td>
                        <td className="py-3 capitalize">{order.type.replace('_', ' ')}</td>
                        <td className="text-right py-3">
                          {order.filledQuantity}/{order.quantity}
                        </td>
                        <td className="text-right py-3">
                          {order.price ? formatCurrency(order.price) : 'بازار'}
                        </td>
                        <td className="py-3">
                          <Badge variant="outline" className="capitalize">
                            {order.status.replace('_', ' ')}
                          </Badge>
                        </td>
                        <td className="py-3">{order.timeInForce}</td>
                        <td className="py-3 text-sm">{formatDate(order.placedAt)}</td>
                        <td className="text-center py-3">
                          <Button variant="outline" size="sm" className="text-red-400">
                            لغو
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>تاریخچه معاملات</span>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    خروجی
                  </Button>
                  <Button variant="outline" size="sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    بازخوانی
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3">تاریخ</th>
                      <th className="text-left py-3">نماد</th>
                      <th className="text-left py-3">جهت</th>
                      <th className="text-right py-3">تعداد</th>
                      <th className="text-right py-3">قیمت</th>
                      <th className="text-right py-3">ارزش</th>
                      <th className="text-right py-3">کارمزد</th>
                      <th className="text-left py-3">وضعیت</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tradeHistory.map((trade) => (
                      <tr key={trade.id} className="border-b border-gray-700/50">
                        <td className="py-3 text-sm">{formatDate(trade.executed_at)}</td>
                        <td className="py-3 font-semibold">{trade.symbol}</td>
                        <td className="py-3">
                          <Badge variant={trade.side === 'buy' ? 'default' : 'secondary'}>
                            {trade.side === 'buy' ? 'خرید' : 'فروش'}
                          </Badge>
                        </td>
                        <td className="text-right py-3">{trade.quantity}</td>
                        <td className="text-right py-3">{formatCurrency(trade.price)}</td>
                        <td className="text-right py-3">{formatCurrency(trade.value)}</td>
                        <td className="text-right py-3">{formatCurrency(trade.fees)}</td>
                        <td className="py-3">
                          <Badge variant="outline" className="capitalize">
                            {trade.status === 'executed' ? 'اجراشده' : trade.status}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Funding Rates Tab */}
        <TabsContent value="funding-rates" className="space-y-6">
          <FundingRatesPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Funding Rates Panel Component
function FundingRatesPanel() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [orderType, setOrderType] = useState('market');
  const [tradeDirection, setTradeDirection] = useState<'long' | 'short'>('long');
  const [leverage, setLeverage] = useState('2x');
  const [notionalSize, setNotionalSize] = useState(0);
  const [reduceOnly, setReduceOnly] = useState(false);

  const timeframes = ['5m', '1H', '1D', '1W'];

  // Mock order book data
  const shortRateOrders = [
    { rate: 6.2, size: 31.5673 },
    { rate: 6.1, size: 31.751 },
    { rate: 6.0, size: 758.225 },
    { rate: 5.9, size: 31.8935 },
    { rate: 5.8, size: 32.4144 },
    { rate: 5.7, size: 32.2105 },
    { rate: 5.6, size: 337.52 },
    { rate: 5.5, size: 132.649 },
  ];

  const longRateOrders = [
    { rate: 5.3, size: 12.695 },
    { rate: 5.2, size: 51.6531 },
    { rate: 5.1, size: 33.2776 },
    { rate: 5.0, size: 35.9013 },
    { rate: 4.9, size: 18.0023 },
    { rate: 4.8, size: 17.8371 },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full bg-gradient-to-r from-blue-500 to-purple-500" />
            <div className="w-6 h-6 rounded-full bg-gradient-to-r from-green-400 to-green-600 -ml-2" />
            <span className="text-xl font-bold">ETHUSDC</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold">۲۵ روز</span>
          <span className="text-sm text-muted-foreground">(سررسید ۲۷ فوریه ۲۰۲۶)</span>
        </div>
      </div>

      {/* Market Stats */}
      <div className="flex items-center gap-8 pb-4 border-b">
        <div>
          <div className="text-3xl font-bold text-green-500">۵.۳۴٪</div>
          <div className="text-xs text-muted-foreground">نرخ سود ضمنی</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">نرخ سود مارک</div>
          <div className="text-sm font-semibold">۵.۳۵٪</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">نرخ سود دارایی پایه</div>
          <div className="text-sm font-semibold text-red-500">-۶.۸۲٪</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">حجم قرارداد باز</div>
          <div className="text-sm font-semibold">۵.۱۸۱۵K ETH</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">حجم ۲۴ ساعته</div>
          <div className="text-sm font-semibold">۶.۳۸۹K ETH</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">تسویه بعدی</div>
          <div className="text-sm font-semibold">۰۰:۴۶:۴۲</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Panel - Chart */}
        <div className="lg:col-span-6 min-w-0">
          <Card className="glass-card h-full min-w-0">
            <CardContent className="p-4">
              {/* Chart Tabs */}
              <div className="flex items-center gap-4 mb-4">
                <Button variant="default" size="sm">نمودار نرخ سود</Button>
                <Button variant="ghost" size="sm">سود/زیان من</Button>
              </div>

              {/* Chart Legend */}
              <div className="flex items-center gap-4 mb-2 text-xs">
                <span>نرخ سود ضمنی</span>
                <span className="text-green-500">O۵.۳۴٪</span>
                <span className="text-green-500">H۵.۳۴٪</span>
                <span className="text-green-500">L۵.۳۴٪</span>
                <span className="text-green-500">C۵.۳۴٪</span>
              </div>
              <div className="text-xs text-red-500 mb-4">
                نرخ سود دارایی پایه <span className="ml-2">-۶.۸۲٪</span>
              </div>

              {/* Chart Area */}
              <div className="relative h-64 bg-muted/30 rounded-lg overflow-hidden">
                {/* Y-axis labels */}
                <div className="absolute right-2 top-0 bottom-0 flex flex-col justify-between text-xs text-muted-foreground py-4">
                  <span>۴٪</span>
                  <span>۰٪</span>
                  <span>-۴٪</span>
                  <span>-۸٪</span>
                  <span>-۱۲٪</span>
                </div>

                {/* Chart visualization */}
                <svg className="w-full h-full" viewBox="0 0 500 200" preserveAspectRatio="none">
                  {[0, 50, 100, 150, 200].map((y, i) => (
                    <line key={i} x1="0" y1={y} x2="500" y2={y} stroke="currentColor" strokeOpacity="0.1" strokeWidth="0.5" />
                  ))}
                  <path
                    d="M 0,150 L 50,160 L 100,140 L 150,120 L 200,145 L 250,130 L 300,90 L 350,75 L 400,60 L 450,45 L 480,55 L 500,85"
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="2"
                  />
                  <rect x="440" y="40" width="50" height="18" fill="rgb(34, 197, 94)" rx="3" />
                  <text x="465" y="53" fill="white" fontSize="10" textAnchor="middle">5.34%</text>
                  <rect x="440" y="80" width="50" height="18" fill="rgb(239, 68, 68)" rx="3" />
                  <text x="465" y="93" fill="white" fontSize="10" textAnchor="middle">-6.82%</text>
                </svg>

                {/* Timeframe selector */}
                <div className="absolute top-2 right-2 flex items-center gap-2">
                  {timeframes.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-2 py-1 text-xs rounded ${
                        selectedTimeframe === tf
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Center Panel - Order Book */}
        <div className="lg:col-span-3 min-w-0">
          <Card className="glass-card h-full min-w-0">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-4">
                <button className="text-sm font-medium border-b-2 border-primary pb-1">دفتر سفارش</button>
                <button className="text-sm text-muted-foreground hover:text-foreground pb-1">معاملات بازار</button>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              {/* Short Rate Section */}
              <div className="mb-4">
                <div className="text-xs text-red-500 font-semibold mb-2">نرخ شورت</div>
                <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                  <span>نرخ سود ضمنی (٪)</span>
                  <span>حجم (ETH YU)</span>
                </div>
                <div className="space-y-1">
                  {shortRateOrders.map((order, i) => (
                    <div key={i} className="flex items-center justify-between text-xs relative">
                      <div
                        className="absolute left-0 top-0 bottom-0 bg-red-500/20"
                        style={{ width: `${Math.min(order.size / 10, 100)}%` }}
                      />
                      <span className="text-red-500 relative z-10">{order.rate.toFixed(1)}</span>
                      <span className="relative z-10">{order.size.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Spread */}
              <div className="flex items-center justify-between text-xs py-2 border-y my-2">
                <span className="text-muted-foreground">اسپرد ۰.۱٪</span>
                <span className="text-cyan-500">بازه تشویقی: ۵.۱۶٪ - ۵.۶۲٪</span>
              </div>

              {/* Long Rate Section */}
              <div>
                <div className="text-xs text-green-500 font-semibold mb-2">نرخ لانگ</div>
                <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                  <span>نرخ سود ضمنی (٪)</span>
                  <span>حجم (ETH YU)</span>
                </div>
                <div className="space-y-1">
                  {longRateOrders.map((order, i) => (
                    <div key={i} className="flex items-center justify-between text-xs relative">
                      <div
                        className="absolute left-0 top-0 bottom-0 bg-green-500/20"
                        style={{ width: `${Math.min(order.size / 10, 100)}%` }}
                      />
                      <span className="text-green-500 relative z-10">{order.rate.toFixed(1)}</span>
                      <span className="relative z-10">{order.size.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Panel - Trading Form */}
        <div className="lg:col-span-3 min-w-0">
          <Card className="glass-card h-full min-w-0">
            <CardContent className="p-4">
              {/* Maker Rewards Banner */}
              <div className="bg-amber-500/20 border border-amber-500/30 rounded-lg p-3 mb-4 flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-400" />
                <span className="text-sm text-amber-200">پاداش سفارش‌های میکر فعال است!</span>
              </div>

              {/* Leverage Selector */}
              <div className="grid grid-cols-3 gap-2 mb-4">
                {['Cross', '2x', 'One-way'].map((lev) => (
                  <Button
                    key={lev}
                    variant={leverage === lev ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setLeverage(lev)}
                  >
                    {lev}
                  </Button>
                ))}
              </div>

              {/* Market/Limit Tabs */}
              <div className="grid grid-cols-2 gap-2 mb-4">
                <Button
                  variant={orderType === 'market' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setOrderType('market')}
                >
                  بازار
                </Button>
                <Button
                  variant={orderType === 'limit' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setOrderType('limit')}
                >
                  محدود
                </Button>
              </div>

              {/* Long/Short Rate Buttons */}
              <div className="grid grid-cols-2 gap-2 mb-4">
                <Button
                  onClick={() => setTradeDirection('long')}
                  className={`flex items-center gap-2 ${
                    tradeDirection === 'long'
                      ? 'bg-green-600 hover:bg-green-700'
                      : ''
                  }`}
                  variant={tradeDirection === 'long' ? 'default' : 'outline'}
                >
                  <TrendingUp className="h-4 w-4" />
                  <div className="text-left">
                    <div className="text-xs font-semibold">نرخ لانگ</div>
                    <div className="text-[10px] opacity-70">پرداخت ثابت</div>
                  </div>
                </Button>
                <Button
                  onClick={() => setTradeDirection('short')}
                  className={`flex items-center gap-2 ${
                    tradeDirection === 'short'
                      ? 'bg-red-600 hover:bg-red-700'
                      : ''
                  }`}
                  variant={tradeDirection === 'short' ? 'default' : 'outline'}
                >
                  <TrendingDown className="h-4 w-4" />
                  <div className="text-left">
                    <div className="text-xs font-semibold">نرخ شورت</div>
                    <div className="text-[10px] opacity-70">دریافت ثابت</div>
                  </div>
                </Button>
              </div>

              {/* Position Info */}
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">حجم اسمی من</span>
                  <span className="text-cyan-500">0 YU</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">قابل معامله</span>
                  <span>0 ETH</span>
                </div>
              </div>

              {/* Notional Size Input */}
              <div className="mb-4">
                <Label className="text-sm text-muted-foreground">حجم اسمی</Label>
                <div className="flex items-center gap-2 mt-1">
                  <Input
                    type="number"
                    value={notionalSize}
                    onChange={(e) => setNotionalSize(parseInt(e.target.value) || 0)}
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground">%</span>
                </div>
              </div>

              {/* Reduce Only */}
              <div className="flex items-center gap-2 mb-4">
                <Switch
                  checked={reduceOnly}
                  onCheckedChange={setReduceOnly}
                />
                <Label className="text-sm text-muted-foreground">فقط کاهش پوزیشن</Label>
              </div>

              {/* Trade Info */}
              <div className="space-y-2 text-sm border-t pt-4">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">نرخ سود ضمنی نقطه انحلال</span>
                  <span>—</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">مارجین موردنیاز</span>
                  <span>—</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">کارمزد</span>
                  <span>0 ETH ($0.00)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">لغزش قیمت</span>
                  <span>تخمینی: ۰٪ / حداکثر: <span className="text-cyan-500">۰.۵٪</span></span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
