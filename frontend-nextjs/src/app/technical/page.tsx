'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { TradingViewChart } from '@/components/charts/tradingview-chart';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Target,
  Zap,
  Eye,
  Settings,
  LineChart,
  Gauge,
  AreaChart,
  BarChart,
  Calendar,
  Clock,
  Layers,
  Maximize2,
  Volume2,
  RefreshCw,
  Download,
  Share2,
  Bookmark,
  AlertTriangle,
  Minus,
  Star,
} from 'lucide-react';

const SYMBOL_TO_API = (s: string) => s.replace('/', '-');
const API_SYMBOLS = 'BTC-USD,ETH-USD,AAPL,TSLA,NVDA,MSFT,GOOGL';

export default function TechnicalPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');
  const [showVolume, setShowVolume] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartRefreshKey, setChartRefreshKey] = useState(0);
  const [realMarketData, setRealMarketData] = useState<Record<string, { price: number; open?: number; high?: number; low?: number; volume?: number; change?: number; change_percent?: number }>>({});
  const [activePanel, setActivePanel] = useState<'screener' | 'watchlist' | 'calendar' | 'chartSettings' | 'marketOverview' | null>(null);

  const fetchRealMarketData = useCallback(async () => {
    try {
      const res = await fetch(`/api/real-market-data?symbols=${API_SYMBOLS}`);
      const json = await res.json();
      if (json?.data) setRealMarketData(json.data);
    } catch (e) {
      console.error('Failed to fetch real market data', e);
    }
  }, []);

  useEffect(() => {
    fetchRealMarketData();
  }, [fetchRealMarketData]);

  const currentSymbolKey = SYMBOL_TO_API(selectedSymbol);
  const currentMarket = realMarketData[currentSymbolKey];

  // Mock price data for chart simulation
  const generateMockData = () => {
    const data = [];
    let price = 42350;
    for (let i = 0; i < 100; i++) {
      const change = (Math.random() - 0.5) * 1000;
      price += change;
      data.push({
        time: new Date(Date.now() - (100 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        open: price - Math.random() * 500,
        high: price + Math.random() * 800,
        low: price - Math.random() * 800,
        close: price,
        volume: Math.floor(Math.random() * 1000000) + 500000
      });
    }
    return data;
  };

  const chartData = generateMockData();

  const technicalIndicators = [
    { 
      name: 'RSI (14)', 
      value: 68.5, 
      signal: 'Overbought', 
      color: 'orange',
      description: 'Relative Strength Index',
      recommendation: 'Sell Signal'
    },
    { 
      name: 'MACD (12,26,9)', 
      value: 0.45, 
      signal: 'Bullish Cross', 
      color: 'green',
      description: 'Moving Average Convergence Divergence',
      recommendation: 'Buy Signal'
    },
    { 
      name: 'Stochastic (14,3,3)', 
      value: 75.2, 
      signal: 'Overbought', 
      color: 'red',
      description: 'Stochastic Oscillator',
      recommendation: 'Sell Signal'
    },
    { 
      name: 'Williams %R (14)', 
      value: -25.8, 
      signal: 'Bullish', 
      color: 'green',
      description: 'Williams Percent Range',
      recommendation: 'Buy Signal'
    },
    { 
      name: 'CCI (20)', 
      value: 145.6, 
      signal: 'Overbought', 
      color: 'orange',
      description: 'Commodity Channel Index',
      recommendation: 'Neutral'
    },
    { 
      name: 'ADX (14)', 
      value: 32.1, 
      signal: 'Strong Trend', 
      color: 'blue',
      description: 'Average Directional Index',
      recommendation: 'Trend Following'
    },
    { 
      name: 'Bollinger Bands', 
      value: 0.82, 
      signal: 'Upper Band', 
      color: 'purple',
      description: 'Volatility Indicator',
      recommendation: 'Sell Signal'
    },
    { 
      name: 'ATR (14)', 
      value: 1250.5, 
      signal: 'High Volatility', 
      color: 'yellow',
      description: 'Average True Range',
      recommendation: 'Caution'
    }
  ];

  const movingAverages = [
    { period: 'SMA 20', value: '$42,150', position: 'Above', signal: 'Bullish', distance: '+1.2%' },
    { period: 'SMA 50', value: '$41,850', position: 'Above', signal: 'Bullish', distance: '+2.8%' },
    { period: 'SMA 200', value: '$39,200', position: 'Above', signal: 'Bullish', distance: '+8.0%' },
    { period: 'EMA 12', value: '$42,280', position: 'Above', signal: 'Bullish', distance: '+0.9%' },
    { period: 'EMA 26', value: '$41,950', position: 'Above', signal: 'Bullish', distance: '+1.8%' },
    { period: 'EMA 50', value: '$41,650', position: 'Above', signal: 'Bullish', distance: '+2.5%' }
  ];

  const supportResistance = [
    { level: 'Resistance 3', price: '$45,200', distance: '+6.7%', strength: 'Strong', touches: 5 },
    { level: 'Resistance 2', price: '$43,800', distance: '+3.4%', strength: 'Medium', touches: 3 },
    { level: 'Resistance 1', price: '$42,900', distance: '+1.3%', strength: 'Weak', touches: 2 },
    { level: 'Current Price', price: '$42,350', distance: '0%', strength: 'Current', touches: 0 },
    { level: 'Support 1', price: '$41,200', distance: '-2.7%', strength: 'Strong', touches: 4 },
    { level: 'Support 2', price: '$39,800', distance: '-6.0%', strength: 'Medium', touches: 3 },
    { level: 'Support 3', price: '$38,100', distance: '-10.0%', strength: 'Strong', touches: 6 }
  ];

  const patterns = [
    { name: 'Bull Flag', probability: 75, timeframe: '4H', target: '$44,500', status: 'Active', confidence: 'High' },
    { name: 'Ascending Triangle', probability: 68, timeframe: '1D', target: '$45,200', status: 'Forming', confidence: 'Medium' },
    { name: 'Double Bottom', probability: 82, timeframe: '1W', target: '$47,800', status: 'Confirmed', confidence: 'High' },
    { name: 'Head & Shoulders', probability: 45, timeframe: '4H', target: '$39,500', status: 'Potential', confidence: 'Low' }
  ];

  const fibonacciLevels = [
    { level: '0%', price: '$38,100', type: 'Support' },
    { level: '23.6%', price: '$39,450', type: 'Support' },
    { level: '38.2%', price: '$40,850', type: 'Support' },
    { level: '50%', price: '$41,650', type: 'Pivot' },
    { level: '61.8%', price: '$42,450', type: 'Resistance' },
    { level: '78.6%', price: '$43,250', type: 'Resistance' },
    { level: '100%', price: '$45,200', type: 'Resistance' }
  ];

  const volumeProfile = [
    { price: '$42,000-42,500', volume: 2450000, percentage: 18.5 },
    { price: '$41,500-42,000', volume: 3200000, percentage: 24.2 },
    { price: '$41,000-41,500', volume: 1850000, percentage: 14.0 },
    { price: '$40,500-41,000', volume: 2100000, percentage: 15.9 },
    { price: '$40,000-40,500', volume: 1650000, percentage: 12.5 }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="space-y-6">
        {/* Header with Controls */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold gradient-text">Technical Analysis</h1>
            <p className="text-gray-400 mt-2">Professional charting platform with advanced technical analysis</p>
          </div>
          <div className="flex items-center space-x-3">
            <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse" />
              Live Data
            </Badge>
            <Button onClick={() => setIsFullscreen(!isFullscreen)} className="btn-morphic">
              <Maximize2 className="h-4 w-4 mr-2" />
              Fullscreen
            </Button>
          </div>
        </div>

        {/* Quick Actions - at top, all wired */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <Button
            className="btn-liquid text-white h-12"
            onClick={() => {
              setChartRefreshKey((k) => k + 1);
              fetchRealMarketData();
            }}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Data
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('screener')}>
            <Zap className="h-4 w-4 mr-2" />
            Screener
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('watchlist')}>
            <Target className="h-4 w-4 mr-2" />
            Watchlist
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('calendar')}>
            <Calendar className="h-4 w-4 mr-2" />
            Economic Calendar
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('chartSettings')}>
            <Settings className="h-4 w-4 mr-2" />
            Chart Settings
          </Button>
          <Button className="btn-morphic h-12" onClick={() => setActivePanel('marketOverview')}>
            <Eye className="h-4 w-4 mr-2" />
            Market Overview
          </Button>
        </div>

        {/* Panels (Sheet) for Screener, Watchlist, Calendar, Chart Settings, Market Overview */}
        <Sheet open={activePanel !== null} onOpenChange={(open) => !open && setActivePanel(null)}>
          <SheetContent side="right" className="w-full sm:max-w-md overflow-y-auto">
            <SheetHeader>
              <SheetTitle>
                {activePanel === 'screener' && 'Screener'}
                {activePanel === 'watchlist' && 'Watchlist'}
                {activePanel === 'calendar' && 'Economic Calendar'}
                {activePanel === 'chartSettings' && 'Chart Settings'}
                {activePanel === 'marketOverview' && 'Market Overview'}
              </SheetTitle>
            </SheetHeader>
            <div className="mt-6 space-y-4">
              {activePanel === 'screener' && (
                <>
                  <p className="text-sm text-gray-400">Filter symbols by price, volume, and technicals.</p>
                  <div className="space-y-2">
                    {['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'NVDA'].map((s) => (
                      <Button
                        key={s}
                        variant="outline"
                        className="w-full justify-between"
                        onClick={() => {
                          setSelectedSymbol(s);
                          setActivePanel(null);
                        }}
                      >
                        <span>{s}</span>
                        {realMarketData[SYMBOL_TO_API(s)] && (
                          <span className="text-green-400">${realMarketData[SYMBOL_TO_API(s)].price?.toLocaleString()}</span>
                        )}
                      </Button>
                    ))}
                  </div>
                </>
              )}
              {activePanel === 'watchlist' && (
                <>
                  <p className="text-sm text-gray-400">Your saved symbols. Click to load on chart.</p>
                  <div className="space-y-2">
                    {['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'].map((s) => (
                      <Button
                        key={s}
                        variant="outline"
                        className="w-full justify-between"
                        onClick={() => {
                          setSelectedSymbol(s);
                          setActivePanel(null);
                        }}
                      >
                        <Star className="h-4 w-4" />
                        <span>{s}</span>
                      </Button>
                    ))}
                  </div>
                </>
              )}
              {activePanel === 'calendar' && (
                <>
                  <p className="text-sm text-gray-400">Upcoming economic events (free tier).</p>
                  <ul className="space-y-2 text-sm">
                    <li className="flex justify-between p-2 rounded bg-slate-800/50">
                      <span>FOMC Statement</span>
                      <span className="text-gray-400">This week</span>
                    </li>
                    <li className="flex justify-between p-2 rounded bg-slate-800/50">
                      <span>Non-Farm Payrolls</span>
                      <span className="text-gray-400">Next Fri</span>
                    </li>
                    <li className="flex justify-between p-2 rounded bg-slate-800/50">
                      <span>CPI Release</span>
                      <span className="text-gray-400">Next month</span>
                    </li>
                  </ul>
                </>
              )}
              {activePanel === 'chartSettings' && (
                <>
                  <p className="text-sm text-gray-400">Chart type and timeframe are in the Trading Controls below. Fullscreen is in the header.</p>
                  <div className="space-y-2">
                    <p className="text-xs text-gray-500">Symbol: {selectedSymbol}</p>
                    <p className="text-xs text-gray-500">Timeframe: {selectedTimeframe}</p>
                  </div>
                </>
              )}
              {activePanel === 'marketOverview' && (
                <>
                  <p className="text-sm text-gray-400">Live snapshot from API.</p>
                  <div className="space-y-2">
                    {Object.entries(realMarketData).slice(0, 8).map(([sym, d]) => (
                      <div key={sym} className="flex justify-between items-center p-2 rounded bg-slate-800/50 text-sm">
                        <span>{sym}</span>
                        <span className={d.change_percent != null && d.change_percent >= 0 ? 'text-green-400' : 'text-red-400'}>
                          ${d.price?.toLocaleString()} ({d.change_percent != null ? (d.change_percent >= 0 ? '+' : '') + d.change_percent.toFixed(2) : '—'}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </SheetContent>
        </Sheet>

        {/* Trading Controls */}
        <Card className="glass-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center space-x-4">
                <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                  <SelectTrigger className="w-[140px] bg-slate-800/50 border-slate-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="BTC/USD">BTC/USD</SelectItem>
                    <SelectItem value="ETH/USD">ETH/USD</SelectItem>
                    <SelectItem value="AAPL">AAPL</SelectItem>
                    <SelectItem value="TSLA">TSLA</SelectItem>
                    <SelectItem value="NVDA">NVDA</SelectItem>
                  </SelectContent>
                </Select>

                <div className="flex space-x-1">
                  {['1m', '5m', '15m', '1H', '4H', '1D', '1W'].map((tf) => (
                    <Button
                      key={tf}
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-3 py-1 text-xs ${
                        selectedTimeframe === tf
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                      }`}
                    >
                      {tf}
                    </Button>
                  ))}
                </div>

                <div className="flex space-x-1">
                  {[
                    { type: 'candlestick', icon: BarChart3 },
                    { type: 'line', icon: LineChart },
                    { type: 'area', icon: AreaChart },
                    { type: 'bar', icon: BarChart }
                  ].map(({ type, icon: Icon }) => (
                    <Button
                      key={type}
                      onClick={() => setChartType(type)}
                      className={`p-2 ${
                        chartType === type
                          ? 'bg-blue-500 text-white'
                          : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                    </Button>
                  ))}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Button className="btn-morphic text-xs">
                  <Download className="h-3 w-3 mr-1" />
                  Export
                </Button>
                <Button className="btn-morphic text-xs">
                  <Share2 className="h-3 w-3 mr-1" />
                  Share
                </Button>
                <Button className="btn-morphic text-xs">
                  <Bookmark className="h-3 w-3 mr-1" />
                  Save
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional TradingView Chart */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Chart Panel */}
          <div className="xl:col-span-3">
            <TradingViewChart
              key={chartRefreshKey}
              symbol={selectedSymbol}
              interval={selectedTimeframe}
              height={600}
              autoRefresh={true}
            />
          </div>

          {/* Side Panel */}
          <div className="space-y-4">
            {/* Market Info - real data when available */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Market Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">24h Change</span>
                  <span className={currentMarket?.change_percent != null ? (currentMarket.change_percent >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-400'}>
                    {currentMarket?.change_percent != null
                      ? (currentMarket.change_percent >= 0 ? '+' : '') + currentMarket.change_percent.toFixed(2) + '%'
                      : '+2.45%'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">24h Volume</span>
                  <span>{currentMarket?.volume != null ? (currentMarket.volume / 1e6).toFixed(2) + 'M' : '$2.4B'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Price</span>
                  <span>{currentMarket?.price != null ? '$' + currentMarket.price.toLocaleString() : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Open</span>
                  <span>{currentMarket?.open != null ? '$' + currentMarket.open.toLocaleString() : '—'}</span>
                </div>
              </CardContent>
            </Card>

            {/* Quick Indicators */}
            <Card className="glass-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Quick Signals</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {technicalIndicators.slice(0, 4).map((indicator) => (
                  <div key={indicator.name} className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium">{indicator.name}</div>
                      <div className="text-xs text-gray-400">{indicator.recommendation}</div>
                    </div>
                    <Badge className={`text-xs ${
                      indicator.color === 'green' ? 'bg-green-500/20 text-green-300' :
                      indicator.color === 'red' ? 'bg-red-500/20 text-red-300' :
                      indicator.color === 'orange' ? 'bg-orange-500/20 text-orange-300' :
                      'bg-blue-500/20 text-blue-300'
                    }`}>
                      {indicator.signal}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Tabbed Analysis */}
        <Tabs defaultValue="indicators" className="w-full">
          <TabsList className="grid w-full grid-cols-6 bg-slate-800/50">
            <TabsTrigger value="indicators">Indicators</TabsTrigger>
            <TabsTrigger value="patterns">Patterns</TabsTrigger>
            <TabsTrigger value="levels">Levels</TabsTrigger>
            <TabsTrigger value="fibonacci">Fibonacci</TabsTrigger>
            <TabsTrigger value="volume">Volume</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
          </TabsList>

          <TabsContent value="indicators" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Gauge className="h-5 w-5 text-blue-400" />
                  <span>Technical Indicators</span>
                </CardTitle>
                <CardDescription>Comprehensive technical analysis indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {technicalIndicators.map((indicator, index) => (
                    <div key={indicator.name} className="neomorphic p-4 rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{indicator.name}</p>
                          <p className="text-xs text-gray-400">{indicator.description}</p>
                        </div>
                        <Badge className={`text-xs ${
                          indicator.color === 'green' ? 'bg-green-500/20 text-green-300' :
                          indicator.color === 'red' ? 'bg-red-500/20 text-red-300' :
                          indicator.color === 'orange' ? 'bg-orange-500/20 text-orange-300' :
                          indicator.color === 'purple' ? 'bg-purple-500/20 text-purple-300' :
                          indicator.color === 'yellow' ? 'bg-yellow-500/20 text-yellow-300' :
                          'bg-blue-500/20 text-blue-300'
                        }`}>
                          {indicator.signal}
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-lg font-bold">{indicator.value}</span>
                          <span className="text-xs text-gray-400">{indicator.recommendation}</span>
                        </div>
                        <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-1000 ${
                              indicator.color === 'green' ? 'bg-green-400' :
                              indicator.color === 'red' ? 'bg-red-400' :
                              indicator.color === 'orange' ? 'bg-orange-400' :
                              indicator.color === 'purple' ? 'bg-purple-400' :
                              indicator.color === 'yellow' ? 'bg-yellow-400' :
                              'bg-blue-400'
                            }`}
                            style={{ width: `${Math.min(Math.abs(indicator.value), 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="patterns" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5 text-cyan-400" />
                  <span>Chart Patterns</span>
                </CardTitle>
                <CardDescription>Detected patterns and price targets</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {patterns.map((pattern, index) => (
                    <div key={pattern.name} className="neomorphic p-6 rounded-lg space-y-4">
                      <div className="text-center">
                        <h3 className="font-bold text-lg">{pattern.name}</h3>
                        <p className="text-sm text-gray-400">{pattern.timeframe} • {pattern.status}</p>
                      </div>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Probability</span>
                          <span className="font-bold text-green-400">{pattern.probability}%</span>
                        </div>
                        <Progress value={pattern.probability} className="h-2" />
                        <div className="flex justify-between">
                          <span className="text-gray-400">Target</span>
                          <span className="font-bold text-blue-400">{pattern.target}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Confidence</span>
                          <Badge className={`text-xs ${
                            pattern.confidence === 'High' ? 'bg-green-500/20 text-green-300' :
                            pattern.confidence === 'Medium' ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>
                            {pattern.confidence}
                          </Badge>
                        </div>
                      </div>
                      <Button className="w-full btn-morphic">
                        <Eye className="h-4 w-4 mr-2" />
                        View on Chart
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="levels" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Support & Resistance */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="h-5 w-5 text-yellow-400" />
                    <span>Support & Resistance</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {supportResistance.map((level, index) => (
                      <div 
                        key={level.level} 
                        className={`flex items-center justify-between p-3 rounded-lg ${
                          level.level === 'Current Price' 
                            ? 'bg-blue-500/20 border border-blue-500/30' 
                            : 'neomorphic'
                        }`}
                      >
                        <div className="flex items-center space-x-4">
                          <span className={`font-medium w-24 text-sm ${
                            level.level.includes('Resistance') ? 'text-red-400' :
                            level.level.includes('Support') ? 'text-green-400' :
                            'text-blue-400'
                          }`}>
                            {level.level}
                          </span>
                          <span className="text-lg font-bold">{level.price}</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className={`text-sm ${
                            level.distance.startsWith('+') ? 'text-green-400' :
                            level.distance.startsWith('-') ? 'text-red-400' :
                            'text-gray-400'
                          }`}>
                            {level.distance}
                          </span>
                          <Badge className={`text-xs ${
                            level.strength === 'Strong' ? 'bg-red-500/20 text-red-300' :
                            level.strength === 'Medium' ? 'bg-yellow-500/20 text-yellow-300' :
                            level.strength === 'Current' ? 'bg-blue-500/20 text-blue-300' :
                            'bg-gray-500/20 text-gray-300'
                          }`}>
                            {level.touches} touches
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Moving Averages */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <LineChart className="h-5 w-5 text-purple-400" />
                    <span>Moving Averages</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {movingAverages.map((ma, index) => (
                      <div key={ma.period} className="flex items-center justify-between p-3 neomorphic rounded-lg">
                        <div className="flex items-center space-x-4">
                          <span className="font-medium w-16 text-sm">{ma.period}</span>
                          <span className="text-lg font-bold">{ma.value}</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <span className="text-sm text-green-400">{ma.distance}</span>
                          <Badge className={`text-xs ${
                            ma.signal === 'Bullish' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                          }`}>
                            {ma.signal}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="fibonacci" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Layers className="h-5 w-5 text-orange-400" />
                  <span>Fibonacci Retracement</span>
                </CardTitle>
                <CardDescription>Key Fibonacci levels for price analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {fibonacciLevels.map((fib, index) => (
                    <div key={fib.level} className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div className="flex items-center space-x-4">
                        <span className="font-medium w-16 text-sm">{fib.level}</span>
                        <span className="text-lg font-bold">{fib.price}</span>
                      </div>
                      <Badge className={`text-xs ${
                        fib.type === 'Support' ? 'bg-green-500/20 text-green-300' :
                        fib.type === 'Resistance' ? 'bg-red-500/20 text-red-300' :
                        'bg-blue-500/20 text-blue-300'
                      }`}>
                        {fib.type}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="volume" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Volume2 className="h-5 w-5 text-cyan-400" />
                  <span>Volume Profile</span>
                </CardTitle>
                <CardDescription>Volume distribution by price levels</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {volumeProfile.map((vol, index) => (
                    <div key={vol.price} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{vol.price}</span>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm">{vol.volume.toLocaleString()}</span>
                          <Badge className="bg-cyan-500/20 text-cyan-300 text-xs">
                            {vol.percentage}%
                          </Badge>
                        </div>
                      </div>
                      <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-cyan-400 transition-all duration-1000"
                          style={{ width: `${vol.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="alerts" className="space-y-4">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-400" />
                  <span>Price Alerts</span>
                </CardTitle>
                <CardDescription>Set up technical analysis alerts</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button className="btn-morphic h-12">
                    <Target className="h-4 w-4 mr-2" />
                    Price Alert
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Gauge className="h-4 w-4 mr-2" />
                    Indicator Alert
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Activity className="h-4 w-4 mr-2" />
                    Pattern Alert
                  </Button>
                  <Button className="btn-morphic h-12">
                    <Volume2 className="h-4 w-4 mr-2" />
                    Volume Alert
                  </Button>
                </div>
                
                <div className="space-y-3 mt-6">
                  <h4 className="font-medium">Active Alerts</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div>
                        <p className="font-medium">BTC/USD &gt; $43,000</p>
                        <p className="text-xs text-gray-400">Price Alert</p>
                      </div>
                      <Badge className="bg-green-500/20 text-green-300">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 neomorphic rounded-lg">
                      <div>
                        <p className="font-medium">RSI &lt; 30</p>
                        <p className="text-xs text-gray-400">Oversold Alert</p>
                      </div>
                      <Badge className="bg-yellow-500/20 text-yellow-300">Pending</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 