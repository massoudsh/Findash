'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart3, 
  CandlestickChart,
  LineChart,
  TrendingUp,
  TrendingDown,
  Volume2,
  Settings,
  Maximize2,
  Download,
  Share2,
  Layers,
  Target,
  Zap,
  Eye,
  EyeOff,
  RefreshCw,
  Play,
  Pause,
  Clock
} from 'lucide-react';

interface TradingViewChartProps {
  symbol: string;
  interval?: string;
  height?: number;
  autoRefresh?: boolean;
}

interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TechnicalIndicator {
  id: string;
  name: string;
  enabled: boolean;
  type: 'overlay' | 'oscillator';
  params: Record<string, any>;
  data?: number[];
}

export function TradingViewChart({ 
  symbol = 'BTCUSDT', 
  interval = '1D', 
  height = 600,
  autoRefresh = true 
}: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chartData, setChartData] = useState<OHLCV[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [selectedInterval, setSelectedInterval] = useState(interval);
  const [showVolume, setShowVolume] = useState(true);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(autoRefresh);
  
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([
    { id: 'sma20', name: 'SMA 20', enabled: true, type: 'overlay', params: { period: 20 } },
    { id: 'sma50', name: 'SMA 50', enabled: true, type: 'overlay', params: { period: 50 } },
    { id: 'ema12', name: 'EMA 12', enabled: false, type: 'overlay', params: { period: 12 } },
    { id: 'bb', name: 'Bollinger Bands', enabled: false, type: 'overlay', params: { period: 20, stdDev: 2 } },
    { id: 'rsi', name: 'RSI', enabled: false, type: 'oscillator', params: { period: 14 } },
    { id: 'macd', name: 'MACD', enabled: false, type: 'oscillator', params: { fast: 12, slow: 26, signal: 9 } },
    { id: 'stoch', name: 'Stochastic', enabled: false, type: 'oscillator', params: { k: 14, d: 3 } },
  ]);

  const intervals = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' },
    { value: '4h', label: '4h' },
    { value: '1D', label: '1D' },
    { value: '1W', label: '1W' },
    { value: '1M', label: '1M' }
  ];

  // Generate realistic OHLCV data
  const generateChartData = useCallback((symbol: string, interval: string, points: number = 200): OHLCV[] => {
    const data: OHLCV[] = [];
    let price = 42000 + Math.random() * 8000; // Starting price
    let timestamp = Date.now() - (points * getIntervalMs(interval));
    
    for (let i = 0; i < points; i++) {
      const volatility = 0.02; // 2% volatility
      const trend = Math.sin(i / 20) * 0.001; // Subtle trend
      
      const change = (Math.random() - 0.5) * volatility + trend;
      const open = price;
      const close = price * (1 + change);
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      const volume = Math.floor(Math.random() * 1000000) + 500000;
      
      data.push({
        timestamp,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume
      });
      
      price = close;
      timestamp += getIntervalMs(interval);
    }
    
    return data;
  }, []);

  const getIntervalMs = (interval: string): number => {
    const intervalMap: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1D': 24 * 60 * 60 * 1000,
      '1W': 7 * 24 * 60 * 60 * 1000,
      '1M': 30 * 24 * 60 * 60 * 1000
    };
    return intervalMap[interval] || 24 * 60 * 60 * 1000;
  };

  // Calculate technical indicators
  const calculateSMA = (data: OHLCV[], period: number): number[] => {
    const sma: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        sma.push(NaN);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((acc, curr) => acc + curr.close, 0);
        sma.push(sum / period);
      }
    }
    return sma;
  };

  const calculateEMA = (data: OHLCV[], period: number): number[] => {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        ema.push(data[i].close);
      } else {
        ema.push((data[i].close - ema[i - 1]) * multiplier + ema[i - 1]);
      }
    }
    return ema;
  };

  const calculateRSI = (data: OHLCV[], period: number): number[] => {
    const rsi: number[] = [];
    const gains: number[] = [];
    const losses: number[] = [];
    
    for (let i = 1; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    for (let i = 0; i < gains.length; i++) {
      if (i < period - 1) {
        rsi.push(NaN);
      } else {
        const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
        const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
        const rs = avgGain / (avgLoss || 1);
        rsi.push(100 - (100 / (1 + rs)));
      }
    }
    
    return [NaN, ...rsi]; // Add NaN for first data point
  };

  // Load chart data
  const loadChartData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const newData = generateChartData(symbol, selectedInterval);
      setChartData(newData);
      
      // Calculate indicators
      setIndicators(prev => prev.map(indicator => {
        let data: number[] = [];
        
        switch (indicator.id) {
          case 'sma20':
            data = calculateSMA(newData, 20);
            break;
          case 'sma50':
            data = calculateSMA(newData, 50);
            break;
          case 'ema12':
            data = calculateEMA(newData, 12);
            break;
          case 'rsi':
            data = calculateRSI(newData, 14);
            break;
          // Add more indicators as needed
        }
        
        return { ...indicator, data };
      }));
      
    } catch (error) {
      console.error('Error loading chart data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [symbol, selectedInterval, generateChartData]);

  // Auto-refresh logic
  useEffect(() => {
    if (autoRefreshEnabled) {
      const interval = setInterval(loadChartData, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefreshEnabled, loadChartData]);

  // Initial data load
  useEffect(() => {
    loadChartData();
  }, [loadChartData]);

  const toggleIndicator = (indicatorId: string) => {
    setIndicators(prev => prev.map(ind => 
      ind.id === indicatorId ? { ...ind, enabled: !ind.enabled } : ind
    ));
  };

  const currentPrice = chartData[chartData.length - 1]?.close || 0;
  const previousPrice = chartData[chartData.length - 2]?.close || 0;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = (priceChange / previousPrice) * 100;
  const isPositive = priceChange >= 0;

  return (
    <Card className={`glass-card ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <CardTitle className="text-xl font-bold">{symbol}</CardTitle>
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold">
                ${currentPrice.toLocaleString()}
              </span>
              <div className={`flex items-center space-x-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                <span className="font-medium">
                  {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
            <Badge className="bg-green-500/20 text-green-300">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse" />
              زنده
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Interval Selector */}
            <Select value={selectedInterval} onValueChange={setSelectedInterval}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {intervals.map(int => (
                  <SelectItem key={int.value} value={int.value}>
                    {int.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            {/* Chart Type Buttons */}
            <div className="flex rounded-lg bg-gray-800 p-1">
              <Button
                variant={chartType === 'candlestick' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setChartType('candlestick')}
              >
                <CandlestickChart className="h-4 w-4" />
              </Button>
              <Button
                variant={chartType === 'line' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setChartType('line')}
              >
                <LineChart className="h-4 w-4" />
              </Button>
              <Button
                variant={chartType === 'area' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setChartType('area')}
              >
                <BarChart3 className="h-4 w-4" />
              </Button>
            </div>
            
            {/* Control Buttons */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoRefreshEnabled(!autoRefreshEnabled)}
            >
              {autoRefreshEnabled ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            
            <Button variant="ghost" size="sm" onClick={() => setShowVolume(!showVolume)}>
              <Volume2 className={`h-4 w-4 ${showVolume ? 'text-blue-400' : 'text-gray-400'}`} />
            </Button>
            
            <Button variant="ghost" size="sm" onClick={loadChartData} disabled={isLoading}>
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            
            <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(!isFullscreen)}>
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="space-y-4">
          {/* Indicators Panel */}
          <div className="flex flex-wrap gap-2">
            {indicators.map(indicator => (
              <Button
                key={indicator.id}
                variant={indicator.enabled ? 'default' : 'outline'}
                size="sm"
                onClick={() => toggleIndicator(indicator.id)}
                className="h-8 text-xs"
              >
                {indicator.name}
              </Button>
            ))}
          </div>
          
          {/* Chart Container */}
          <div 
            ref={chartContainerRef}
            className="relative bg-gray-900/50 rounded-lg border border-gray-700"
            style={{ height: isFullscreen ? 'calc(100vh - 200px)' : height }}
          >
            {isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="flex items-center space-x-2">
                  <RefreshCw className="h-6 w-6 animate-spin text-blue-400" />
                  <span className="text-gray-400">در حال بارگذاری داده‌های نمودار...</span>
                </div>
              </div>
            ) : (
              <div className="p-4 h-full">
                {/* Chart will be rendered here */}
                <div className="w-full h-full flex items-center justify-center text-gray-400">
                  <div className="text-center">
                    <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-600" />
                    <p>نمودار حرفه‌ای TradingView</p>
                    <p className="text-sm mt-2">
                      {chartData.length} نقطه داده • نمای {chartType} • بازه {selectedInterval}
                    </p>
                    {indicators.filter(i => i.enabled).length > 0 && (
                      <p className="text-xs mt-1">
                        اندیکاتورهای فعال: {indicators.filter(i => i.enabled).map(i => i.name).join('، ')}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Volume Chart */}
          {showVolume && (
            <div className="h-24 bg-gray-900/30 rounded-lg border border-gray-700 p-2">
              <div className="flex items-center justify-center h-full text-gray-400 text-sm">
                <Volume2 className="h-4 w-4 mr-2" />
                نمودار حجم ({chartData.length} کندل)
              </div>
            </div>
          )}
          
          {/* Market Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">بیشترین ۲۴ ساعت</div>
              <div className="font-semibold">${Math.max(...chartData.map(d => d.high)).toLocaleString()}</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">کمترین ۲۴ ساعت</div>
              <div className="font-semibold">${Math.min(...chartData.map(d => d.low)).toLocaleString()}</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">حجم ۲۴ ساعت</div>
              <div className="font-semibold">{(chartData.reduce((sum, d) => sum + d.volume, 0) / 1e6).toFixed(1)}M</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">ارزش بازار</div>
              <div className="font-semibold">$821.5B</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">نوسان</div>
              <div className="font-semibold">2.4%</div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">آخرین به‌روزرسانی</div>
              <div className="font-semibold text-xs">{new Date().toLocaleTimeString('fa-IR')}</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 