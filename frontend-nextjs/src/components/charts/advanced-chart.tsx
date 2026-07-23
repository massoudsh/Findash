'use client';

import { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Brush
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  CandlestickChart,
  Activity,
  Settings,
  Download,
  Maximize2,
  Eye,
  EyeOff,
  Plus,
  Minus,
  RotateCcw
} from 'lucide-react';

interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20?: number;
  sma50?: number;
  rsi?: number;
  macd?: number;
  signal?: number;
  histogram?: number;
}

interface AdvancedChartProps {
  symbol: string;
  data?: CandlestickData[];
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
}

type TimeframeType = '1D' | '1W' | '1M' | '3M' | '1Y';
type ChartType = 'candlestick' | 'line' | 'area';

// Generate mock data with technical indicators
const generateMockData = (days: number): CandlestickData[] => {
  const data: CandlestickData[] = [];
  let price = 150 + Math.random() * 50; // Starting price between 150-200
  
  for (let i = 0; i < days; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (days - i));
    
    // Generate OHLC with some realistic movement
    const openPrice = price + (Math.random() - 0.5) * 2;
    const closePrice = openPrice + (Math.random() - 0.5) * 4;
    const highPrice = Math.max(openPrice, closePrice) + Math.random() * 2;
    const lowPrice = Math.min(openPrice, closePrice) - Math.random() * 2;
    const volume = Math.floor(Math.random() * 1000000) + 500000;
    
    data.push({
      timestamp: date.toISOString().split('T')[0],
      open: Number(openPrice.toFixed(2)),
      high: Number(highPrice.toFixed(2)),
      low: Number(lowPrice.toFixed(2)),
      close: Number(closePrice.toFixed(2)),
      volume
    });
    
    price = closePrice;
  }
  
  // Calculate technical indicators
  return calculateTechnicalIndicators(data);
};

// Calculate technical indicators
const calculateTechnicalIndicators = (data: CandlestickData[]): CandlestickData[] => {
  const result = [...data];
  
  // Simple Moving Averages
  for (let i = 19; i < result.length; i++) {
    const sma20 = result.slice(i - 19, i + 1).reduce((sum, item) => sum + item.close, 0) / 20;
    result[i].sma20 = Number(sma20.toFixed(2));
  }
  
  for (let i = 49; i < result.length; i++) {
    const sma50 = result.slice(i - 49, i + 1).reduce((sum, item) => sum + item.close, 0) / 50;
    result[i].sma50 = Number(sma50.toFixed(2));
  }
  
  // RSI calculation (simplified)
  for (let i = 14; i < result.length; i++) {
    const gains = [];
    const losses = [];
    
    for (let j = 1; j <= 14; j++) {
      const change = result[i - j + 1].close - result[i - j].close;
      if (change > 0) gains.push(change);
      else losses.push(Math.abs(change));
    }
    
    const avgGain = gains.reduce((a, b) => a + b, 0) / 14;
    const avgLoss = losses.reduce((a, b) => a + b, 0) / 14;
    const rs = avgGain / (avgLoss || 1);
    result[i].rsi = Number((100 - (100 / (1 + rs))).toFixed(2));
  }
  
  // MACD calculation (simplified)
  for (let i = 25; i < result.length; i++) {
    const ema12 = result.slice(i - 11, i + 1).reduce((sum, item) => sum + item.close, 0) / 12;
    const ema26 = result.slice(i - 25, i + 1).reduce((sum, item) => sum + item.close, 0) / 26;
    result[i].macd = Number((ema12 - ema26).toFixed(2));
    
    if (i >= 34) {
      const signal = result.slice(i - 8, i + 1).reduce((sum, item) => sum + (item.macd || 0), 0) / 9;
      result[i].signal = Number(signal.toFixed(2));
      result[i].histogram = Number(((result[i].macd || 0) - signal).toFixed(2));
    }
  }
  
  return result;
};

export function AdvancedChart({ 
  symbol, 
  data, 
  height = 600, 
  showVolume = true, 
  showIndicators = true 
}: AdvancedChartProps) {
  const [timeframe, setTimeframe] = useState<TimeframeType>('1M');
  const [chartType, setChartType] = useState<ChartType>('candlestick');
  const [indicators, setIndicators] = useState({
    sma20: true,
    sma50: true,
    rsi: false,
    macd: false,
    volume: showVolume
  });
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Generate or use provided data
  const chartData = useMemo(() => {
    if (data) return data;
    
    const days = {
      '1D': 1,
      '1W': 7,
      '1M': 30,
      '3M': 90,
      '1Y': 365
    }[timeframe];
    
    return generateMockData(days);
  }, [data, timeframe]);

  const currentPrice = chartData[chartData.length - 1]?.close || 0;
  const previousPrice = chartData[chartData.length - 2]?.close || 0;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = (priceChange / previousPrice) * 100;

  const timeframeButtons: TimeframeType[] = ['1D', '1W', '1M', '3M', '1Y'];

  const toggleIndicator = (indicator: keyof typeof indicators) => {
    setIndicators(prev => ({
      ...prev,
      [indicator]: !prev[indicator]
    }));
  };

  // Custom Candlestick Component
  const CandlestickBar = (props: any) => {
    const { payload, x, y, width, height } = props;
    if (!payload) return null;
    
    const { open, close, high, low } = payload;
    const isPositive = close >= open;
    const color = isPositive ? '#10b981' : '#ef4444';
    const wickX = x + width / 2;
    
    return (
      <g>
        {/* Wick */}
        <line
          x1={wickX}
          y1={y}
          x2={wickX}
          y2={y + height}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x + width * 0.2}
          y={isPositive ? y + height * 0.3 : y + height * 0.1}
          width={width * 0.6}
          height={height * 0.4}
          fill={isPositive ? color : color}
          stroke={color}
          strokeWidth={1}
        />
      </g>
    );
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg">
          <p className="text-gray-300 text-sm mb-2">{label}</p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">باز شدن:</span>
              <span className="text-white">${data.open?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">بیشترین:</span>
              <span className="text-green-400">${data.high?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">کمترین:</span>
              <span className="text-red-400">${data.low?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">بسته شدن:</span>
              <span className="text-white">${data.close?.toFixed(2)}</span>
            </div>
            {data.volume && (
              <div className="flex justify-between gap-4">
                <span className="text-gray-400">حجم:</span>
                <span className="text-blue-400">{data.volume.toLocaleString('fa-IR')}</span>
              </div>
            )}
            {indicators.rsi && data.rsi && (
              <div className="flex justify-between gap-4">
                <span className="text-gray-400">RSI:</span>
                <span className="text-purple-400">{data.rsi.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-4">
            <CardTitle className="text-2xl font-bold">{symbol}</CardTitle>
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold">
                ${currentPrice.toFixed(2)}
              </span>
              <div className={`flex items-center space-x-1 ${
                priceChange >= 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {priceChange >= 0 ? 
                  <TrendingUp className="h-4 w-4" /> : 
                  <TrendingDown className="h-4 w-4" />
                }
                <span className="font-medium">
                  {priceChange >= 0 ? '+' : ''}
                  ${priceChange.toFixed(2)} ({priceChangePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4 pt-4 border-t">
          {/* Timeframe Selection */}
          <div className="flex items-center space-x-1">
            {timeframeButtons.map((tf) => (
              <Button
                key={tf}
                variant={timeframe === tf ? "default" : "outline"}
                size="sm"
                onClick={() => setTimeframe(tf)}
              >
                {tf}
              </Button>
            ))}
          </div>
          
          {/* Chart Type */}
          <div className="flex items-center space-x-1">
            <Button
              variant={chartType === 'candlestick' ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType('candlestick')}
            >
              <CandlestickChart className="h-4 w-4" />
            </Button>
            <Button
              variant={chartType === 'line' ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType('line')}
            >
              <Activity className="h-4 w-4" />
            </Button>
            <Button
              variant={chartType === 'area' ? "default" : "outline"}
              size="sm"
              onClick={() => setChartType('area')}
            >
              <BarChart3 className="h-4 w-4" />
            </Button>
          </div>
          
          {/* Indicators */}
          {showIndicators && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-muted-foreground">اندیکاتورها:</span>
              {Object.entries(indicators).map(([key, enabled]) => (
                <Button
                  key={key}
                  variant={enabled ? "default" : "outline"}
                  size="sm"
                  onClick={() => toggleIndicator(key as keyof typeof indicators)}
                >
                  {enabled ? <Eye className="h-3 w-3 mr-1" /> : <EyeOff className="h-3 w-3 mr-1" />}
                  {key === 'volume' ? 'حجم' : key.toUpperCase()}
                </Button>
              ))}
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        <div style={{ width: '100%', height: isFullscreen ? '80vh' : height }}>
          <ResponsiveContainer>
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#9ca3af"
                fontSize={12}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis 
                stroke="#9ca3af"
                fontSize={12}
                domain={['dataMin - 5', 'dataMax + 5']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Volume bars (if enabled) */}
              {indicators.volume && (
                <Bar
                  dataKey="volume"
                  fill="#6b7280"
                  fillOpacity={0.3}
                  yAxisId="volume"
                />
              )}
              
              {/* Main price chart */}
              {chartType === 'line' && (
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
              )}
              
              {chartType === 'area' && (
                <Area
                  type="monotone"
                  dataKey="close"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.2}
                />
              )}
              
              {/* Technical Indicators */}
              {indicators.sma20 && (
                <Line
                  type="monotone"
                  dataKey="sma20"
                  stroke="#f59e0b"
                  strokeWidth={1}
                  dot={false}
                  name="SMA 20"
                />
              )}
              
              {indicators.sma50 && (
                <Line
                  type="monotone"
                  dataKey="sma50"
                  stroke="#3b82f6"
                  strokeWidth={1}
                  dot={false}
                  name="SMA 50"
                />
              )}
              
              <Brush dataKey="timestamp" height={30} stroke="#6b7280" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        {/* Additional indicator charts */}
        {(indicators.rsi || indicators.macd) && (
          <div className="mt-6 space-y-4">
            {indicators.rsi && (
              <div>
                <h4 className="text-sm font-medium mb-2">RSI (14)</h4>
                <div style={{ width: '100%', height: 150 }}>
                  <ResponsiveContainer>
                    <ComposedChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="timestamp" 
                        stroke="#9ca3af"
                        fontSize={10}
                        tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <YAxis 
                        stroke="#9ca3af"
                        fontSize={10}
                        domain={[0, 100]}
                      />
                      <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="2 2" />
                      <ReferenceLine y={30} stroke="#10b981" strokeDasharray="2 2" />
                      <Line
                        type="monotone"
                        dataKey="rsi"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            
            {indicators.macd && (
              <div>
                <h4 className="text-sm font-medium mb-2">MACD</h4>
                <div style={{ width: '100%', height: 150 }}>
                  <ResponsiveContainer>
                    <ComposedChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="timestamp" 
                        stroke="#9ca3af"
                        fontSize={10}
                        tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <YAxis stroke="#9ca3af" fontSize={10} />
                      <Bar dataKey="histogram" fill="#6b7280" />
                      <Line
                        type="monotone"
                        dataKey="macd"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="signal"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 