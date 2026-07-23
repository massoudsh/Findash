'use client';

import { useState, useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  LineChart,
  Activity,
  Target,
} from 'lucide-react';

interface CandlestickData {
  date: string;
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
  data: CandlestickData[];
  symbol: string;
  timeframe?: string;
  className?: string;
}

type Timeframe = '1D' | '1W' | '1M' | '3M' | '1Y';
type Indicator = 'sma20' | 'sma50' | 'rsi' | 'macd' | 'volume';

const timeframes: Timeframe[] = ['1D', '1W', '1M', '3M', '1Y'];

export function AdvancedChart({ 
  data, 
  symbol, 
  timeframe = '1M', 
  className 
}: AdvancedChartProps) {
  console.log('AdvancedChart: Received data:', data?.length, 'points for symbol:', symbol);
  
  const [selectedTimeframe, setSelectedTimeframe] = useState<Timeframe>(timeframe as Timeframe);
  const [activeIndicators, setActiveIndicators] = useState<Set<Indicator>>(
    new Set(['sma20', 'volume'])
  );
  const [chartType, setChartType] = useState<'candlestick' | 'line'>('line');

  // Simplified data processing with error handling
  const processedData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      console.warn('Invalid data provided to AdvancedChart');
      return [];
    }

    try {
      return data.map((item, index) => {
        const result = { ...item };
        
        // Ensure numbers are valid
        if (typeof item.close !== 'number' || isNaN(item.close)) {
          console.warn(`Invalid close price at index ${index}:`, item.close);
          return result;
        }
        
        // Simple Moving Average 20
        if (index >= 19) {
          const validPrices = data.slice(index - 19, index + 1)
            .map(d => d.close)
            .filter(price => typeof price === 'number' && !isNaN(price));
          
          if (validPrices.length === 20) {
            result.sma20 = validPrices.reduce((sum, price) => sum + price, 0) / 20;
          }
        }
        
        // Simple Moving Average 50
        if (index >= 49) {
          const validPrices = data.slice(index - 49, index + 1)
            .map(d => d.close)
            .filter(price => typeof price === 'number' && !isNaN(price));
          
          if (validPrices.length === 50) {
            result.sma50 = validPrices.reduce((sum, price) => sum + price, 0) / 50;
          }
        }

        // Simplified RSI calculation
        if (index >= 14) {
          try {
            const period = 14;
            const changes = [];
            
            for (let i = 1; i <= period && (index - period + i) >= 0; i++) {
              const prevIndex = index - period + i - 1;
              const currIndex = index - period + i;
              if (data[prevIndex] && data[currIndex]) {
                changes.push(data[currIndex].close - data[prevIndex].close);
              }
            }
            
            if (changes.length === period) {
              const gains = changes.filter(c => c > 0);
              const losses = changes.filter(c => c < 0).map(c => Math.abs(c));
              
              const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
              const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
              
              if (avgLoss > 0) {
                const rs = avgGain / avgLoss;
                result.rsi = 100 - (100 / (1 + rs));
              }
            }
          } catch (error) {
            console.warn('Error calculating RSI:', error);
          }
        }

        return result;
      });
    } catch (error) {
      console.error('Error processing chart data:', error);
      return data; // Return original data if processing fails
    }
  }, [data]);

  const toggleIndicator = (indicator: Indicator) => {
    const newIndicators = new Set(activeIndicators);
    if (newIndicators.has(indicator)) {
      newIndicators.delete(indicator);
    } else {
      newIndicators.add(indicator);
    }
    setActiveIndicators(newIndicators);
  };

  const formatPrice = (value: number) => {
    if (typeof value !== 'number' || isNaN(value)) return '$0.00';
    return `$${value.toFixed(2)}`;
  };
  
  const formatVolume = (value: number) => {
    if (typeof value !== 'number' || isNaN(value)) return '0';
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toString();
  };

  // Safe data access
  const latestData = processedData.length > 0 ? processedData[processedData.length - 1] : null;
  const previousData = processedData.length > 1 ? processedData[processedData.length - 2] : null;
  
  const priceChange = latestData && previousData ? latestData.close - previousData.close : 0;
  const priceChangePercent = previousData && previousData.close !== 0 ? (priceChange / previousData.close) * 100 : 0;

  // Show loading/error state if no data
  if (!processedData || processedData.length === 0) {
    return (
      <Card className={`glass-card ${className}`}>
        <CardHeader>
          <CardTitle>در حال بارگذاری نمودار...</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Activity className="h-8 w-8 mx-auto mb-2 animate-pulse" />
              <p>در حال بارگذاری داده‌های نمودار...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`glass-card ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center space-x-2">
              <span className="text-2xl font-bold">{symbol}</span>
              {latestData && (
                <Badge
                  variant={priceChange >= 0 ? "default" : "destructive"}
                  className={priceChange >= 0 ? "bg-green-500/20 text-green-300" : ""}
                >
                  {priceChange >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                  {priceChangePercent.toFixed(2)}%
                </Badge>
              )}
            </CardTitle>
            {latestData && (
              <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                <span>قیمت: {formatPrice(latestData.close)}</span>
                <span>حجم: {formatVolume(latestData.volume)}</span>
                {latestData.rsi && <span>RSI: {latestData.rsi.toFixed(1)}</span>}
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant={chartType === 'line' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setChartType('line')}
            >
              <LineChart className="h-4 w-4" />
            </Button>
            <Button
              variant={chartType === 'candlestick' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setChartType('candlestick')}
            >
              <BarChart3 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Timeframe Selection */}
        <div className="flex items-center space-x-2">
          {timeframes.map((tf) => (
            <Button
              key={tf}
              variant={selectedTimeframe === tf ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedTimeframe(tf)}
              className="text-xs"
            >
              {tf}
            </Button>
          ))}
        </div>

        {/* Indicator Controls */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={activeIndicators.has('sma20') ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleIndicator('sma20')}
            className="text-xs"
          >
            SMA 20
          </Button>
          <Button
            variant={activeIndicators.has('sma50') ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleIndicator('sma50')}
            className="text-xs"
          >
            SMA 50
          </Button>
          <Button
            variant={activeIndicators.has('rsi') ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleIndicator('rsi')}
            className="text-xs"
          >
            RSI
          </Button>
          <Button
            variant={activeIndicators.has('volume') ? 'default' : 'outline'}
            size="sm"
            onClick={() => toggleIndicator('volume')}
            className="text-xs"
          >
            حجم
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="price" className="space-y-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="price">نمودار قیمت</TabsTrigger>
            <TabsTrigger value="rsi">RSI</TabsTrigger>
          </TabsList>

          <TabsContent value="price" className="space-y-4">
            {/* Main Price Chart */}
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart 
                  data={processedData} 
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: '#888' }}
                  />
                  <YAxis 
                    domain={['dataMin - 5', 'dataMax + 5']}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: '#888' }}
                    tickFormatter={formatPrice}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      border: '1px solid #333',
                      borderRadius: '8px',
                      color: 'white',
                    }}
                    formatter={(value: number, name: string) => [
                      name === 'volume' ? formatVolume(value) : formatPrice(value),
                      name.toUpperCase()
                    ]}
                  />
                  <Legend />

                  {/* Main price line */}
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#00D2FF"
                    strokeWidth={2}
                    dot={false}
                    name="قیمت"
                  />

                  {/* Moving Averages */}
                  {activeIndicators.has('sma20') && (
                    <Line
                      type="monotone"
                      dataKey="sma20"
                      stroke="#FF6B6B"
                      strokeWidth={1}
                      dot={false}
                      name="SMA 20"
                      strokeDasharray="5 5"
                    />
                  )}
                  {activeIndicators.has('sma50') && (
                    <Line
                      type="monotone"
                      dataKey="sma50"
                      stroke="#4ECDC4"
                      strokeWidth={1}
                      dot={false}
                      name="SMA 50"
                      strokeDasharray="5 5"
                    />
                  )}

                  {/* Volume bars */}
                  {activeIndicators.has('volume') && (
                    <Bar
                      dataKey="volume"
                      fill="rgba(255, 255, 255, 0.1)"
                      name="حجم"
                      yAxisId="volume"
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="rsi">
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart 
                  data={processedData} 
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="date" 
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: '#888' }}
                  />
                  <YAxis 
                    domain={[0, 100]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 12, fill: '#888' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(0, 0, 0, 0.8)',
                      border: '1px solid #333',
                      borderRadius: '8px',
                      color: 'white',
                    }}
                  />
                  <ReferenceLine y={70} stroke="#FF6B6B" strokeDasharray="3 3" />
                  <ReferenceLine y={30} stroke="#4ECDC4" strokeDasharray="3 3" />
                  <Line
                    type="monotone"
                    dataKey="rsi"
                    stroke="#FFD93D"
                    strokeWidth={2}
                    dot={false}
                    name="RSI"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
} 