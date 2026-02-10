'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface SimpleChartProps {
  data: Array<{
    date: string;
    close: number;
    volume: number;
  }>;
  symbol: string;
  className?: string;
}

export function SimpleChart({ data, symbol, className }: SimpleChartProps) {
  console.log('SimpleChart: Rendering with data:', data?.length, 'points');

  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data)) {
      console.warn('SimpleChart: Invalid data provided');
      return [];
    }

    return data.slice(-30).map((item, index) => ({
      ...item,
      index,
      shortDate: new Date(item.date).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      })
    }));
  }, [data]);

  const latestPrice = chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const previousPrice = chartData.length > 1 ? chartData[chartData.length - 2] : null;
  
  const priceChange = latestPrice && previousPrice ? latestPrice.close - previousPrice.close : 0;
  const priceChangePercent = previousPrice && previousPrice.close !== 0 ? (priceChange / previousPrice.close) * 100 : 0;

  if (!chartData || chartData.length === 0) {
    return (
      <Card className={`glass-card ${className}`}>
        <CardHeader>
          <CardTitle>{symbol} - No Data Available</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <p>No chart data to display</p>
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
              {latestPrice && (
                <Badge
                  variant={priceChange >= 0 ? "default" : "destructive"}
                  className={priceChange >= 0 ? "bg-green-500/20 text-green-300" : ""}
                >
                  {priceChange >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                  {priceChangePercent.toFixed(2)}%
                </Badge>
              )}
            </CardTitle>
            {latestPrice && (
              <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                <span>Price: ${latestPrice.close.toFixed(2)}</span>
                <span>Volume: {(latestPrice.volume / 1000000).toFixed(1)}M</span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          {/* Price Chart */}
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="shortDate" 
                  stroke="#9CA3AF"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={12}
                  tickFormatter={(value) => `$${value.toFixed(0)}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F9FAFB'
                  }}
                  formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
                />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: '#3B82F6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Volume Chart */}
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="shortDate" 
                  stroke="#9CA3AF"
                  fontSize={10}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={10}
                  tickFormatter={(value) => `${(value / 1000000).toFixed(0)}M`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F9FAFB'
                  }}
                  formatter={(value: number) => [`${(value / 1000000).toFixed(1)}M`, 'Volume']}
                />
                <Bar
                  dataKey="volume"
                  fill="#6366F1"
                  opacity={0.7}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 