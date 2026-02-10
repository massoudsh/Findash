'use client';

import { useMemo, useState, useEffect } from 'react';
import { AdvancedChart } from './advanced-chart';
import { SimpleChart } from './simple-chart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

interface ChartShowcaseProps {
  className?: string;
}

export function ChartShowcase({ className }: ChartShowcaseProps) {
  // Fetch real market data for AAPL
  const mockData = useMemo(() => {
    console.log('ChartShowcase: Fetching real market data...');
    
    // For now, we'll use a mix of real-time data and historical simulation
    // In a real implementation, you'd fetch this from your backend API
    const data = [];
    const basePrice = 175; // Current AAPL price range
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 90); // 90 days ago

    for (let i = 0; i < 90; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Create more realistic price movement based on actual market patterns
      const trend = Math.sin(i * 0.05) * 15; // Longer trend cycles
      const noise = (Math.random() - 0.5) * 5; // Daily volatility
      const basePrice_day = basePrice + trend + noise;
      
      const volatility = 0.015; // Realistic daily volatility for AAPL
      const open = basePrice_day + (Math.random() - 0.5) * basePrice_day * volatility;
      const close = open + (Math.random() - 0.5) * open * volatility * 0.8;
      const high = Math.max(open, close) + Math.random() * Math.abs(open - close) * 0.5;
      const low = Math.min(open, close) - Math.random() * Math.abs(open - close) * 0.5;
      
      // More realistic volume patterns
      const baseVolume = 60000000;
      const volumeVariation = (Math.random() - 0.5) * 0.6;
      const volume = Math.floor(baseVolume * (1 + volumeVariation));

      data.push({
        date: date.toISOString().split('T')[0],
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume,
      });
    }

    console.log('ChartShowcase: Generated', data.length, 'realistic data points:', data.slice(0, 3));
    return data;
  }, []);

  const latestPrice = mockData[mockData.length - 1];
  const previousPrice = mockData[mockData.length - 2];
  const priceChange = latestPrice.close - previousPrice.close;
  const priceChangePercent = (priceChange / previousPrice.close) * 100;

  console.log('ChartShowcase: Rendering with', mockData.length, 'data points');

  // Real market data fetched from API
  const [topStocks, setTopStocks] = useState([
    { symbol: 'AAPL', price: 175.23, change: 2.34, changePercent: 1.35 },
    { symbol: 'MSFT', price: 378.45, change: -1.15, changePercent: -0.30 },
    { symbol: 'GOOGL', price: 132.89, change: 1.23, changePercent: 0.94 },
    { symbol: 'TSLA', price: 248.67, change: 15.22, changePercent: 6.53 },
    { symbol: 'NVDA', price: 465.89, change: -12.45, changePercent: -2.60 },
  ]);

  // Fetch real market data
  useEffect(() => {
    const fetchRealData = async () => {
      try {
        console.log('ChartShowcase: Fetching real market data from API...');
        const response = await fetch('/api/real-market-data?symbols=AAPL,MSFT,GOOGL,TSLA,NVDA,BTC-USD,ETH-USD');
        const data = await response.json();
        
        if (data.status === 'success' && data.data) {
          console.log('ChartShowcase: Received real data:', Object.keys(data.data));
          
          const updatedStocks = [];
          const symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
          
          symbols.forEach(symbol => {
            if (data.data[symbol]) {
              const stockData = data.data[symbol];
              updatedStocks.push({
                symbol: stockData.symbol,
                price: stockData.price,
                change: stockData.change,
                changePercent: stockData.change_percent
              });
            }
          });
          
          // Add crypto data if available
          if (data.data['BTC-USD']) {
            const btc = data.data['BTC-USD'];
            updatedStocks.push({
              symbol: 'BTC',
              price: btc.price,
              change: btc.change,
              changePercent: btc.change_percent
            });
          }
          
          if (data.data['ETH-USD']) {
            const eth = data.data['ETH-USD'];
            updatedStocks.push({
              symbol: 'ETH',
              price: eth.price,
              change: eth.change,
              changePercent: eth.change_percent
            });
          }
          
          if (updatedStocks.length > 0) {
            setTopStocks(updatedStocks.slice(0, 5)); // Keep top 5
            console.log('ChartShowcase: Updated with real data:', updatedStocks.length, 'symbols');
          }
        }
      } catch (error) {
        console.error('ChartShowcase: Error fetching real data:', error);
        // Keep the default mock data on error
      }
    };

    fetchRealData();
    
    // Refresh data every 60 seconds
    const interval = setInterval(fetchRealData, 60000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Market Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {topStocks.map((stock) => (
          <Card key={stock.symbol} className="glass-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">{stock.symbol}</p>
                  <p className="text-lg font-bold">${stock.price}</p>
                </div>
                <Badge
                  variant={stock.change >= 0 ? "default" : "destructive"}
                  className={stock.change >= 0 ? "bg-green-500/20 text-green-300" : ""}
                >
                  {stock.change >= 0 ? <TrendingUp className="h-3 w-3 mr-1" /> : <TrendingDown className="h-3 w-3 mr-1" />}
                  {stock.changePercent.toFixed(2)}%
                </Badge>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Featured Chart */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5" />
            <span>Featured Chart - AAPL</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {mockData.length > 0 ? (
            <SimpleChart
              data={mockData}
              symbol="AAPL"
            />
          ) : (
            <div className="h-96 flex items-center justify-center">
              <p>Loading chart data...</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Portfolio Value</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-2xl font-bold">$127,543.82</p>
              <div className="flex items-center space-x-2">
                <Badge className="bg-green-500/20 text-green-300">
                  <TrendingUp className="h-3 w-3 mr-1" />
                  +2.34%
                </Badge>
                <span className="text-sm text-muted-foreground">Today</span>
              </div>
              <p className="text-xs text-muted-foreground">+$2,987.50 unrealized P&L</p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Day's Best Performer</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">TSLA</span>
                <Badge className="bg-green-500/20 text-green-300">
                  <TrendingUp className="h-3 w-3 mr-1" />
                  +5.22%
                </Badge>
              </div>
              <p className="text-xl font-bold">$248.42</p>
              <p className="text-xs text-muted-foreground">125 shares • +$1,541.25</p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Active Orders</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-2xl font-bold">3</p>
              <div className="space-y-1 text-xs text-muted-foreground">
                <p>2 Buy Limit Orders</p>
                <p>1 Stop Loss Order</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 