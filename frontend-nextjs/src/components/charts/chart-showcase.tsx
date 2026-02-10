'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AdvancedChart } from './advanced-chart';
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  Activity,
  Target,
  Sparkles,
  Eye,
  Settings,
  Download,
  RefreshCw,
  Star,
  ArrowUp,
  ArrowDown
} from 'lucide-react';

interface MarketOverview {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  marketCap: string;
  pe?: number;
  category: 'tech' | 'finance' | 'healthcare' | 'energy' | 'consumer';
  trending: boolean;
}

const marketData: MarketOverview[] = [
  {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 185.92,
    change: 2.47,
    changePercent: 1.35,
    volume: '45.2M',
    marketCap: '2.89T',
    pe: 28.4,
    category: 'tech',
    trending: true
  },
  {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    price: 378.85,
    change: -1.23,
    changePercent: -0.32,
    volume: '23.1M',
    marketCap: '2.81T',
    pe: 32.1,
    category: 'tech',
    trending: true
  },
  {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    price: 142.56,
    change: 0.89,
    changePercent: 0.63,
    volume: '28.7M',
    marketCap: '1.78T',
    pe: 25.8,
    category: 'tech',
    trending: false
  },
  {
    symbol: 'TSLA',
    name: 'Tesla Inc.',
    price: 248.42,
    change: 8.92,
    changePercent: 3.73,
    volume: '67.3M',
    marketCap: '785B',
    pe: 45.2,
    category: 'consumer',
    trending: true
  },
  {
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    price: 875.28,
    change: 15.47,
    changePercent: 1.80,
    volume: '41.8M',
    marketCap: '2.16T',
    pe: 58.7,
    category: 'tech',
    trending: true
  },
  {
    symbol: 'JPM',
    name: 'JPMorgan Chase & Co.',
    price: 164.73,
    change: -0.82,
    changePercent: -0.49,
    volume: '12.4M',
    marketCap: '482B',
    pe: 12.3,
    category: 'finance',
    trending: false
  }
];

const categoryColors = {
  tech: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  finance: 'bg-green-500/20 text-green-300 border-green-500/30',
  healthcare: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  energy: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  consumer: 'bg-pink-500/20 text-pink-300 border-pink-500/30'
};

export function ChartShowcase() {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [viewMode, setViewMode] = useState<'grid' | 'featured'>('featured');

  const selectedStock = marketData.find(stock => stock.symbol === selectedSymbol) || marketData[0];
  const trendingStocks = marketData.filter(stock => stock.trending);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Advanced Charts</h1>
          <p className="text-muted-foreground">
            Professional-grade charting with technical analysis and real-time data
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant={viewMode === 'featured' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setViewMode('featured')}
          >
            <Eye className="h-4 w-4 mr-2" />
            Featured
          </Button>
          <Button 
            variant={viewMode === 'grid' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            Grid View
          </Button>
        </div>
      </div>

      {/* Market Overview Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        {marketData.map((stock) => (
          <Card 
            key={stock.symbol}
            className={`cursor-pointer transition-all duration-200 hover:scale-105 ${
              selectedSymbol === stock.symbol ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedSymbol(stock.symbol)}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm">{stock.symbol}</span>
                {stock.trending && (
                  <Badge className="bg-orange-500/20 text-orange-300 border-orange-500/30">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    Hot
                  </Badge>
                )}
              </div>
              <div className="space-y-1">
                <div className="text-lg font-bold">${stock.price.toFixed(2)}</div>
                <div className={`flex items-center text-sm ${
                  stock.change >= 0 ? 'text-green-500' : 'text-red-500'
                }`}>
                  {stock.change >= 0 ? 
                    <ArrowUp className="h-3 w-3 mr-1" /> : 
                    <ArrowDown className="h-3 w-3 mr-1" />
                  }
                  {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} 
                  ({stock.changePercent.toFixed(2)}%)
                </div>
                <Badge className={categoryColors[stock.category]}>
                  {stock.category}
                </Badge>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {viewMode === 'featured' ? (
        <div className="space-y-6">
          {/* Featured Chart */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-6 w-6" />
                    <span>{selectedStock.name} ({selectedStock.symbol})</span>
                  </CardTitle>
                  <Badge className={categoryColors[selectedStock.category]}>
                    {selectedStock.category.toUpperCase()}
                  </Badge>
                  {selectedStock.trending && (
                    <Badge className="bg-orange-500/20 text-orange-300 border-orange-500/30">
                      <Sparkles className="h-3 w-3 mr-1" />
                      Trending
                    </Badge>
                  )}
                </div>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <div>Vol: {selectedStock.volume}</div>
                  <div>•</div>
                  <div>Mkt Cap: {selectedStock.marketCap}</div>
                  {selectedStock.pe && (
                    <>
                      <div>•</div>
                      <div>P/E: {selectedStock.pe}</div>
                    </>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <AdvancedChart
                symbol={selectedStock.symbol}
                height={600}
                showVolume={true}
                showIndicators={true}
              />
            </CardContent>
          </Card>

          {/* Portfolio Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PieChart className="h-5 w-5" />
                  <span>Portfolio Breakdown</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {trendingStocks.slice(0, 4).map((stock, index) => (
                    <div key={stock.symbol} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${
                          index === 0 ? 'bg-blue-500' :
                          index === 1 ? 'bg-green-500' :
                          index === 2 ? 'bg-purple-500' :
                          'bg-orange-500'
                        }`} />
                        <div>
                          <div className="font-medium text-sm">{stock.symbol}</div>
                          <div className="text-xs text-muted-foreground">{stock.name}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-sm">${stock.price.toFixed(2)}</div>
                        <div className={`text-xs ${
                          stock.change >= 0 ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {stock.changePercent.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="h-5 w-5" />
                  <span>Performance Metrics</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Total Return</span>
                    <span className="font-bold text-green-500">+12.45%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Daily P&L</span>
                    <span className="font-bold text-green-500">+$2,847.32</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Win Rate</span>
                    <span className="font-bold">68.2%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Sharpe Ratio</span>
                    <span className="font-bold">1.84</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Max Drawdown</span>
                    <span className="font-bold text-red-500">-5.23%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5" />
                  <span>Market Alerts</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-green-500 mt-2" />
                    <div>
                      <div className="text-sm font-medium">AAPL Target Hit</div>
                      <div className="text-xs text-muted-foreground">Price reached $185.00 resistance</div>
                      <div className="text-xs text-green-500">2 minutes ago</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-blue-500 mt-2" />
                    <div>
                      <div className="text-sm font-medium">RSI Oversold</div>
                      <div className="text-xs text-muted-foreground">MSFT RSI below 30</div>
                      <div className="text-xs text-blue-500">15 minutes ago</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-orange-500 mt-2" />
                    <div>
                      <div className="text-sm font-medium">Volume Spike</div>
                      <div className="text-xs text-muted-foreground">TSLA volume 3x average</div>
                      <div className="text-xs text-orange-500">1 hour ago</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        /* Grid View */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {marketData.slice(0, 4).map((stock) => (
            <Card key={stock.symbol}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{stock.symbol}</CardTitle>
                  <Badge className={categoryColors[stock.category]}>
                    {stock.category}
                  </Badge>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-xl font-bold">${stock.price.toFixed(2)}</span>
                  <div className={`flex items-center text-sm ${
                    stock.change >= 0 ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {stock.change >= 0 ? 
                      <ArrowUp className="h-3 w-3 mr-1" /> : 
                      <ArrowDown className="h-3 w-3 mr-1" />
                    }
                    {stock.changePercent.toFixed(2)}%
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <AdvancedChart
                  symbol={stock.symbol}
                  height={300}
                  showVolume={false}
                  showIndicators={false}
                />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Chart Types Demonstration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>Chart Types & Indicators</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="candlestick" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="candlestick">Candlestick</TabsTrigger>
              <TabsTrigger value="technical">Technical Analysis</TabsTrigger>
              <TabsTrigger value="volume">Volume Analysis</TabsTrigger>
              <TabsTrigger value="indicators">Indicators</TabsTrigger>
            </TabsList>
            
            <TabsContent value="candlestick" className="mt-6">
              <AdvancedChart
                symbol="DEMO"
                height={400}
                showVolume={true}
                showIndicators={false}
              />
            </TabsContent>
            
            <TabsContent value="technical" className="mt-6">
              <AdvancedChart
                symbol="DEMO"
                height={400}
                showVolume={false}
                showIndicators={true}
              />
            </TabsContent>
            
            <TabsContent value="volume" className="mt-6">
              <AdvancedChart
                symbol="DEMO"
                height={400}
                showVolume={true}
                showIndicators={false}
              />
            </TabsContent>
            
            <TabsContent value="indicators" className="mt-6">
              <div className="space-y-4">
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-green-500">72.4</div>
                      <div className="text-sm text-muted-foreground">RSI</div>
                      <Badge className="bg-red-500/20 text-red-300">Overbought</Badge>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-blue-500">1.23</div>
                      <div className="text-sm text-muted-foreground">MACD</div>
                      <Badge className="bg-green-500/20 text-green-300">Bullish</Badge>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-purple-500">24.8</div>
                      <div className="text-sm text-muted-foreground">Bollinger %B</div>
                      <Badge className="bg-yellow-500/20 text-yellow-300">Neutral</Badge>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl font-bold text-orange-500">45.2</div>
                      <div className="text-sm text-muted-foreground">Williams %R</div>
                      <Badge className="bg-blue-500/20 text-blue-300">Oversold</Badge>
                    </CardContent>
                  </Card>
                </div>
                <AdvancedChart
                  symbol="DEMO"
                  height={400}
                  showVolume={false}
                  showIndicators={true}
                />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
} 