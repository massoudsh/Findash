'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Map } from 'lucide-react';
import { DataCollectorAgentPanel } from '@/components/trading/data-collector-agent-panel';
import { MarketMap } from '@/components/market/market-map';

interface MarketDataItem {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap?: string;
  category: 'stocks' | 'crypto' | 'stablecoins' | 'commodities_etfs';
}

const SYMBOLS = [
  'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
  'BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD',
  'USDT-USD', 'USDC-USD',
  'GLD', 'SLV', 'SPY', 'QQQ',
];

export default function MarketDataPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [marketData, setMarketData] = useState<MarketDataItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function fetchMarketData() {
    setLoading(true);
    setError(null);
    try {
      const symbols = SYMBOLS.join(',');
      const res = await fetch(`/api/real-market-data?symbols=${symbols}`);
      if (!res.ok) throw new Error('Failed to fetch market data');
      const data = await res.json();
      if (!data.data) throw new Error('No data returned');
      // Transform API data to MarketDataItem[]
      const items: MarketDataItem[] = Object.entries(data.data).map(([symbol, d]: any) => ({
        symbol,
        name: d.name || symbol,
        price: d.price,
        change: d.change,
        changePercent: d.change_percent,
        volume: d.volume,
        marketCap: d.market_cap ? `$${d.market_cap}` : undefined,
        category: getCategory(symbol),
      }));
      setMarketData(items);
    } catch (err: any) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(fetchMarketData, 60000); // 60s
    return () => clearInterval(interval);
  }, []);

  function getCategory(symbol: string): MarketDataItem['category'] {
    if ([
      'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    ].includes(symbol)) return 'stocks';
    if ([
      'BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD',
    ].includes(symbol)) return 'crypto';
    if ([
      'USDT-USD', 'USDC-USD',
    ].includes(symbol)) return 'stablecoins';
    return 'commodities_etfs';
  }

  const categories = [
    { id: 'all', name: 'همه دارایی‌ها', count: marketData.length },
    { id: 'stocks', name: 'سهام', count: marketData.filter(item => item.category === 'stocks').length },
    { id: 'crypto', name: 'ارز دیجیتال', count: marketData.filter(item => item.category === 'crypto').length },
    { id: 'stablecoins', name: 'استیبل‌کوین', count: marketData.filter(item => item.category === 'stablecoins').length },
    { id: 'commodities_etfs', name: 'کالا و ETF', count: marketData.filter(item => item.category === 'commodities_etfs').length },
  ];

  const filteredData = selectedCategory === 'all' 
    ? marketData 
    : marketData.filter(item => item.category === selectedCategory);

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'stocks': return 'سهام';
      case 'crypto': return 'ارز دیجیتال';
      case 'stablecoins': return 'استیبل‌کوین';
      case 'commodities_etfs': return 'کالا و ETF';
      default: return category;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'stocks': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'crypto': return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
      case 'stablecoins': return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'commodities_etfs': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
      <div className="min-w-0 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">داده بازار</h1>
        <p className="text-muted-foreground">
          داده بازار زنده برای سهام، ارزهای دیجیتال، استیبل‌کوین‌ها و ETFها
        </p>
      </div>

      {/* Market Map */}
      <MarketMap
        data={marketData.map((d) => ({
          ...d,
          marketCap: d.marketCap ? parseFloat(d.marketCap.replace(/[$,BMK]/g, '') || '0') * (
            d.marketCap.includes('B') ? 1e9 : d.marketCap.includes('M') ? 1e6 : 1e3
          ) : d.volume,
        }))}
        loading={loading}
      />

      {/* Category Filter */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            دسته‌بندی دارایی‌ها
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => (
              <Button
                key={category.id}
                variant={selectedCategory === category.id ? "default" : "outline"}
                onClick={() => setSelectedCategory(category.id)}
                className="flex items-center gap-2"
              >
                {category.name}
                <Badge variant="secondary" className="ml-1">
                  {category.count}
                </Badge>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Market Overview Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">مجموع دارایی‌ها</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{filteredData.length}</div>
            <p className="text-xs text-muted-foreground">
              {selectedCategory === 'all' ? 'همه دسته‌ها' : `فقط ${categories.find(c => c.id === selectedCategory)?.name}`}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">میانگین تغییرات</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(filteredData.reduce((acc, item) => acc + item.changePercent, 0) / filteredData.length).toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              در دارایی‌های انتخاب‌شده
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">در حال رشد</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              {filteredData.filter(item => item.change > 0).length}
            </div>
            <p className="text-xs text-muted-foreground">
              دارایی‌های با تغییر مثبت
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">در حال افت</CardTitle>
            <TrendingDown className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">
              {filteredData.filter(item => item.change < 0).length}
            </div>
            <p className="text-xs text-muted-foreground">
              دارایی‌های با تغییر منفی
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Market Data Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            داده بازار زنده
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-8 text-center text-muted-foreground">در حال بارگذاری داده‌های بازار...</div>
          ) : error ? (
            <div className="py-8 text-center text-red-500">{error}</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3">دارایی</th>
                    <th className="text-right py-3">قیمت</th>
                    <th className="text-right py-3">تغییر</th>
                    <th className="text-right py-3">درصد تغییر</th>
                    <th className="text-right py-3">حجم</th>
                    <th className="text-right py-3">ارزش بازار</th>
                    <th className="text-left py-3">دسته</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((item, index) => (
                    <tr key={index} className="border-b hover:bg-muted/50">
                      <td className="py-3">
                        <div>
                          <div className="font-medium">{item.symbol}</div>
                          <div className="text-sm text-muted-foreground">{item.name}</div>
                        </div>
                      </td>
                      <td className="text-right py-3 font-mono">
                        ${item.price < 1 ? item.price.toFixed(4) : item.price.toFixed(2)}
                      </td>
                      <td className={`text-right py-3 font-mono ${item.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}
                      </td>
                      <td className={`text-right py-3 font-mono ${item.changePercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
                      </td>
                      <td className="text-right py-3 font-mono text-sm">
                        {item.volume.toLocaleString()}
                      </td>
                      <td className="text-right py-3 font-mono text-sm">
                        {item.marketCap}
                      </td>
                      <td className="py-3">
                        <Badge variant="outline" className={getCategoryColor(item.category)}>
                          {getCategoryLabel(item.category)}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
      </div>
      <aside className="hidden xl:block min-h-[360px]">
        <DataCollectorAgentPanel />
      </aside>
    </div>
  );
} 