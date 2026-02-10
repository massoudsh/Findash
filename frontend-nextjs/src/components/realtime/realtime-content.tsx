'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatCurrency } from '@/lib/utils';
import { Activity, Wifi, WifiOff, Play, Pause } from 'lucide-react';

interface StreamData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  timestamp: string;
  status: 'connected' | 'disconnected';
}

interface DataFeed {
  id: string;
  name: string;
  type: 'market_data' | 'news' | 'social' | 'options';
  status: 'active' | 'inactive' | 'error';
  latency: number;
  throughput: number;
}

const SYMBOLS = [
  'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
  'BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD',
  'USDT-USD', 'USDC-USD',
  'GLD', 'SLV', 'SPY', 'QQQ',
];

export function RealtimeContent() {
  const [streamData, setStreamData] = useState<StreamData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataFeeds] = useState<DataFeed[]>([
    { id: 'market', name: 'Market Data Feed', type: 'market_data', status: 'active', latency: 12, throughput: 1500 },
    { id: 'news', name: 'News Feed', type: 'news', status: 'active', latency: 45, throughput: 250 },
    { id: 'social', name: 'Social Sentiment', type: 'social', status: 'inactive', latency: 0, throughput: 0 },
    { id: 'options', name: 'Options Flow', type: 'options', status: 'active', latency: 8, throughput: 800 },
  ]);
  const [isStreaming, setIsStreaming] = useState(true);

  async function fetchStreamData() {
    setLoading(true);
    setError(null);
    try {
      const symbols = SYMBOLS.join(',');
      const res = await fetch(`/api/real-market-data?symbols=${symbols}`);
      if (!res.ok) throw new Error('Failed to fetch real-time data');
      const data = await res.json();
      if (!data.data) throw new Error('No data returned');
      const items: StreamData[] = Object.entries(data.data).map(([symbol, d]: any) => ({
        symbol,
        price: d.price,
        change: d.change,
        volume: d.volume,
        timestamp: d.timestamp || new Date().toISOString(),
        status: 'connected',
      }));
      setStreamData(items);
    } catch (err: any) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!isStreaming) return;
    fetchStreamData();
    const interval = setInterval(fetchStreamData, 10000); // 10s
    return () => clearInterval(interval);
  }, [isStreaming]);

  return (
    <div className="space-y-6">
      {/* Stream Controls */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Stream Control</CardTitle>
          <Button
            onClick={() => setIsStreaming(!isStreaming)}
            variant={isStreaming ? "default" : "outline"}
          >
            {isStreaming ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            {isStreaming ? 'Pause' : 'Start'} Streaming
          </Button>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {dataFeeds.map((feed) => (
              <div key={feed.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-3">
                  {feed.status === 'active' ? (
                    <Wifi className="h-5 w-5 text-green-500" />
                  ) : (
                    <WifiOff className="h-5 w-5 text-red-500" />
                  )}
                  <div>
                    <p className="font-medium">{feed.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {feed.status === 'active' ? `${feed.latency}ms latency` : 'Disconnected'}
                    </p>
                  </div>
                </div>
                <Badge variant={feed.status === 'active' ? 'default' : 'secondary'}>
                  {feed.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Live Data Stream */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Activity className="h-5 w-5 mr-2" />
            Live Market Data
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-8 text-center text-muted-foreground">Loading real-time data...</div>
          ) : error ? (
            <div className="py-8 text-center text-red-500">{error}</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-right py-2">Price</th>
                    <th className="text-right py-2">Change</th>
                    <th className="text-right py-2">Volume</th>
                    <th className="text-right py-2">Last Update</th>
                    <th className="text-left py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {streamData.map((item, index) => (
                    <tr key={index} className="border-b">
                      <td className="py-2 font-medium">{item.symbol}</td>
                      <td className="text-right py-2">{formatCurrency(item.price)}</td>
                      <td className={`text-right py-2 ${item.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>{item.change >= 0 ? '+' : ''}{item.change?.toFixed(2)}</td>
                      <td className="text-right py-2">{item.volume?.toLocaleString()}</td>
                      <td className="text-right py-2 text-sm text-muted-foreground">{new Date(item.timestamp).toLocaleTimeString()}</td>
                      <td className="py-2">
                        <Badge variant={item.status === 'connected' ? 'default' : 'destructive'}>{item.status}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Processing Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Messages/sec</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2,547</div>
            <p className="text-sm text-muted-foreground">Real-time throughput</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Avg Latency</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12ms</div>
            <p className="text-sm text-muted-foreground">End-to-end processing</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Active Streams</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataFeeds.filter(f => f.status === 'active').length}</div>
            <p className="text-sm text-muted-foreground">of {dataFeeds.length} total feeds</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 