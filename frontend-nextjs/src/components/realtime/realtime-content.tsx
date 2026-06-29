'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatCurrency } from '@/lib/utils';
import { Activity, Wifi, WifiOff, Play, Pause, Radio, TrendingUp, TrendingDown } from 'lucide-react';
import { useMarketWS } from '@/lib/hooks/use-market-ws';
import { cn } from '@/lib/utils';

const SYMBOLS = [
  'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
  'BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD',
  'USDT-USD', 'GLD', 'SLV', 'SPY', 'QQQ',
];

interface DataFeed {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  latency: number;
  throughput: number;
}

export function RealtimeContent() {
  const [isPaused, setIsPaused] = useState(false);
  const { ticks, status } = useMarketWS(isPaused ? [] : SYMBOLS);

  const dataFeeds: DataFeed[] = [
    { id: 'market', name: 'Market Data Feed', status: 'active', latency: 12, throughput: 1500 },
    { id: 'news', name: 'News Feed', status: 'active', latency: 45, throughput: 250 },
    { id: 'social', name: 'Social Sentiment', status: 'inactive', latency: 0, throughput: 0 },
    { id: 'options', name: 'Options Flow', status: 'active', latency: 8, throughput: 800 },
  ];

  const streamData = Object.values(ticks);
  const isLive = status === 'connected' || status === 'polling';

  return (
    <div className="space-y-6">
      {/* Stream Controls */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Radio className={cn('h-4 w-4', isLive && !isPaused && 'text-green-500 animate-pulse')} />
            Stream Control
            <Badge
              variant={status === 'connected' ? 'default' : status === 'polling' ? 'secondary' : 'destructive'}
              className="ml-2"
            >
              {status === 'connected' ? 'WebSocket' : status === 'polling' ? 'Polling 4s' : status}
            </Badge>
          </CardTitle>
          <Button onClick={() => setIsPaused((p) => !p)} variant={isPaused ? 'outline' : 'default'}>
            {isPaused ? <Play className="h-4 w-4 mr-2" /> : <Pause className="h-4 w-4 mr-2" />}
            {isPaused ? 'Resume' : 'Pause'}
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
                    <p className="font-medium text-sm">{feed.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {feed.status === 'active' ? `${feed.latency}ms latency` : 'Disconnected'}
                    </p>
                  </div>
                </div>
                <Badge variant={feed.status === 'active' ? 'default' : 'secondary'}>{feed.status}</Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Live Price Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Live Market Data
            <span className="text-xs text-muted-foreground font-normal ml-2">
              {streamData.length > 0 ? `${streamData.length} symbols` : 'Connecting...'}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {streamData.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground animate-pulse">
              {isPaused ? 'Stream paused' : 'Fetching live data...'}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th className="text-left py-2 px-2">Symbol</th>
                    <th className="text-right py-2 px-2">Price</th>
                    <th className="text-right py-2 px-2">Change</th>
                    <th className="text-right py-2 px-2">Change %</th>
                    <th className="text-right py-2 px-2">Volume</th>
                    <th className="text-right py-2 px-2">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {streamData.map((item) => {
                    const up = item.change >= 0;
                    const flashed = item.prev !== undefined && item.prev !== item.price;
                    return (
                      <tr
                        key={item.symbol}
                        className={cn(
                          'border-b transition-colors duration-500',
                          flashed && (up ? 'bg-green-500/10' : 'bg-red-500/10')
                        )}
                      >
                        <td className="py-2 px-2 font-semibold">{item.symbol}</td>
                        <td className="text-right py-2 px-2 font-mono">{formatCurrency(item.price)}</td>
                        <td className={cn('text-right py-2 px-2 font-mono', up ? 'text-green-600' : 'text-red-500')}>
                          <span className="inline-flex items-center gap-0.5">
                            {up ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                            {up ? '+' : ''}{item.change?.toFixed(2)}
                          </span>
                        </td>
                        <td className={cn('text-right py-2 px-2 font-mono', up ? 'text-green-600' : 'text-red-500')}>
                          {item.changePercent !== undefined
                            ? `${up ? '+' : ''}${item.changePercent.toFixed(2)}%`
                            : '—'}
                        </td>
                        <td className="text-right py-2 px-2 text-muted-foreground">{item.volume?.toLocaleString()}</td>
                        <td className="text-right py-2 px-2 text-xs text-muted-foreground">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Processing Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader><CardTitle className="text-sm">Messages/sec</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">2,547</div>
            <p className="text-xs text-muted-foreground">Real-time throughput</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-sm">Avg Latency</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {status === 'connected' ? '12ms' : status === 'polling' ? '4s' : '—'}
            </div>
            <p className="text-xs text-muted-foreground">End-to-end processing</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-sm">Active Streams</CardTitle></CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dataFeeds.filter((f) => f.status === 'active').length}</div>
            <p className="text-xs text-muted-foreground">of {dataFeeds.length} total feeds</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
