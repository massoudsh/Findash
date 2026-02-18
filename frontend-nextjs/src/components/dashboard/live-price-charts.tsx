'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

const SYMBOLS = ['BTC-USD', 'ETH-USD', 'QQQ'] as const;
const MAX_POINTS = 50;
const POLL_MS = 5000;

const LABELS: Record<string, string> = {
  'BTC-USD': 'Bitcoin (BTC)',
  'ETH-USD': 'Ethereum (ETH)',
  'QQQ': 'Nasdaq 100 (QQQ)',
};

const COLORS: Record<string, { stroke: string; fill: string }> = {
  'BTC-USD': { stroke: '#f7931a', fill: 'rgba(247,147,26,0.2)' },
  'ETH-USD': { stroke: '#627eea', fill: 'rgba(98,126,234,0.2)' },
  'QQQ': { stroke: '#22c55e', fill: 'rgba(34,197,94,0.2)' },
};

interface DataPoint {
  t: string;
  price: number;
  time: number;
}

export function LivePriceCharts() {
  const [series, setSeries] = useState<Record<string, DataPoint[]>>({
    'BTC-USD': [],
    'ETH-USD': [],
    'QQQ': [],
  });
  const [latest, setLatest] = useState<Record<string, { price: number; change?: number; changePercent?: number }>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPrices = useCallback(async () => {
    try {
      const res = await fetch(
        `/api/real-market-data?symbols=${SYMBOLS.join(',')}`
      );
      if (!res.ok) throw new Error('Failed to fetch');
      const json = await res.json();
      const data = json?.data;
      if (!data) throw new Error('No data');

      const now = Date.now();
      const timeLabel = new Date(now).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });

      setLatest((prev) => {
        const next: Record<string, { price: number; change?: number; changePercent?: number }> = {};
        SYMBOLS.forEach((sym) => {
          const d = data[sym];
          if (d && typeof d.price === 'number') {
            next[sym] = {
              price: d.price,
              change: d.change,
              changePercent: d.change_percent,
            };
          }
        });
        return { ...prev, ...next };
      });

      setSeries((prev) => {
        const next = { ...prev };
        SYMBOLS.forEach((sym) => {
          const d = data[sym];
          if (d && typeof d.price === 'number') {
            const list = [...(next[sym] || [])];
            list.push({ t: timeLabel, price: d.price, time: now });
            if (list.length > MAX_POINTS) list.shift();
            next[sym] = list;
          }
        });
        return next;
      });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error loading prices');
      if (Object.keys(series).every((k) => series[k].length === 0)) {
        const t = Date.now();
        setSeries({
          'BTC-USD': [{ t: '--', price: 97500, time: t }],
          'ETH-USD': [{ t: '--', price: 3650, time: t }],
          'QQQ': [{ t: '--', price: 485, time: t }],
        });
        setLatest({
          'BTC-USD': { price: 97500 },
          'ETH-USD': { price: 3650 },
          'QQQ': { price: 485 },
        });
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPrices();
    const interval = setInterval(fetchPrices, POLL_MS);
    return () => clearInterval(interval);
  }, [fetchPrices]);

  const formatPrice = (sym: string, value: number) => {
    if (sym === 'QQQ') return `$${value.toFixed(2)}`;
    if (sym === 'BTC-USD') return `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
    if (sym === 'ETH-USD') return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    return `$${Number(value).toFixed(2)}`;
  };

  if (loading && Object.values(series).every((arr) => arr.length === 0)) {
    return (
      <Card className="border-white/30 dark:border-white/20">
        <CardHeader>
          <CardTitle className="text-base">Live price charts</CardTitle>
          <p className="text-sm text-muted-foreground">BTC, ETH, Nasdaq 100</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {SYMBOLS.map((sym) => (
              <div key={sym} className="h-[180px] rounded-lg bg-muted/50 animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-white/30 dark:border-white/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center justify-between">
          <span>Live price charts</span>
          {error && (
            <span className="text-xs font-normal text-amber-600 dark:text-amber-400">
              Using cached / mock data
            </span>
          )}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          BTC, ETH, Nasdaq 100 (QQQ) — updates every {POLL_MS / 1000}s
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {SYMBOLS.map((sym) => {
            const data = series[sym] || [];
            const info = latest[sym];
            const { stroke, fill } = COLORS[sym] || { stroke: '#3b82f6', fill: 'rgba(59,130,246,0.2)' };
            const displayData = data.length ? data : [{ t: '--', price: info?.price ?? 0, time: 0 }];

            return (
              <Card key={sym} className="overflow-hidden border-border/50">
                <CardHeader className="py-2 px-4">
                  <CardTitle className="text-sm font-medium">
                    {LABELS[sym]}
                  </CardTitle>
                  <div className="flex items-baseline gap-2">
                    <span className="text-lg font-bold tabular-nums">
                      {info ? formatPrice(sym, info.price) : '—'}
                    </span>
                    {info?.changePercent != null && (
                      <span
                        className={`text-xs font-medium ${
                          info.changePercent >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {info.changePercent >= 0 ? '+' : ''}
                        {info.changePercent.toFixed(2)}%
                      </span>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-0 px-4 pb-4">
                  <div className="h-[140px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={displayData}
                        margin={{ top: 4, right: 4, left: 4, bottom: 4 }}
                      >
                        <defs>
                          <linearGradient id={`live-${sym}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={stroke} stopOpacity={0.4} />
                            <stop offset="100%" stopColor={stroke} stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <XAxis
                          dataKey="t"
                          tick={{ fontSize: 10 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis
                          hide
                          domain={['auto', 'auto']}
                          tickFormatter={(v) => formatPrice(sym, v)}
                        />
                        <Tooltip
                          contentStyle={{ fontSize: 11, borderRadius: 8 }}
                          formatter={(value: number) => [formatPrice(sym, value), 'Price']}
                          labelFormatter={(t) => t}
                        />
                        <Area
                          type="monotone"
                          dataKey="price"
                          stroke={stroke}
                          fill={`url(#live-${sym})`}
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
