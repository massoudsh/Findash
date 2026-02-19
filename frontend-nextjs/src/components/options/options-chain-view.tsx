'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { RefreshCw, Link2 } from 'lucide-react';
import type { OptionContractSelection } from './option-calculation-panel';

const ASSETS = [
  { symbol: 'BTC', name: 'Bitcoin' },
  { symbol: 'ETH', name: 'Ethereum' },
  { symbol: 'SPY', name: 'S&P 500 ETF' },
  { symbol: 'AAPL', name: 'Apple' },
  { symbol: 'NVDA', name: 'NVIDIA' },
  { symbol: 'TSLA', name: 'Tesla' },
];

interface OptionRow {
  strike: number;
  bid: number;
  ask: number;
  mark: number;
  bidIv: number;
  askIv: number;
  delta?: number;
  bidSize?: number;
  askSize?: number;
}

interface ChainData {
  spot: number;
  expiries: string[];
  expiry: string;
  calls: OptionRow[];
  puts: OptionRow[];
}

const symbolToMarketDataKey: Record<string, string> = {
  BTC: 'BTC-USD',
  ETH: 'ETH-USD',
  SPY: 'SPY',
  AAPL: 'AAPL',
  NVDA: 'NVDA',
  TSLA: 'TSLA',
};

function formatPrice(value: number, isCrypto: boolean): string {
  if (value >= 1000) return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (value >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

function formatPct(value: number): string {
  return `${Number(value).toFixed(1)}%`;
}

interface OptionsChainViewProps {
  onSelectContract?: (contract: OptionContractSelection) => void;
}

export function OptionsChainView({ onSelectContract }: OptionsChainViewProps = {}) {
  const [asset, setAsset] = useState('BTC');
  const [expiry, setExpiry] = useState<string>('');
  const [chain, setChain] = useState<ChainData | null>(null);
  const [spotFromMarket, setSpotFromMarket] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isCrypto = asset === 'BTC' || asset === 'ETH';
  const marketKey = symbolToMarketDataKey[asset] ?? asset;

  const fetchChain = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ symbol: asset });
      if (expiry) params.set('expiry', expiry);
      const res = await fetch(`/api/options-chain?${params}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load chain');
      setChain({
        spot: data.spot ?? 0,
        expiries: data.expiries ?? [],
        expiry: data.expiry ?? expiry,
        calls: data.calls ?? [],
        puts: data.puts ?? [],
      });
      if (data.expiries?.length && !expiry) setExpiry(data.expiry ?? data.expiries[0]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load options chain');
      setChain(null);
    } finally {
      setLoading(false);
    }
  }, [asset, expiry]);

  useEffect(() => {
    fetchChain();
  }, [fetchChain]);

  useEffect(() => {
    if (!marketKey) return;
    fetch(`/api/real-market-data?symbols=${marketKey}`)
      .then((r) => r.json())
      .then((json) => {
        const d = json?.data?.[marketKey];
        if (d?.price != null) setSpotFromMarket(Number(d.price));
      })
      .catch(() => {});
  }, [marketKey]);

  const spot = spotFromMarket ?? chain?.spot ?? 0;
  const expiries = chain?.expiries ?? [];
  const selectedExpiry = chain?.expiry ?? expiry;
  const calls = chain?.calls ?? [];
  const puts = chain?.puts ?? [];

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        {/* Header: Asset + Expiry tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 py-3 border-b bg-muted/30">
          <div className="flex items-center gap-3">
            <Select value={asset} onValueChange={(v) => { setAsset(v); setExpiry(''); setChain(null); }}>
              <SelectTrigger className="w-[120px] bg-background">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ASSETS.map((a) => (
                  <SelectItem key={a.symbol} value={a.symbol}>
                    {a.symbol}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="ghost" size="icon" onClick={() => fetchChain()} disabled={loading} aria-label="Refresh">
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
          <div className="flex flex-wrap gap-1">
            {expiries.slice(0, 10).map((e) => (
              <Button
                key={e}
                variant={selectedExpiry === e ? 'secondary' : 'ghost'}
                size="sm"
                className="text-xs"
                onClick={() => { setExpiry(e); }}
              >
                {e}
              </Button>
            ))}
          </div>
        </div>

        {error && (
          <div className="px-4 py-3 text-sm text-destructive bg-destructive/10">
            {error}
          </div>
        )}

        {/* Options chain table */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left py-2 px-2 font-medium text-muted-foreground">Bid Size</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Bid IV</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Bid</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Mark</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask IV</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask Size</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Delta</th>
                <th className="text-center py-2 px-3 font-semibold bg-primary/10 min-w-[90px]">{asset} ${formatPrice(spot, isCrypto)}</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Bid Size</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Bid IV</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Bid</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Mark</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask IV</th>
                <th className="text-right py-2 px-2 font-medium text-muted-foreground">Ask Size</th>
              </tr>
              <tr className="border-b text-muted-foreground text-xs">
                <th colSpan={8} className="text-left py-1 px-2 font-medium">Calls</th>
                <th className="py-1 px-2 font-medium">Strike</th>
                <th colSpan={8} className="text-left py-1 px-2 font-medium">Puts</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={17} className="py-8 text-center text-muted-foreground">
                    Loading options chain…
                  </td>
                </tr>
              )}
              {!loading && calls.length === 0 && !error && (
                <tr>
                  <td colSpan={17} className="py-8 text-center text-muted-foreground">
                    No options data for this expiry. Try another expiry or asset.
                  </td>
                </tr>
              )}
              {!loading && calls.length > 0 && calls.map((call, i) => {
                const put = puts[i];
                const strike = call?.strike ?? put?.strike ?? 0;
                const handleRowClick = () => {
                  if (!onSelectContract) return;
                  const spotVal = spotFromMarket ?? chain?.spot ?? 0;
                  onSelectContract({
                    underlying: asset,
                    strike,
                    type: 'call',
                    expiry: selectedExpiry,
                    premium: call?.mark ?? 0,
                    spot: spotVal,
                    callPremium: call?.mark,
                    putPremium: put?.mark,
                  });
                };
                return (
                  <tr
                    key={strike}
                    className={`border-b hover:bg-muted/30 ${onSelectContract ? 'cursor-pointer' : ''}`}
                    onClick={onSelectContract ? handleRowClick : undefined}
                  >
                    <td className="py-1.5 px-2 text-right tabular-nums">{call?.bidSize ?? '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-muted-foreground">{call?.bidIv ? formatPct(call.bidIv) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-red-600 dark:text-red-400">{call?.bid != null && call.bid > 0 ? formatPrice(call.bid, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{call?.mark != null && call.mark > 0 ? formatPrice(call.mark, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-green-600 dark:text-green-400">{call?.ask != null && call.ask > 0 ? formatPrice(call.ask, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-muted-foreground">{call?.askIv ? formatPct(call.askIv) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{call?.askSize ?? '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{call?.delta != null ? call.delta.toFixed(2) : '—'}</td>
                    <td className="py-1.5 px-3 text-center font-semibold bg-muted/50 tabular-nums">${formatPrice(strike, isCrypto)}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{put?.bidSize ?? '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-muted-foreground">{put?.bidIv ? formatPct(put.bidIv) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-red-600 dark:text-red-400">{put?.bid != null && put.bid > 0 ? formatPrice(put.bid, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{put?.mark != null && put.mark > 0 ? formatPrice(put.mark, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-green-600 dark:text-green-400">{put?.ask != null && put.ask > 0 ? formatPrice(put.ask, isCrypto) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-muted-foreground">{put?.askIv ? formatPct(put.askIv) : '—'}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums">{put?.askSize ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer: Balances, Positions, Orders, Greeks + Connect */}
        <div className="border-t px-4 py-3 bg-muted/20">
          <Tabs defaultValue="balances" className="w-full">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <TabsList className="h-9 bg-muted/50">
                <TabsTrigger value="balances" className="text-xs">Balances</TabsTrigger>
                <TabsTrigger value="positions" className="text-xs">Positions</TabsTrigger>
                <TabsTrigger value="orders" className="text-xs">Orders</TabsTrigger>
                <TabsTrigger value="greeks" className="text-xs">Greeks</TabsTrigger>
              </TabsList>
              <Button size="sm" variant="secondary" className="gap-1.5">
                <Link2 className="h-4 w-4" />
                Connect
              </Button>
            </div>
            <TabsContent value="balances" className="mt-3 min-h-[60px] text-sm text-muted-foreground">
              Connect a broker or data source to see balances.
            </TabsContent>
            <TabsContent value="positions" className="mt-3 min-h-[60px] text-sm text-muted-foreground">
              No open positions. Connect to view.
            </TabsContent>
            <TabsContent value="orders" className="mt-3 min-h-[60px] text-sm text-muted-foreground">
              No open orders. Connect to view.
            </TabsContent>
            <TabsContent value="greeks" className="mt-3 min-h-[60px] text-sm text-muted-foreground">
              Greeks are shown per strike in the chain. Connect for portfolio-level Greeks.
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>
    </Card>
  );
}
