'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  Zap,
  BarChart3,
  ChevronRight,
  Gauge,
  Calendar,
  Layers,
} from 'lucide-react';

const UNDERLYINGS = [
  { symbol: 'SPY', name: 'S&P 500 ETF' },
  { symbol: 'QQQ', name: 'Nasdaq 100 ETF' },
  { symbol: 'AAPL', name: 'Apple' },
  { symbol: 'NVDA', name: 'NVIDIA' },
  { symbol: 'TSLA', name: 'Tesla' },
  { symbol: 'ETH', name: 'Ethereum' },
  { symbol: 'BTC', name: 'Bitcoin' },
];

interface MarketSnapshot {
  spot: number;
  change: number;
  changePct: number;
  iv: number;       // ATM IV %
  ivRank: number;   // 0-100
  ivPercentile: number;
}

const MOCK_SNAPSHOTS: Record<string, MarketSnapshot> = {
  SPY:  { spot: 582.34,  change: 2.14,  changePct: 0.37,  iv: 12.4,  ivRank: 28,  ivPercentile: 35 },
  QQQ:  { spot: 521.88,  change: 3.22,  changePct: 0.62,  iv: 14.1,  ivRank: 32,  ivPercentile: 41 },
  AAPL: { spot: 228.12,  change: -0.84, changePct: -0.37, iv: 18.2,  ivRank: 45,  ivPercentile: 52 },
  NVDA: { spot: 138.56,  change: 4.12,  changePct: 3.06,  iv: 42.5,  ivRank: 78,  ivPercentile: 85 },
  TSLA: { spot: 248.90,  change: -2.10, changePct: -0.84, iv: 55.3,  ivRank: 82,  ivPercentile: 88 },
  ETH:  { spot: 3842.50, change: 52.30, changePct: 1.38,  iv: 48.0,  ivRank: 65,  ivPercentile: 70 },
  BTC:  { spot: 97250.00, change: 420.00, changePct: 0.43, iv: 52.0, ivRank: 58,  ivPercentile: 62 },
};

interface GreeksAtm {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
}

const MOCK_GREEKS: Record<string, GreeksAtm> = {
  SPY:  { delta: 0.50, gamma: 0.042, theta: -0.08, vega: 0.12 },
  QQQ:  { delta: 0.50, gamma: 0.038, theta: -0.09, vega: 0.14 },
  AAPL: { delta: 0.50, gamma: 0.055, theta: -0.15, vega: 0.22 },
  NVDA: { delta: 0.50, gamma: 0.082, theta: -0.35, vega: 0.48 },
  TSLA: { delta: 0.50, gamma: 0.065, theta: -0.42, vega: 0.55 },
  ETH:  { delta: 0.50, gamma: 0.00012, theta: -0.02, vega: 0.08 },
  BTC:  { delta: 0.50, gamma: 0.00001, theta: -0.01, vega: 0.15 },
};

const EXPIRIES = [
  { label: 'Weekly', date: '21 Feb 2026', dte: 4 },
  { label: 'Monthly', date: '21 Mar 2026', dte: 32 },
  { label: 'Quarterly', date: '20 Jun 2026', dte: 123 },
  { label: 'LEAPS', date: '20 Jan 2027', dte: 337 },
];

const QUICK_STRATEGIES = [
  { id: 'covered-call', name: 'Covered Call', desc: 'Sell OTM call, collect premium', risk: 'Neutral to bullish', icon: Layers },
  { id: 'put-spread', name: 'Bull Put Spread', desc: 'Sell put, buy lower put', risk: 'Bullish', icon: TrendingUp },
  { id: 'straddle', name: 'Straddle', desc: 'Long call + long put, same strike', risk: 'Volatility', icon: Activity },
  { id: 'iron-condor', name: 'Iron Condor', desc: 'Sell strangle, buy wings', risk: 'Range / theta', icon: Target },
];

export interface OptionsMarketToolsProps {
  onSelectStrategy?: (strategyId: string) => void;
  onOpenTrade?: () => void;
}

export function OptionsMarketTools({ onSelectStrategy, onOpenTrade }: OptionsMarketToolsProps) {
  const [underlying, setUnderlying] = useState('SPY');
  const [expiry, setExpiry] = useState(EXPIRIES[0].label);
  const snapshot = MOCK_SNAPSHOTS[underlying] ?? MOCK_SNAPSHOTS.SPY;
  const greeks = MOCK_GREEKS[underlying] ?? MOCK_GREEKS.SPY;

  return (
    <div className="space-y-4">
      {/* Header: Options Market — decision tools */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Gauge className="h-5 w-5 text-emerald-500" />
            Options Market — Decision Tools
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            Underlying, IV, Greeks, and expirations to drive your options decisions
          </p>
        </div>
        <Select value={underlying} onValueChange={setUnderlying}>
          <SelectTrigger className="w-[180px] bg-background">
            <SelectValue placeholder="Underlying" />
          </SelectTrigger>
          <SelectContent>
            {UNDERLYINGS.map((u) => (
              <SelectItem key={u.symbol} value={u.symbol}>
                {u.symbol} — {u.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Row 1: Spot + IV + IV Rank + Expiry strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="bg-card/80">
          <CardContent className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Spot</p>
            <p className="text-xl font-bold mt-0.5">
              {underlying.startsWith('B') || underlying.startsWith('E') ? `$${snapshot.spot.toLocaleString('en-US', { minimumFractionDigits: 2 })}` : `$${snapshot.spot.toFixed(2)}`}
            </p>
            <p className={snapshot.changePct >= 0 ? 'text-green-600 text-sm flex items-center gap-0.5' : 'text-red-600 text-sm flex items-center gap-0.5'}>
              {snapshot.changePct >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
              {snapshot.changePct >= 0 ? '+' : ''}{snapshot.changePct.toFixed(2)}% ({snapshot.change >= 0 ? '+' : ''}{snapshot.change.toFixed(2)})
            </p>
          </CardContent>
        </Card>
        <Card className="bg-card/80">
          <CardContent className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">ATM IV</p>
            <p className="text-xl font-bold mt-0.5 text-emerald-600">{snapshot.iv.toFixed(1)}%</p>
            <p className="text-xs text-muted-foreground">Implied volatility (ATM)</p>
          </CardContent>
        </Card>
        <Card className="bg-card/80">
          <CardContent className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">IV Rank / %ile</p>
            <p className="text-xl font-bold mt-0.5">{snapshot.ivRank}% / {snapshot.ivPercentile}%</p>
            <p className="text-xs text-muted-foreground">1Y rank · percentile</p>
          </CardContent>
        </Card>
        <Card className="bg-card/80">
          <CardContent className="p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1">
              <Calendar className="h-3 w-3" /> Expiry
            </p>
            <Select value={expiry} onValueChange={setExpiry}>
              <SelectTrigger className="mt-1.5 h-9 border-0 bg-muted/50 p-2">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EXPIRIES.map((e) => (
                  <SelectItem key={e.label} value={e.label}>
                    {e.label} · {e.date}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
      </div>

      {/* Row 2: Greeks at a glance + Quick strategies */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="bg-card/80">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Greeks (ATM)
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-4 gap-3">
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Delta</p>
                <p className="font-mono font-semibold">{greeks.delta.toFixed(2)}</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Gamma</p>
                <p className="font-mono font-semibold">{greeks.gamma < 0.01 ? greeks.gamma.toFixed(5) : greeks.gamma.toFixed(3)}</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Theta</p>
                <p className="font-mono font-semibold text-red-600">{greeks.theta.toFixed(2)}</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Vega</p>
                <p className="font-mono font-semibold">{greeks.vega.toFixed(2)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-card/80">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Quick strategies
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 gap-2">
              {QUICK_STRATEGIES.map((s) => {
                const Icon = s.icon;
                return (
                  <Button
                    key={s.id}
                    variant="outline"
                    className="h-auto py-3 px-3 justify-start gap-2 text-left"
                    onClick={() => {
                      onSelectStrategy?.(s.id);
                      onOpenTrade?.();
                    }}
                  >
                    <Icon className="h-4 w-4 shrink-0 text-emerald-500" />
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-sm truncate">{s.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{s.risk}</p>
                    </div>
                    <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                  </Button>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
