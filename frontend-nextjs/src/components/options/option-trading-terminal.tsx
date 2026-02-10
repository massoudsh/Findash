'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Star,
  ChevronDown,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Settings,
  RefreshCw,
  ExternalLink,
  Gift,
  Info,
} from 'lucide-react';

// Mock order book
const SHORT_RATE_ORDERS = [
  { rate: 6.2, size: 31.5673 },
  { rate: 6.1, size: 31.751 },
  { rate: 6.0, size: 758.225 },
  { rate: 5.9, size: 31.8935 },
  { rate: 5.8, size: 32.4144 },
  { rate: 5.7, size: 32.2105 },
  { rate: 5.6, size: 337.52 },
  { rate: 5.5, size: 132.649 },
  { rate: 5.4, size: 1042.4 },
];

const LONG_RATE_ORDERS = [
  { rate: 5.3, size: 12.695 },
  { rate: 5.2, size: 51.6531 },
  { rate: 5.1, size: 33.2776 },
  { rate: 5.0, size: 35.9013 },
  { rate: 4.9, size: 18.0023 },
  { rate: 4.8, size: 17.8371 },
  { rate: 4.7, size: 337.968 },
];

interface FundingMetrics {
  impliedApr: number;
  markApr: number;
  underlyingApr: number;
  notionalOi: string;
  volume24h: string;
  nextSettlement: string; // "HH:MM:SS"
}

const DEFAULT_METRICS: FundingMetrics = {
  impliedApr: 5.34,
  markApr: 5.35,
  underlyingApr: -6.82,
  notionalOi: '5.1815K ETH',
  volume24h: '6.389K ETH',
  nextSettlement: '00:46:42',
};

function formatCountdown(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h, m, s].map((v) => String(v).padStart(2, '0')).join(':');
}

/** Backend symbol (e.g. ETHUSDT). Display symbol can be ETHUSDC. */
function toBackendSymbol(displaySymbol: string): string {
  return displaySymbol.replace('USDC', 'USDT');
}

interface ChartPoint {
  timestamp: string;
  funding_rate_annualized_pct: number;
}

export function OptionTradingTerminal() {
  const [symbol, setSymbol] = useState('ETHUSDC');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [tradeDirection, setTradeDirection] = useState<'long' | 'short'>('long');
  const [leverage, setLeverage] = useState('One-way');
  const [notionalSize, setNotionalSize] = useState([0]);
  const [reduceOnly, setReduceOnly] = useState(false);
  const [chartTab, setChartTab] = useState<'apr' | 'pnl'>('apr');
  const [orderbookTab, setOrderbookTab] = useState<'orderbook' | 'trades'>('orderbook');
  const [metrics, setMetrics] = useState<FundingMetrics>(DEFAULT_METRICS);
  const [countdownSeconds, setCountdownSeconds] = useState(46 * 60 + 42);
  const [loadingFunding, setLoadingFunding] = useState(false);
  const [fundingError, setFundingError] = useState<string | null>(null);
  const [supportedSymbols, setSupportedSymbols] = useState<string[]>(['ETHUSDT', 'BTCUSDT']);
  const [chartSeries, setChartSeries] = useState<ChartPoint[]>([]);
  const [chartLoading, setChartLoading] = useState(false);
  const [symbolPickerOpen, setSymbolPickerOpen] = useState(false);

  const timeframes = ['5m', '1H', '1D', '1W'];
  const backendSymbol = toBackendSymbol(symbol);

  // Supported symbols (display as USDC for UX)
  useEffect(() => {
    fetch('/api/funding/supported-symbols', { cache: 'no-store' })
      .then((res) => res.json())
      .then((data: { supported_symbols?: string[] }) => {
        const list = data?.supported_symbols ?? [];
        setSupportedSymbols(list.length ? list : ['ETHUSDT', 'BTCUSDT']);
      })
      .catch(() => {});
  }, []);

  // Fetch funding rate + set countdown
  const fetchFunding = useCallback(async () => {
    setLoadingFunding(true);
    setFundingError(null);
    try {
      const res = await fetch(`/api/funding/refresh/${encodeURIComponent(backendSymbol)}`, {
        cache: 'no-store',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setFundingError(data?.detail || data?.error || res.statusText);
        setMetrics(DEFAULT_METRICS);
        return;
      }
      const annualized = (data.funding_rate_annualized ?? (data.funding_rate ?? 0) * 365 * 3) * 100;
      const nextMinutes = data.time_to_next_funding_minutes;
      if (typeof nextMinutes === 'number') {
        setCountdownSeconds(Math.max(0, Math.round(nextMinutes * 60)));
      }
      setMetrics((m) => ({
        ...m,
        impliedApr: annualized,
        markApr: annualized,
      }));
    } catch (e) {
      setFundingError(e instanceof Error ? e.message : 'Network error');
      setMetrics(DEFAULT_METRICS);
    } finally {
      setLoadingFunding(false);
    }
  }, [backendSymbol]);

  useEffect(() => {
    fetchFunding();
  }, [fetchFunding]);

  // Chart: fetch history
  useEffect(() => {
    if (chartTab !== 'apr') return;
    setChartLoading(true);
    fetch(`/api/funding/history/${encodeURIComponent(backendSymbol)}?limit=100`, { cache: 'no-store' })
      .then((res) => res.json())
      .then((data: { series?: ChartPoint[] }) => {
        setChartSeries(Array.isArray(data?.series) ? data.series : []);
      })
      .catch(() => setChartSeries([]))
      .finally(() => setChartLoading(false));
  }, [backendSymbol, chartTab]);

  // Countdown ticker
  useEffect(() => {
    const t = setInterval(() => {
      setCountdownSeconds((s) => Math.max(0, s - 1));
    }, 1000);
    return () => clearInterval(t);
  }, []);

  // Close symbol picker on outside click
  useEffect(() => {
    if (!symbolPickerOpen) return;
    const onClose = () => setSymbolPickerOpen(false);
    document.addEventListener('click', onClose);
    return () => document.removeEventListener('click', onClose);
  }, [symbolPickerOpen]);

  // Build chart path from series (annualized %) — viewBox 0 0 500 280
  const chartPath = (() => {
    if (!chartSeries.length) return null;
    const pts = chartSeries.map((p) => p.funding_rate_annualized_pct);
    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const range = max - min || 1;
    const h = 280;
    const w = 500;
    const x = (i: number) => (pts.length > 1 ? (i / (pts.length - 1)) * w : 0);
    const y = (v: number) => h - ((v - min) / range) * (h - 20);
    const d = pts.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ');
    return d;
  })();

  return (
    <div className="flex flex-col h-full min-h-[calc(100vh-8rem)] bg-background text-foreground">
      {/* Asset bar */}
      <div className="flex items-center justify-between py-3 px-4 border-b">
        <div className="flex items-center gap-3">
          <Star className="h-4 w-4 text-muted-foreground hover:text-yellow-500 cursor-pointer" />
          <div className="relative">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setSymbolPickerOpen((o) => !o);
              }}
              className="flex items-center gap-1.5 focus:outline-none"
            >
              <div className="w-6 h-6 rounded-full bg-gradient-to-r from-blue-500 to-purple-500" />
              <div className="w-6 h-6 rounded-full bg-gradient-to-r from-green-400 to-green-600 -ml-2" />
              <span className="text-xl font-bold">{symbol}</span>
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            </button>
            {symbolPickerOpen && (
              <div
                className="absolute top-full left-0 mt-1 z-50 bg-popover border rounded-lg shadow-lg max-h-64 overflow-auto min-w-[140px]"
                onClick={(e) => e.stopPropagation()}
              >
                {supportedSymbols.map((s) => {
                  const display = s.replace('USDT', 'USDC');
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => {
                        setSymbol(display);
                        setSymbolPickerOpen(false);
                      }}
                      className="w-full px-3 py-2 text-left hover:bg-muted text-sm font-medium"
                    >
                      {display}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          <Info className="h-4 w-4 text-muted-foreground" />
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => fetchFunding()}
            disabled={loadingFunding}
          >
            <RefreshCw className={`h-4 w-4 ${loadingFunding ? 'animate-spin' : ''}`} />
          </Button>
          {fundingError && (
            <span className="text-xs text-destructive max-w-[180px] truncate" title={fundingError}>
              {fundingError}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold">25 days</span>
          <span className="text-sm text-muted-foreground">(Matures 27 Feb 2026)</span>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>

      {/* Metrics row */}
      <div className="flex items-center gap-8 py-4 px-4 border-b flex-wrap">
        <div>
          <div className="text-3xl font-bold text-green-500">
            {loadingFunding ? '…' : `${metrics.impliedApr.toFixed(2)}%`}
          </div>
          <div className="text-xs text-muted-foreground">Implied APR</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Mark APR</div>
          <div className="text-sm font-semibold">{metrics.markApr.toFixed(2)}%</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Underlying APR</div>
          <div className="text-sm font-semibold text-red-500">
            {metrics.underlyingApr.toFixed(2)}% <ExternalLink className="h-3 w-3 inline" />
          </div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Notional OI</div>
          <div className="text-sm font-semibold">{metrics.notionalOi}</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">24h Volume</div>
          <div className="text-sm font-semibold">{metrics.volume24h}</div>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Next Settlement</div>
          <div className="text-sm font-semibold tabular-nums">
            {formatCountdown(countdownSeconds)}
          </div>
        </div>
      </div>

      {/* Main: Chart | Order book | Order entry */}
      <div className="grid grid-cols-12 gap-4 flex-1 p-4 min-h-0">
        {/* Left — Chart */}
        <div className="col-span-6 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0 border rounded-lg">
            <CardContent className="p-4 flex flex-col flex-1 min-h-0">
              <div className="flex items-center gap-4 mb-3">
                <Button
                  variant={chartTab === 'apr' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setChartTab('apr')}
                >
                  APR Chart
                </Button>
                <Button
                  variant={chartTab === 'pnl' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setChartTab('pnl')}
                >
                  My PnL
                </Button>
              </div>
              {chartTab === 'apr' && (
                <>
                  <div className="flex items-center gap-4 mb-1 text-xs">
                    <span>Implied APR</span>
                    <span className="text-green-500">O{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">H{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">L{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">C{metrics.impliedApr.toFixed(2)}%</span>
                  </div>
                  <div className="text-xs text-red-500 mb-3">
                    Underlying APR <span className="ml-2">{metrics.underlyingApr.toFixed(2)}%</span>
                  </div>
                </>
              )}
              <div className="relative flex-1 min-h-[240px] bg-muted/30 rounded-lg overflow-hidden">
                {chartLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-muted/20 z-10">
                    <span className="text-sm text-muted-foreground">Loading chart…</span>
                  </div>
                )}
                <div className="absolute right-2 top-0 bottom-0 flex flex-col justify-between text-xs text-muted-foreground py-4">
                  <span>4%</span>
                  <span>0%</span>
                  <span>-4%</span>
                  <span>-8%</span>
                  <span>-12%</span>
                  <span>-20%</span>
                  <span>-28%</span>
                </div>
                <svg className="w-full h-full" viewBox="0 0 500 280" preserveAspectRatio="none">
                  {[0, 35, 70, 105, 140, 175, 210, 245, 280].map((y, i) => (
                    <line
                      key={i}
                      x1="0"
                      y1={y}
                      x2="500"
                      y2={y}
                      stroke="currentColor"
                      strokeOpacity="0.12"
                      strokeWidth="0.5"
                    />
                  ))}
                  {chartPath ? (
                    <path
                      d={chartPath}
                      fill="none"
                      stroke="rgb(59, 130, 246)"
                      strokeWidth="2"
                    />
                  ) : (
                    <path
                      d="M 0,220 L 50,230 L 100,200 L 150,180 L 200,210 L 250,190 L 300,140 L 350,120 L 400,95 L 450,70 L 480,90 L 500,130"
                      fill="none"
                      stroke="rgb(59, 130, 246)"
                      strokeWidth="2"
                    />
                  )}
                  <rect x="440" y="65" width="52" height="20" fill="rgb(34, 197, 94)" rx="3" />
                  <text x="466" y="79" fill="white" fontSize="10" textAnchor="middle">
                    {metrics.impliedApr.toFixed(2)}%
                  </text>
                  <rect x="440" y="125" width="52" height="20" fill="rgb(239, 68, 68)" rx="3" />
                  <text x="466" y="139" fill="white" fontSize="10" textAnchor="middle">
                    {metrics.underlyingApr.toFixed(2)}%
                  </text>
                </svg>
                <div className="absolute top-2 right-2 flex items-center gap-2">
                  {timeframes.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setSelectedTimeframe(tf)}
                      className={`px-2 py-1 text-xs rounded ${
                        selectedTimeframe === tf
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                  <button className="p-1 text-muted-foreground hover:text-foreground">
                    <BarChart3 className="h-4 w-4" />
                  </button>
                  <button className="p-1 text-muted-foreground hover:text-foreground">
                    <Settings className="h-4 w-4" />
                  </button>
                  <button className="p-1 text-muted-foreground hover:text-foreground">
                    <RefreshCw className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Center — Order book */}
        <div className="col-span-3 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0 border rounded-lg">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-4">
                <button
                  className={`text-sm pb-1 border-b-2 ${orderbookTab === 'orderbook' ? 'border-primary font-medium' : 'border-transparent text-muted-foreground'}`}
                  onClick={() => setOrderbookTab('orderbook')}
                >
                  Orderbook
                </button>
                <button
                  className={`text-sm pb-1 border-b-2 ${orderbookTab === 'trades' ? 'border-primary font-medium' : 'border-transparent text-muted-foreground'}`}
                  onClick={() => setOrderbookTab('trades')}
                >
                  Market Trades
                </button>
              </div>
            </CardHeader>
            <CardContent className="p-4 flex-1 overflow-auto">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-muted-foreground">0.1%</span>
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
                <div className="flex items-center gap-1">
                  <div className="w-4 h-4 rounded bg-muted" />
                  <div className="w-4 h-4 rounded bg-muted" />
                  <div className="w-4 h-4 rounded bg-muted" />
                </div>
                <div className="flex items-center gap-1">
                  <Checkbox id="inc-range" className="h-3 w-3" />
                  <label htmlFor="inc-range" className="text-xs text-muted-foreground">
                    Incentivized Range
                  </label>
                </div>
              </div>
              <div className="mb-4">
                <div className="text-xs text-red-500 font-semibold mb-2">SHORT RATE</div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Implied APR (%)</span>
                  <span>Size (ETH YU)</span>
                </div>
                <div className="space-y-1">
                  {SHORT_RATE_ORDERS.map((order, i) => (
                    <div key={i} className="flex justify-between text-xs relative py-0.5">
                      <div
                        className="absolute left-0 top-0 bottom-0 bg-red-500/20 rounded"
                        style={{ width: `${Math.min(order.size / 12, 100)}%` }}
                      />
                      <span className="text-red-500 relative z-10">{order.rate.toFixed(1)}</span>
                      <span className="relative z-10">{order.size.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex justify-between text-xs py-2 border-y my-2">
                <span className="text-muted-foreground">0.1% Spread</span>
                <span className="text-cyan-500">Incent. Range: 5.16% - 5.62%</span>
              </div>
              <div>
                <div className="text-xs text-green-500 font-semibold mb-2">LONG RATE</div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Implied APR (%)</span>
                  <span>Size (ETH YU)</span>
                </div>
                <div className="space-y-1">
                  {LONG_RATE_ORDERS.map((order, i) => (
                    <div key={i} className="flex justify-between text-xs relative py-0.5">
                      <div
                        className="absolute left-0 top-0 bottom-0 bg-green-500/20 rounded"
                        style={{ width: `${Math.min(order.size / 12, 100)}%` }}
                      />
                      <span className="text-green-500 relative z-10">{order.rate.toFixed(1)}</span>
                      <span className="relative z-10">{order.size.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right — Order entry */}
        <div className="col-span-3 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0 border rounded-lg">
            <CardContent className="p-4 flex flex-col">
              <div className="bg-amber-500/20 border border-amber-500/30 rounded-lg p-3 mb-4 flex items-center gap-2">
                <Gift className="h-4 w-4 text-amber-400" />
                <span className="text-sm text-amber-200">Maker Order Rewards Live!</span>
              </div>
              <div className="grid grid-cols-3 gap-2 mb-4">
                {['Cross', '2x', 'One-way'].map((lev) => (
                  <Button
                    key={lev}
                    variant={leverage === lev ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setLeverage(lev)}
                  >
                    {lev}
                  </Button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2 mb-4">
                <Button
                  variant={orderType === 'market' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setOrderType('market')}
                >
                  Market
                </Button>
                <Button
                  variant={orderType === 'limit' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setOrderType('limit')}
                >
                  Limit
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-2 mb-4">
                <Button
                  onClick={() => setTradeDirection('long')}
                  className={`flex items-center gap-2 ${tradeDirection === 'long' ? 'bg-green-600 hover:bg-green-700' : ''}`}
                  variant={tradeDirection === 'long' ? 'default' : 'outline'}
                  size="sm"
                >
                  <TrendingUp className="h-4 w-4 shrink-0" />
                  <div className="text-left min-w-0">
                    <div className="text-xs font-semibold">Long Rates</div>
                    <div className="text-[10px] opacity-70 truncate">Pay Fixed, Rcv. Underlying</div>
                  </div>
                </Button>
                <Button
                  onClick={() => setTradeDirection('short')}
                  className={`flex items-center gap-2 ${tradeDirection === 'short' ? 'bg-red-600 hover:bg-red-700' : ''}`}
                  variant={tradeDirection === 'short' ? 'default' : 'outline'}
                  size="sm"
                >
                  <TrendingDown className="h-4 w-4 shrink-0" />
                  <div className="text-left min-w-0">
                    <div className="text-xs font-semibold">Short Rates</div>
                    <div className="text-[10px] opacity-70 truncate">Pay Underlying, Rcv. Fixed</div>
                  </div>
                </Button>
              </div>
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">My Notional Size</span>
                  <span className="text-cyan-500">0 YU</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Available to Trade</span>
                  <span>0 ETH</span>
                </div>
              </div>
              <div className="mb-4">
                <Label className="text-sm text-muted-foreground">Notional Size</Label>
                <div className="flex items-center gap-2 mt-1">
                  <Input
                    type="number"
                    value={notionalSize[0]}
                    onChange={(e) => setNotionalSize([parseInt(e.target.value, 10) || 0])}
                    className="bg-muted/50"
                    placeholder="YU"
                  />
                  <span className="text-xs text-muted-foreground">0 ETH</span>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <Slider
                    value={notionalSize}
                    onValueChange={setNotionalSize}
                    max={100}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-xs w-8">{notionalSize[0]} %</span>
                </div>
              </div>
              <div className="flex items-center gap-2 mb-4">
                <Checkbox
                  id="reduce-only"
                  checked={reduceOnly}
                  onCheckedChange={(c) => setReduceOnly(c === true)}
                />
                <Label htmlFor="reduce-only" className="text-sm text-muted-foreground">
                  Reduce Only
                </Label>
              </div>
              <div className="space-y-2 text-sm border-t pt-4 mt-auto">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Liquidation Implied APR</span>
                  <span>NA</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Margin Required</span>
                  <span>NA</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Fees</span>
                  <span>0 ETH ($0.00)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Slippage</span>
                  <span>Est: 0% / Max: <span className="text-cyan-500">0.5%</span></span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 border-t text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span>Online</span>
          </div>
          <span>Gas: $0.02</span>
        </div>
        <div className="flex items-center gap-4">
          <a href="#" className="hover:text-foreground">Docs</a>
          <a href="#" className="hover:text-foreground">Support</a>
          <a href="#" className="hover:text-foreground">Terms</a>
          <a href="#" className="hover:text-foreground">Policy</a>
          <a href="#" className="hover:text-cyan-500">Help & Support</a>
        </div>
      </div>
    </div>
  );
}
