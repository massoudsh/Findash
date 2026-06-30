'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from '@/components/ui/dropdown-menu';
import { toast } from '@/components/ui/toast';
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
  Target,
  Calculator,
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

/** Open funding-rate position (from place order). */
export interface FundingPosition {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  notionalEth: number;
  entryApr: number;
  currentApr: number;
  marginMode: string;
  openTime: string;
  reduceOnly: boolean;
}

export interface OptionTradingTerminalStrategy {
  name: string;
  description?: string;
  category?: string;
}

interface OptionTradingTerminalProps {
  selectedStrategy?: OptionTradingTerminalStrategy | null;
  strategyPnl?: number;
  onClearStrategy?: () => void;
}

export function OptionTradingTerminal({ selectedStrategy, strategyPnl = 0, onClearStrategy }: OptionTradingTerminalProps = {}) {
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
  const [isFavorite, setIsFavorite] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const [selectedExpiry, setSelectedExpiry] = useState({ label: '25 days', matures: '27 Feb 2026' });
  const [chartType, setChartType] = useState<'line' | 'area'>('line');
  const [chartSettingsOpen, setChartSettingsOpen] = useState(false);
  const [orderbookSpread, setOrderbookSpread] = useState('0.1%');
  const [orderbookDepth, setOrderbookDepth] = useState(10);
  const [placingOrder, setPlacingOrder] = useState(false);
  const [positions, setPositions] = useState<FundingPosition[]>([
    {
      id: 'pos-1',
      symbol: 'ETHUSDC',
      direction: 'long',
      notionalEth: 2.5,
      entryApr: 5.12,
      currentApr: 5.34,
      marginMode: 'Cross',
      openTime: new Date(Date.now() - 86400 * 2 * 1000).toISOString(),
      reduceOnly: false,
    },
    {
      id: 'pos-2',
      symbol: 'BTCUSDC',
      direction: 'short',
      notionalEth: 0.15,
      entryApr: 4.85,
      currentApr: 5.1,
      marginMode: 'One-way',
      openTime: new Date(Date.now() - 3600 * 12 * 1000).toISOString(),
      reduceOnly: false,
    },
  ]);
  const [selectedPosition, setSelectedPosition] = useState<FundingPosition | null>(null);
  const [orderbookCompact, setOrderbookCompact] = useState(false);
  const [makerRewardsBannerDismissed, setMakerRewardsBannerDismissed] = useState(false);

  const timeframes = ['5m', '1H', '4H', '1D', '1W'];
  const expiryOptions = [
    { label: '7 days', matures: '6 Feb 2026' },
    { label: '14 days', matures: '13 Feb 2026' },
    { label: '25 days', matures: '27 Feb 2026' },
    { label: '30 days', matures: '2 Mar 2026' },
  ];
  const spreadOptions = ['0.05%', '0.1%', '0.25%', '0.5%'];
  const depthOptions = [10, 25, 50];
  const backendSymbol = toBackendSymbol(symbol);

  // Mock available balance (ETH)
  const MOCK_AVAILABLE_ETH = 10;
  const notionalEth = notionalSize[0];
  const marginRequiredEth =
    leverage === 'Cross'
      ? notionalEth // Cross: position size = margin used (fully backed by shared balance)
      : leverage === '2x'
        ? notionalEth / 2 // 2x: 50% margin
        : notionalEth; // One-way: 1x, full notional as margin
  const canOpen = notionalEth > 0 && marginRequiredEth <= MOCK_AVAILABLE_ETH;
  const estFundingPer8h =
    notionalEth > 0 ? (notionalEth * (metrics.impliedApr / 100) * (8 / (365 * 24))) : 0;

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

  // Chart: fetch history — limit by timeframe for functional chart
  const chartLimitByTimeframe: Record<string, number> = { '5m': 96, '1H': 168, '4H': 42, '1D': 24, '1W': 7 };
  const chartLimit = chartLimitByTimeframe[selectedTimeframe] ?? 100;

  useEffect(() => {
    if (chartTab !== 'apr') return;
    setChartLoading(true);
    fetch(`/api/funding/history/${encodeURIComponent(backendSymbol)}?limit=${chartLimit}`, { cache: 'no-store' })
      .then((res) => res.json())
      .then((data: { series?: ChartPoint[] }) => {
        setChartSeries(Array.isArray(data?.series) ? data.series : []);
      })
      .catch(() => setChartSeries([]))
      .finally(() => setChartLoading(false));
  }, [backendSymbol, chartTab, chartLimit]);

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

  // Persist favorite by symbol
  useEffect(() => {
    try {
      const key = 'findash-options-favorites';
      const raw = localStorage.getItem(key);
      const favs: string[] = raw ? JSON.parse(raw) : [];
      setIsFavorite(favs.includes(symbol));
    } catch {
      setIsFavorite(false);
    }
  }, [symbol]);

  const toggleFavorite = () => {
    try {
      const key = 'findash-options-favorites';
      const raw = localStorage.getItem(key);
      const favs: string[] = raw ? JSON.parse(raw) : [];
      const next = favs.includes(symbol) ? favs.filter((s) => s !== symbol) : [...favs, symbol];
      localStorage.setItem(key, JSON.stringify(next));
      setIsFavorite(next.includes(symbol));
      toast({ title: next.includes(symbol) ? 'Added to favorites' : 'Removed from favorites', type: 'success' });
    } catch {
      toast({ title: 'Could not update favorites', type: 'error' });
    }
  };

  const refetchChart = useCallback(() => {
    setChartLoading(true);
    const limit = chartLimitByTimeframe[selectedTimeframe] ?? 100;
    fetch(`/api/funding/history/${encodeURIComponent(backendSymbol)}?limit=${limit}`, { cache: 'no-store' })
      .then((res) => res.json())
      .then((data: { series?: ChartPoint[] }) => {
        setChartSeries(Array.isArray(data?.series) ? data.series : []);
      })
      .catch(() => setChartSeries([]))
      .finally(() => setChartLoading(false));
  }, [backendSymbol, selectedTimeframe]);

  const handlePlaceOrder = () => {
    if (notionalEth <= 0) {
      toast({ title: 'Enter notional size', type: 'warning' });
      return;
    }
    if (!canOpen) {
      toast({ title: 'Insufficient balance', type: 'error' });
      return;
    }
    setPlacingOrder(true);
    setTimeout(() => {
      const newPos: FundingPosition = {
        id: `pos-${Date.now()}`,
        symbol,
        direction: tradeDirection,
        notionalEth,
        entryApr: metrics.impliedApr,
        currentApr: metrics.impliedApr,
        marginMode: leverage,
        openTime: new Date().toISOString(),
        reduceOnly,
      };
      setPositions((prev) => [newPos, ...prev]);
      setPlacingOrder(false);
      toast({
        title: 'Order placed (simulated)',
        description: `${tradeDirection === 'long' ? 'Long' : 'Short'} ${notionalEth} ETH @ ${metrics.impliedApr.toFixed(2)}% APR`,
        type: 'success',
      });
    }, 800);
  };

  const closePosition = (pos: FundingPosition) => {
    setPositions((prev) => prev.filter((p) => p.id !== pos.id));
    setSelectedPosition(null);
    toast({ title: 'Position closed (simulated)', description: `${pos.symbol} ${pos.direction}`, type: 'success' });
  };

  // Build chart path from series (annualized %) — viewBox 0 0 500 280; area = line + close to bottom
  const chartPaths = (() => {
    if (!chartSeries.length) return { line: null, area: null };
    const pts = chartSeries.map((p) => p.funding_rate_annualized_pct);
    const min = Math.min(...pts);
    const max = Math.max(...pts);
    const range = max - min || 1;
    const h = 280;
    const w = 500;
    const x = (i: number) => (pts.length > 1 ? (i / (pts.length - 1)) * w : 0);
    const y = (v: number) => h - ((v - min) / range) * (h - 20);
    const linePath = pts.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ');
    const areaPath = linePath
      ? `${linePath} L ${x(pts.length - 1)} ${h} L ${x(0)} ${h} Z`
      : null;
    return { line: linePath, area: areaPath };
  })();

  return (
    <div className="flex flex-col h-full min-h-[calc(100vh-8rem)] bg-background text-foreground">
      {selectedStrategy && (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-2 px-4 bg-primary/10 border-b text-sm flex-wrap">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Target className="h-4 w-4 shrink-0 text-primary" />
            <span className="font-medium truncate">Trading with strategy: {selectedStrategy.name}</span>
            {selectedStrategy.description && (
              <span className="text-muted-foreground truncate hidden sm:inline">— {selectedStrategy.description}</span>
            )}
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <span className={`font-semibold tabular-nums ${strategyPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              Strategy PnL: {strategyPnl >= 0 ? '+' : ''}${strategyPnl.toFixed(2)}
            </span>
            {onClearStrategy && (
              <Button variant="ghost" size="sm" className="shrink-0" onClick={onClearStrategy}>
                Clear
              </Button>
            )}
          </div>
        </div>
      )}
      {/* Asset bar — responsive: stack on small screens */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-3 px-4 border-b">
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          <button
            type="button"
            onClick={toggleFavorite}
            className="p-1 rounded-md hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
            aria-label={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
          >
            <Star className={`h-4 w-4 ${isFavorite ? 'fill-yellow-500 text-yellow-500' : 'text-muted-foreground hover:text-yellow-500'}`} />
          </button>
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
          <button
            type="button"
            onClick={() => setInfoOpen(true)}
            className="p-1 rounded-md hover:bg-muted focus:outline-none focus:ring-2 focus:ring-ring"
            aria-label="Pair info"
          >
            <Info className="h-4 w-4 text-muted-foreground hover:text-foreground" />
          </button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => fetchFunding()}
            disabled={loadingFunding}
            aria-label="Refresh funding"
          >
            <RefreshCw className={`h-4 w-4 ${loadingFunding ? 'animate-spin' : ''}`} />
          </Button>
          {fundingError && (
            <span className="text-xs text-destructive max-w-[180px] truncate" title={fundingError}>
              {fundingError}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-2 px-2 py-1 rounded-md hover:bg-muted text-left focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <span className="text-lg font-semibold">{selectedExpiry.label}</span>
                <span className="text-sm text-muted-foreground">(Matures {selectedExpiry.matures})</span>
                <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {expiryOptions.map((opt) => (
                <DropdownMenuItem
                  key={opt.label}
                  onClick={() => setSelectedExpiry(opt)}
                >
                  {opt.label} (Matures {opt.matures})
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Pair info dialog */}
      <Dialog open={infoOpen} onOpenChange={setInfoOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>{symbol} — Funding rate option</DialogTitle>
            <DialogDescription>
              Perpetual funding rate option. Implied APR reflects the fixed rate; underlying APR is the floating reference.
              Settlement every 8h. Long = pay fixed, receive floating; Short = pay floating, receive fixed.
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>

      {/* Chart settings dialog */}
      <Dialog open={chartSettingsOpen} onOpenChange={setChartSettingsOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Chart settings</DialogTitle>
            <DialogDescription>
              Display options for the APR chart. More indicators can be added when connected to a full charting library.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>Chart type: <span className="text-foreground font-medium">{chartType}</span></p>
            <p>Timeframe: <span className="text-foreground font-medium">{selectedTimeframe}</span></p>
          </div>
        </DialogContent>
      </Dialog>

      {/* Position detail dialog — full detail when user clicks a position */}
      <Dialog open={!!selectedPosition} onOpenChange={(open) => !open && setSelectedPosition(null)}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedPosition?.direction === 'long' ? (
                <TrendingUp className="h-5 w-5 text-green-500" />
              ) : (
                <TrendingDown className="h-5 w-5 text-red-500" />
              )}
              Position — {selectedPosition?.symbol} {selectedPosition?.direction.toUpperCase()}
            </DialogTitle>
            <DialogDescription>Full position details. Close position to realize PnL (simulated).</DialogDescription>
          </DialogHeader>
          {selectedPosition && (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="col-span-2 flex justify-between border-b pb-2">
                <span className="text-muted-foreground">Symbol</span>
                <span className="font-semibold">{selectedPosition.symbol}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Direction</span>
                <span className={selectedPosition.direction === 'long' ? 'text-green-600' : 'text-red-600'}>{selectedPosition.direction.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Notional</span>
                <span className="tabular-nums">{selectedPosition.notionalEth} ETH</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Entry APR</span>
                <span className="tabular-nums">{selectedPosition.entryApr.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Current APR</span>
                <span className="tabular-nums">{selectedPosition.currentApr.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Margin mode</span>
                <span>{selectedPosition.marginMode}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Reduce only</span>
                <span>{selectedPosition.reduceOnly ? 'Yes' : 'No'}</span>
              </div>
              <div className="col-span-2 flex justify-between border-t pt-2">
                <span className="text-muted-foreground">Opened</span>
                <span className="tabular-nums">{new Date(selectedPosition.openTime).toLocaleString()}</span>
              </div>
              <div className="col-span-2 flex justify-between">
                <span className="text-muted-foreground">Est. unrealized (per 8h)</span>
                <span className={selectedPosition.direction === 'long' ? 'text-red-500' : 'text-green-500'}>
                  {selectedPosition.direction === 'long'
                    ? (selectedPosition.notionalEth * (selectedPosition.currentApr - selectedPosition.entryApr) / 100 * (8 / (365 * 24))).toFixed(4)
                    : (selectedPosition.notionalEth * (selectedPosition.entryApr - selectedPosition.currentApr) / 100 * (8 / (365 * 24))).toFixed(4)}{' '}
                  ETH
                </span>
              </div>
            </div>
          )}
          {selectedPosition && (
            <div className="flex gap-2 justify-end pt-2 border-t">
              <Button variant="outline" onClick={() => setSelectedPosition(null)}>Done</Button>
              <Button variant="destructive" onClick={() => closePosition(selectedPosition)}>Close position</Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Metrics row — wrap by view size */}
      <div className="flex flex-wrap items-center gap-4 sm:gap-6 md:gap-8 py-4 px-4 border-b">
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

      {/* Top: APR Chart — horizontal (chart left, controls right) */}
      <div className="px-4 pt-4">
        <Card className="border rounded-lg">
          <CardContent className="p-4 flex flex-col lg:flex-row gap-4">
            {/* Chart area — takes space on left */}
            <div className="relative flex-1 min-h-[200px] lg:min-h-[180px] bg-muted/30 rounded-lg overflow-hidden transition-opacity duration-200">
                {chartLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-muted/20 z-10">
                    <span className="text-sm text-muted-foreground">در حال بارگذاری نمودار…</span>
                  </div>
                )}
                <div className="absolute left-2 top-0 bottom-0 flex flex-col justify-between text-[10px] text-muted-foreground py-4">
                  {chartSeries.length > 0 ? (
                    <>
                      <span>{Math.max(...chartSeries.map((p) => p.funding_rate_annualized_pct)).toFixed(1)}%</span>
                      <span>0%</span>
                      <span>{Math.min(...chartSeries.map((p) => p.funding_rate_annualized_pct)).toFixed(1)}%</span>
                    </>
                  ) : (
                    <>
                      <span>4%</span>
                      <span>0%</span>
                      <span>-4%</span>
                    </>
                  )}
                </div>
                <svg className="w-full h-full min-h-[200px]" viewBox="0 0 500 280" preserveAspectRatio="none" style={{ transition: 'opacity 0.2s ease' }}>
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
                  {chartPaths.area && chartType === 'area' && (
                    <path
                      d={chartPaths.area}
                      fill="rgb(59, 130, 246)"
                      fillOpacity="0.2"
                      stroke="none"
                    />
                  )}
                  {chartPaths.line ? (
                    <path
                      d={chartPaths.line}
                      fill="none"
                      stroke="rgb(59, 130, 246)"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
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
                <div className="absolute top-2 right-2 flex flex-wrap items-center justify-end gap-1">
                  {timeframes.map((tf) => (
                    <button
                      key={tf}
                      type="button"
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
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50"
                        aria-label="Chart type"
                      >
                        <BarChart3 className="h-4 w-4" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuRadioGroup value={chartType} onValueChange={(v) => setChartType(v as 'line' | 'area')}>
                        <DropdownMenuRadioItem value="line">Line</DropdownMenuRadioItem>
                        <DropdownMenuRadioItem value="area">Area</DropdownMenuRadioItem>
                      </DropdownMenuRadioGroup>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <button
                    type="button"
                    onClick={() => setChartSettingsOpen(true)}
                    className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    aria-label="Chart settings"
                  >
                    <Settings className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => refetchChart()}
                    disabled={chartLoading}
                    className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 disabled:opacity-50"
                    aria-label="Refresh chart"
                  >
                    <RefreshCw className={`h-4 w-4 ${chartLoading ? 'animate-spin' : ''}`} />
                  </button>
                </div>
              </div>
            {/* Right — Tabs, OHLC, timeframes (horizontal strip) */}
            <div className="flex flex-row flex-wrap items-center gap-3 lg:gap-4 lg:flex-col lg:items-start lg:min-w-[200px] shrink-0">
              <div className="flex items-center gap-2">
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
                  <div className="flex items-center gap-3 text-xs flex-wrap">
                    <span className="text-muted-foreground">Implied APR</span>
                    <span className="text-green-500">O{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">H{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">L{metrics.impliedApr.toFixed(2)}%</span>
                    <span className="text-green-500">C{metrics.impliedApr.toFixed(2)}%</span>
                  </div>
                  <div className="text-xs text-red-500">
                    Underlying APR <span className="ml-2">{metrics.underlyingApr.toFixed(2)}%</span>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main: Order book | Order entry — orderbook wider */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 p-4 min-h-0">
        {/* Order book — more width */}
        <div className="lg:col-span-5 flex flex-col min-h-0 min-h-[240px]">
          <Card className="flex-1 flex flex-col min-h-0 border rounded-lg">
            <CardHeader className="pb-2">
              <div className="flex flex-wrap items-center gap-4">
                <button
                  type="button"
                  className={`text-sm pb-1 border-b-2 ${orderbookTab === 'orderbook' ? 'border-primary font-medium' : 'border-transparent text-muted-foreground'}`}
                  onClick={() => setOrderbookTab('orderbook')}
                >
                  Orderbook
                </button>
                <button
                  type="button"
                  className={`text-sm pb-1 border-b-2 ${orderbookTab === 'trades' ? 'border-primary font-medium' : 'border-transparent text-muted-foreground'}`}
                  onClick={() => setOrderbookTab('trades')}
                >
                  Market Trades
                </button>
              </div>
            </CardHeader>
            <CardContent className="p-4 flex-1 overflow-auto">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                    >
                      {orderbookSpread}
                      <ChevronDown className="h-3 w-3" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    {spreadOptions.map((s) => (
                      <DropdownMenuItem key={s} onClick={() => setOrderbookSpread(s)}>
                        {s}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      className="flex items-center gap-1 p-1 rounded hover:bg-muted text-muted-foreground"
                      aria-label="Order book depth"
                    >
                      {orderbookDepth} levels
                      <ChevronDown className="h-3 w-3" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    {depthOptions.map((d) => (
                      <DropdownMenuItem key={d} onClick={() => setOrderbookDepth(d)}>
                        {d} levels
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    title="Full orderbook view"
                    className={`w-8 h-6 rounded text-[10px] font-medium transition-colors ${!orderbookCompact ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}
                    onClick={() => setOrderbookCompact(false)}
                  >
                    Full
                  </button>
                  <button
                    type="button"
                    title="Compact orderbook view"
                    className={`w-8 h-6 rounded text-[10px] font-medium transition-colors ${orderbookCompact ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'}`}
                    onClick={() => setOrderbookCompact(true)}
                  >
                    Compact
                  </button>
                </div>
              </div>
              <div className="mb-4">
                <div className="text-xs text-red-500 font-semibold mb-2">SHORT RATE</div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Implied APR (%)</span>
                  <span>Size (ETH YU)</span>
                </div>
                <div className={`space-y-1 ${orderbookCompact ? 'space-y-0' : ''}`}>
                  {(orderbookCompact ? SHORT_RATE_ORDERS.slice(0, 5) : SHORT_RATE_ORDERS).map((order, i) => (
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
                <span className="text-muted-foreground">{orderbookSpread} Spread</span>
              </div>
              <div>
                <div className="text-xs text-green-500 font-semibold mb-2">LONG RATE</div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Implied APR (%)</span>
                  <span>Size (ETH YU)</span>
                </div>
                <div className={`space-y-1 ${orderbookCompact ? 'space-y-0' : ''}`}>
                  {(orderbookCompact ? LONG_RATE_ORDERS.slice(0, 5) : LONG_RATE_ORDERS).map((order, i) => (
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
        <div className="lg:col-span-7 flex flex-col min-h-0">
          <Card className="flex-1 flex flex-col min-h-0 border rounded-lg">
            <CardContent className="p-4 flex flex-col">
              {!makerRewardsBannerDismissed && (
                <div className="bg-amber-500/20 border border-amber-500/30 rounded-lg p-3 mb-4 flex items-center gap-2">
                  <button
                    type="button"
                    className="flex-1 flex items-center gap-2 text-left hover:opacity-90 transition-opacity"
                    onClick={() => toast({ title: 'Opening Maker Rewards…', description: 'Rewards program details coming soon.', type: 'success' })}
                  >
                    <Gift className="h-4 w-4 text-amber-400 shrink-0" />
                    <span className="text-sm text-amber-200">Maker Order Rewards Live!</span>
                  </button>
                  <button
                    type="button"
                    aria-label="Dismiss"
                    className="p-1 rounded hover:bg-amber-500/30 text-amber-200"
                    onClick={() => setMakerRewardsBannerDismissed(true)}
                  >
                    <span className="text-sm leading-none">×</span>
                  </button>
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
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
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-4">
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
                  <span className="text-cyan-500">{notionalEth} ETH</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Available to Trade</span>
                  <span>{MOCK_AVAILABLE_ETH} ETH</span>
                </div>
              </div>
              <div className="mb-4">
                <Label className="text-sm text-muted-foreground">Notional Size (ETH)</Label>
                <div className="flex items-center gap-2 mt-1">
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={notionalSize[0]}
                    onChange={(e) => setNotionalSize([Math.max(0, Math.min(100, parseInt(e.target.value, 10) || 0))])}
                    className="bg-muted/50"
                    placeholder="ETH"
                  />
                  <span className="text-xs text-muted-foreground">ETH</span>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <Slider
                    value={notionalSize}
                    onValueChange={setNotionalSize}
                    max={100}
                    step={1}
                    className="flex-1"
                  />
                  <span className="text-xs w-10 tabular-nums">{notionalSize[0]} ETH</span>
                </div>
              </div>
              {/* Order calculation preview — user sees what happens when toggling */}
              <div className="mb-4 rounded-lg border bg-muted/30 p-3 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Calculator className="h-4 w-4 text-primary" />
                  Order preview
                </div>
                <div className="text-xs space-y-1.5">
                  <p>
                    <span className="text-muted-foreground">Margin mode:</span>{' '}
                    <span className="font-medium">
                      {leverage === 'Cross'
                        ? 'Cross — your full balance backs this position (shared collateral).'
                        : leverage === '2x'
                          ? '2x — isolated margin, 50% of notional required.'
                          : 'One-way — isolated, full notional as margin (1x).'}
                    </span>
                  </p>
                  <p>
                    <span className="text-muted-foreground">Direction:</span>{' '}
                    <span className="font-medium">
                      {tradeDirection === 'long'
                        ? 'Long — you pay fixed rate (~' + metrics.impliedApr.toFixed(2) + '% APR), receive floating (underlying).'
                        : 'Short — you pay floating (underlying), receive fixed rate.'}
                    </span>
                  </p>
                  <p>
                    <span className="text-muted-foreground">Notional:</span>{' '}
                    <span className="font-medium">{notionalEth} ETH</span>
                    {notionalEth > 0 && (
                      <>
                        {' • '}
                        <span className="text-muted-foreground">Margin required:</span>{' '}
                        <span className={canOpen ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                          {marginRequiredEth.toFixed(2)} ETH
                        </span>
                      </>
                    )}
                  </p>
                  {notionalEth > 0 && (
                    <p>
                      <span className="text-muted-foreground">Est. funding per 8h:</span>{' '}
                      <span className="font-medium">
                        {tradeDirection === 'long' ? '−' : '+'}{estFundingPer8h.toFixed(4)} ETH
                      </span>
                      {' (you '}
                      {tradeDirection === 'long' ? 'pay' : 'receive'} fixed)
                    </p>
                  )}
                  {notionalEth > 0 && !canOpen && (
                    <p className="text-red-600 dark:text-red-400 font-medium">
                      Insufficient balance. Need {marginRequiredEth.toFixed(2)} ETH, have {MOCK_AVAILABLE_ETH} ETH.
                    </p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 mb-4">
                <Checkbox
                  id="reduce-only"
                  checked={reduceOnly}
                  onCheckedChange={(c) => setReduceOnly(c === true)}
                />
                <Label htmlFor="reduce-only" className="text-sm text-muted-foreground cursor-pointer">
                  Reduce Only
                </Label>
              </div>
              <Button
                className="w-full mb-4"
                size="lg"
                onClick={handlePlaceOrder}
                disabled={notionalEth <= 0 || !canOpen || placingOrder}
              >
                {placingOrder ? 'Placing…' : orderType === 'market' ? 'Place Market Order' : 'Place Limit Order'}
              </Button>
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

      {/* Open Positions — click row for full detail */}
      {positions.length > 0 && (
        <div className="px-4 py-3 border-t">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Open Positions</h3>
          <div className="rounded-lg border bg-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="text-left py-2 px-3 font-medium">Symbol</th>
                    <th className="text-left py-2 px-3 font-medium">Side</th>
                    <th className="text-right py-2 px-3 font-medium">Size</th>
                    <th className="text-right py-2 px-3 font-medium">Entry APR</th>
                    <th className="text-right py-2 px-3 font-medium">Current APR</th>
                    <th className="text-left py-2 px-3 font-medium">Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos) => (
                    <tr
                      key={pos.id}
                      onClick={() => setSelectedPosition(pos)}
                      className="border-b last:border-0 hover:bg-muted/50 cursor-pointer transition-colors"
                    >
                      <td className="py-2.5 px-3 font-medium">{pos.symbol}</td>
                      <td className="py-2.5 px-3">
                        <span className={pos.direction === 'long' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                          {pos.direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right tabular-nums">{pos.notionalEth} ETH</td>
                      <td className="py-2.5 px-3 text-right tabular-nums">{pos.entryApr.toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-right tabular-nums">{pos.currentApr.toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-muted-foreground">{pos.marginMode}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground px-3 py-2 border-t">Click a row to view full details</p>
          </div>
        </div>
      )}

      {/* Footer — responsive wrap */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 px-4 py-2 border-t text-xs text-muted-foreground">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
            <span>Online</span>
          </div>
          <span>Gas: $0.02</span>
        </div>
        <div className="flex flex-wrap items-center gap-3 sm:gap-4">
          <button type="button" onClick={() => toast({ title: 'Docs', description: 'Documentation coming soon.', type: 'info' })} className="hover:text-foreground">
            Docs
          </button>
          <button type="button" onClick={() => toast({ title: 'Support', description: 'Support page coming soon.', type: 'info' })} className="hover:text-foreground">
            Support
          </button>
          <button type="button" onClick={() => toast({ title: 'Terms', description: 'Terms of service coming soon.', type: 'info' })} className="hover:text-foreground">
            Terms
          </button>
          <button type="button" onClick={() => toast({ title: 'Policy', description: 'Privacy policy coming soon.', type: 'info' })} className="hover:text-foreground">
            Policy
          </button>
          <a href="https://docs.findash.example/help" target="_blank" rel="noopener noreferrer" className="hover:text-cyan-500">Help & Support</a>
        </div>
      </div>
    </div>
  );
}
