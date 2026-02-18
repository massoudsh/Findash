'use client';

import { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, GlassCard } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { getPortfolios, getPositions } from '@/lib/services/api';
import { PortfolioChart } from '@/components/portfolio/portfolio-chart';
import {
  Plus,
  DollarSign,
  TrendingUp,
  TrendingDown,
  PieChart as PieChartIcon,
  Briefcase,
  Wallet,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface Portfolio {
  id: string | number;
  name: string;
  description?: string;
  initial_capital: number;
  current_value: number;
  cash_balance?: number;
  total_return?: number;
  total_return_percent?: number;
  risk_tolerance?: string;
  created_at?: string;
  updated_at?: string;
}

interface Position {
  symbol: string;
  quantity: number;
  average_cost?: number;
  average_price?: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent?: number;
  weight?: number;
  current_price?: number;
}

const DEFAULT_SECTOR: Record<string, string> = {
  AAPL: 'Technology', MSFT: 'Technology', GOOGL: 'Technology', GOOG: 'Technology',
  AMZN: 'Consumer', META: 'Technology', NVDA: 'Technology', TSLA: 'Consumer',
  JNJ: 'Healthcare', PG: 'Consumer', KO: 'Consumer', JPM: 'Financials',
};

function normalizePortfolios(raw: unknown): Portfolio[] {
  const arr = Array.isArray(raw) ? raw : (raw as { data?: unknown })?.data;
  if (!Array.isArray(arr)) return [];
  return arr.map((p: Record<string, unknown>) => ({
    id: (typeof p.id === 'string' || typeof p.id === 'number') ? p.id : String(p.id ?? ''),
    name: String(p.name ?? ''),
    description: p.description != null ? String(p.description) : undefined,
    initial_capital: Number(p.initial_capital ?? 0),
    current_value: Number(p.current_value ?? 0),
    cash_balance: p.cash_balance != null ? Number(p.cash_balance) : undefined,
    total_return: p.total_return != null ? Number(p.total_return) : undefined,
    total_return_percent: p.total_return_percent != null ? Number(p.total_return_percent) : undefined,
    risk_tolerance: p.risk_tolerance != null ? String(p.risk_tolerance) : undefined,
    created_at: p.created_at != null ? String(p.created_at) : undefined,
    updated_at: p.updated_at != null ? String(p.updated_at) : undefined,
  }));
}

function normalizePositions(raw: unknown, totalValue: number): Position[] {
  const arr = Array.isArray(raw) ? raw : (raw as { data?: unknown })?.data;
  if (!Array.isArray(arr)) return [];
  const total = arr.reduce((sum: number, p: Record<string, unknown>) => sum + Number(p.market_value ?? 0), 0) || totalValue || 1;
  return arr.map((p: Record<string, unknown>) => {
    const marketValue = Number(p.market_value ?? 0);
    const qty = Number(p.quantity ?? 0);
    const avg = Number(p.average_cost ?? p.average_price ?? 0);
    const pnl = Number(p.unrealized_pnl ?? 0);
    const cost = avg * qty;
    const pnlPct = cost ? (pnl / cost) * 100 : 0;
    return {
      symbol: String(p.symbol ?? ''),
      quantity: qty,
      average_cost: avg,
      market_value: marketValue,
      unrealized_pnl: pnl,
      unrealized_pnl_percent: pnlPct,
      weight: (marketValue / total) * 100,
    };
  });
}

export function PortfolioContent() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchPortfolios() {
      try {
        const response = await getPortfolios();
        const data = normalizePortfolios(response?.data ?? response);
        setPortfolios(data);
        if (data.length > 0 && !selectedPortfolio) setSelectedPortfolio(data[0]);
      } catch (error) {
        console.error('Error fetching portfolios:', error);
        const mock: Portfolio[] = [
          { id: '1', name: 'Main Portfolio', description: 'Primary', initial_capital: 100000, current_value: 125000, cash_balance: 25000, total_return: 25000, total_return_percent: 25 },
          { id: '2', name: 'Tech Stocks', description: 'Tech focus', initial_capital: 50000, current_value: 62000, cash_balance: 12000, total_return: 12000, total_return_percent: 24 },
        ];
        setPortfolios(mock);
        setSelectedPortfolio(mock[0]);
      } finally {
        setIsLoading(false);
      }
    }
    fetchPortfolios();
  }, []);

  useEffect(() => {
    if (!selectedPortfolio) return;
    async function fetchPositions() {
      try {
        const response = await getPositions(selectedPortfolio.id);
        const total = selectedPortfolio.current_value || 1;
        setPositions(normalizePositions(response?.data ?? response, total));
      } catch (error) {
        console.error('Error fetching positions:', error);
        const fallbackTotal = selectedPortfolio?.current_value || 1;
        setPositions(normalizePositions([
          { symbol: 'AAPL', quantity: 100, average_cost: 150, market_value: 17500, unrealized_pnl: 2500 },
          { symbol: 'MSFT', quantity: 50, average_cost: 300, market_value: 16000, unrealized_pnl: 1000 },
        ], fallbackTotal));
      }
    }
    fetchPositions();
  }, [selectedPortfolio]);

  const totalValue = selectedPortfolio?.current_value ?? 0;
  const totalCost = selectedPortfolio?.initial_capital ?? 0;
  const totalPnl = totalValue - totalCost;
  const totalPnlPct = totalCost ? (totalPnl / totalCost) * 100 : 0;
  const cashBalance = selectedPortfolio?.cash_balance ?? 0;
  const investedValue = positions.reduce((s, p) => s + p.market_value, 0);

  const chartAssets = useMemo(() => {
    const total = investedValue || totalValue || 1;
    return positions.map((p, i) => ({
      id: `${p.symbol}-${i}`,
      symbol: p.symbol,
      name: p.symbol,
      marketValue: p.market_value,
      allocation: (p.market_value / total) * 100,
      sector: DEFAULT_SECTOR[p.symbol] ?? 'Equities',
      type: 'stock' as const,
    }));
  }, [positions, investedValue, totalValue]);

  const topGainer = positions.length ? [...positions].sort((a, b) => b.unrealized_pnl - a.unrealized_pnl)[0] : null;
  const topLoser = positions.length ? [...positions].sort((a, b) => a.unrealized_pnl - b.unrealized_pnl)[0] : null;

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-32 rounded-lg bg-muted/50" />
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((i) => <div key={i} className="h-24 rounded-lg bg-muted/50" />)}
        </div>
        <div className="h-64 rounded-lg bg-muted/50" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary hero */}
      <GlassCard className="border-white/30 dark:border-white/20 bg-card/50 backdrop-blur">
        <CardContent className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Portfolio value</p>
              <p className="text-3xl font-bold tracking-tight text-foreground mt-1">{formatCurrency(totalValue)}</p>
              <div className="flex items-center gap-3 mt-2">
                <span className={cn(
                  'inline-flex items-center gap-1 text-sm font-medium',
                  totalPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                )}>
                  {totalPnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  {totalPnl >= 0 ? '+' : ''}{formatCurrency(totalPnl)} total P&L
                </span>
                <span className={cn(
                  'text-sm font-medium',
                  totalPnlPct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                )}>
                  ({totalPnlPct >= 0 ? '+' : ''}{totalPnlPct.toFixed(2)}%)
                </span>
              </div>
            </div>
            <div className="flex flex-wrap gap-6">
              <div className="text-right">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Cash</p>
                <p className="text-lg font-semibold flex items-center gap-1.5 justify-end">
                  <Wallet className="h-4 w-4 text-muted-foreground" />
                  {formatCurrency(cashBalance)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Invested</p>
                <p className="text-lg font-semibold flex items-center gap-1.5 justify-end">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  {formatCurrency(investedValue)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Positions</p>
                <p className="text-lg font-semibold">{positions.length}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </GlassCard>

      {/* Portfolio selector */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground mr-2">Portfolio:</span>
        {portfolios.map((p) => (
          <Button
            key={String(p.id)}
            variant={selectedPortfolio?.id === p.id ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedPortfolio(p)}
          >
            {p.name}
          </Button>
        ))}
      </div>

      {/* Allocation + top movers */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          {chartAssets.length > 0 ? (
            <PortfolioChart assets={chartAssets} totalValue={investedValue || totalValue} />
          ) : (
            <Card className="border-white/30 dark:border-white/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <PieChartIcon className="h-5 w-5" />
                  Allocation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <DollarSign className="h-12 w-12 mb-3 opacity-50" />
                  <p className="font-medium">No positions yet</p>
                  <p className="text-sm">Add positions to see allocation and sector breakdown</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        <Card className="border-white/30 dark:border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">Quick stats</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Initial capital</span>
              <span className="font-medium">{formatCurrency(totalCost)}</span>
            </div>
            {topGainer && (
              <div className="rounded-lg bg-green-500/10 dark:bg-green-500/20 p-3 border border-green-500/20">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Top gainer</p>
                <p className="font-semibold text-green-600 dark:text-green-400">{topGainer.symbol}</p>
                <p className="text-sm">+{formatCurrency(topGainer.unrealized_pnl)}</p>
              </div>
            )}
            {topLoser && topLoser.symbol !== topGainer?.symbol && (
              <div className="rounded-lg bg-red-500/10 dark:bg-red-500/20 p-3 border border-red-500/20">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Top loser</p>
                <p className="font-semibold text-red-600 dark:text-red-400">{topLoser.symbol}</p>
                <p className="text-sm">{formatCurrency(topLoser.unrealized_pnl)}</p>
              </div>
            )}
            {selectedPortfolio?.risk_tolerance && (
              <div className="flex justify-between text-sm pt-2 border-t">
                <span className="text-muted-foreground">Risk</span>
                <span className="font-medium capitalize">{selectedPortfolio.risk_tolerance}</span>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Positions table */}
      {selectedPortfolio && (
        <Card className="border-white/30 dark:border-white/20">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Holdings — {selectedPortfolio.name}</CardTitle>
            <Button size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add position
            </Button>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Briefcase className="h-10 w-10 mx-auto mb-2 opacity-50" />
                <p className="font-medium">No positions</p>
                <p className="text-sm">Add positions to track performance and allocation</p>
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="text-left py-3 px-4 font-medium">Symbol</th>
                      <th className="text-right py-3 px-4 font-medium">Qty</th>
                      <th className="text-right py-3 px-4 font-medium">Avg cost</th>
                      <th className="text-right py-3 px-4 font-medium">Market value</th>
                      <th className="text-right py-3 px-4 font-medium">Weight</th>
                      <th className="text-right py-3 px-4 font-medium">P&L</th>
                      <th className="text-right py-3 px-4 font-medium">P&L %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p, i) => (
                      <tr key={`${p.symbol}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                        <td className="py-3 px-4 font-medium">{p.symbol}</td>
                        <td className="text-right py-3 px-4">{p.quantity.toLocaleString()}</td>
                        <td className="text-right py-3 px-4">{formatCurrency(p.average_cost ?? 0)}</td>
                        <td className="text-right py-3 px-4">{formatCurrency(p.market_value)}</td>
                        <td className="text-right py-3 px-4 text-muted-foreground">{p.weight?.toFixed(1) ?? '—'}%</td>
                        <td className={cn('text-right py-3 px-4 font-medium', p.unrealized_pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')}>
                          {p.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(p.unrealized_pnl)}
                        </td>
                        <td className={cn('text-right py-3 px-4', p.unrealized_pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')}>
                          {p.unrealized_pnl_percent != null ? `${p.unrealized_pnl_percent >= 0 ? '+' : ''}${p.unrealized_pnl_percent.toFixed(2)}%` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
