'use client';

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, GlassCard } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency, formatPercentage, CurrencyUnit } from '@/lib/utils';
import { PortfolioChart } from '@/components/portfolio/portfolio-chart';
import { IranPortfolioSection } from '@/components/portfolio/iran-portfolio-section';
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
  AAPL: 'فناوری', MSFT: 'فناوری', GOOGL: 'فناوری', GOOG: 'فناوری',
  AMZN: 'مصرفی', META: 'فناوری', NVDA: 'فناوری', TSLA: 'مصرفی',
  JNJ: 'سلامت', PG: 'مصرفی', KO: 'مصرفی', JPM: 'مالی',
};

const CURRENCY_LABELS: Record<CurrencyUnit, string> = {
  IRT: 'تومان',
  IRR: 'ریال',
  USD: 'دلار',
};

const MOCK_PORTFOLIOS: Portfolio[] = [
  { id: '1', name: 'پورتفولیو نمونه', description: 'نمایش عمومی', initial_capital: 1_000_000_000, current_value: 1_247_500_000, cash_balance: 185_000_000, total_return: 247_500_000, total_return_percent: 24.75 },
  { id: '2', name: 'سبد رشد', description: 'تمرکز بر سهام رشدی', initial_capital: 650_000_000, current_value: 782_000_000, cash_balance: 92_000_000, total_return: 132_000_000, total_return_percent: 20.3 },
];

const MOCK_POSITIONS: Position[] = [
  { symbol: 'فولاد', quantity: 12_500, average_cost: 5600, market_value: 82_500_000, unrealized_pnl: 12_500_000, unrealized_pnl_percent: 17.85, weight: 18.6 },
  { symbol: 'شستا', quantity: 24_000, average_cost: 1180, market_value: 34_800_000, unrealized_pnl: 6_480_000, unrealized_pnl_percent: 22.88, weight: 7.85 },
  { symbol: 'خودرو', quantity: 18_000, average_cost: 410, market_value: 6_930_000, unrealized_pnl: -450_000, unrealized_pnl_percent: -6.1, weight: 1.56 },
  { symbol: 'BTC', quantity: 0.18, average_cost: 2_750_000_000, market_value: 585_000_000, unrealized_pnl: 90_000_000, unrealized_pnl_percent: 18.18, weight: 42.1 },
];

export function PortfolioContent() {
  const [portfolios] = useState<Portfolio[]>(MOCK_PORTFOLIOS);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(MOCK_PORTFOLIOS[0]);
  const [positions] = useState<Position[]>(MOCK_POSITIONS);
  const [currencyUnit, setCurrencyUnit] = useState<CurrencyUnit>('IRT');

  const fmt = (v: number) => formatCurrency(v, currencyUnit);

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
      sector: DEFAULT_SECTOR[p.symbol] ?? 'سهام',
      type: 'stock' as const,
    }));
  }, [positions, investedValue, totalValue]);

  const topGainer = positions.length ? [...positions].sort((a, b) => b.unrealized_pnl - a.unrealized_pnl)[0] : null;
  const topLoser = positions.length ? [...positions].sort((a, b) => a.unrealized_pnl - b.unrealized_pnl)[0] : null;

  return (
    <div className="space-y-6">
      {/* Iranian Portfolio Section */}
      <IranPortfolioSection />

      <div className="border-t pt-4" />

      {/* Summary hero */}
      <GlassCard className="border-white/30 dark:border-white/20 bg-card/50 backdrop-blur">
        <CardContent className="p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              {/* Currency unit switcher */}
              <div className="flex items-center gap-1 mb-3">
                {(['IRT', 'IRR', 'USD'] as CurrencyUnit[]).map((u) => (
                  <button
                    key={u}
                    onClick={() => setCurrencyUnit(u)}
                    className={cn(
                      'rounded-lg px-3 py-1 text-xs font-bold transition border',
                      currencyUnit === u
                        ? 'bg-green-500/15 border-green-500/30 text-green-400'
                        : 'border-white/10 text-slate-400 hover:text-white hover:bg-white/[0.04]'
                    )}
                  >
                    {CURRENCY_LABELS[u]}
                  </button>
                ))}
              </div>
              <p className="text-sm font-medium text-muted-foreground uppercase tracking-wider">ارزش پورتفولیو</p>
              <p className="text-3xl font-bold tracking-tight text-foreground mt-1">{fmt(totalValue)}</p>
              <div className="flex items-center gap-3 mt-2">
                <span className={cn(
                  'inline-flex items-center gap-1 text-sm font-medium',
                  totalPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                )}>
                  {totalPnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  {totalPnl >= 0 ? '+' : ''}{fmt(totalPnl)} سود/زیان کل
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
                <p className="text-xs text-muted-foreground uppercase tracking-wider">نقد</p>
                <p className="text-lg font-semibold flex items-center gap-1.5 justify-end">
                  <Wallet className="h-4 w-4 text-muted-foreground" />
                  {fmt(cashBalance)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">سرمایه‌گذاری‌شده</p>
                <p className="text-lg font-semibold flex items-center gap-1.5 justify-end">
                  <Briefcase className="h-4 w-4 text-muted-foreground" />
                  {fmt(investedValue)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">موقعیت‌ها</p>
                <p className="text-lg font-semibold">{positions.length}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </GlassCard>

      {/* Portfolio selector */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-muted-foreground mr-2">پورتفولیو:</span>
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
                  تخصیص دارایی
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <DollarSign className="h-12 w-12 mb-3 opacity-50" />
                  <p className="font-medium">هنوز موقعیتی وجود ندارد</p>
                  <p className="text-sm">برای مشاهده تخصیص و تفکیک بخش، موقعیت اضافه کنید</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        <Card className="border-white/30 dark:border-white/20">
          <CardHeader>
            <CardTitle className="text-lg">آمار سریع</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">سرمایه اولیه</span>
              <span className="font-medium">{fmt(totalCost)}</span>
            </div>
            {topGainer && (
              <div className="rounded-lg bg-green-500/10 dark:bg-green-500/20 p-3 border border-green-500/20">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">بیشترین سود</p>
                <p className="font-semibold text-green-600 dark:text-green-400">{topGainer.symbol}</p>
                <p className="text-sm">+{fmt(topGainer.unrealized_pnl)}</p>
              </div>
            )}
            {topLoser && topLoser.symbol !== topGainer?.symbol && (
              <div className="rounded-lg bg-red-500/10 dark:bg-red-500/20 p-3 border border-red-500/20">
                <p className="text-xs text-muted-foreground uppercase tracking-wider">بیشترین زیان</p>
                <p className="font-semibold text-red-600 dark:text-red-400">{topLoser.symbol}</p>
                <p className="text-sm">{fmt(topLoser.unrealized_pnl)}</p>
              </div>
            )}
            {selectedPortfolio?.risk_tolerance && (
              <div className="flex justify-between text-sm pt-2 border-t">
                <span className="text-muted-foreground">ریسک</span>
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
            <CardTitle>دارایی‌ها — {selectedPortfolio.name}</CardTitle>
            <Button size="sm">
              <Plus className="h-4 w-4 mr-2" />
              افزودن موقعیت
            </Button>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Briefcase className="h-10 w-10 mx-auto mb-2 opacity-50" />
                <p className="font-medium">موقعیتی وجود ندارد</p>
                <p className="text-sm">برای پیگیری عملکرد و تخصیص، موقعیت اضافه کنید</p>
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="text-left py-3 px-4 font-medium">نماد</th>
                      <th className="text-right py-3 px-4 font-medium">تعداد</th>
                      <th className="text-right py-3 px-4 font-medium">میانگین قیمت</th>
                      <th className="text-right py-3 px-4 font-medium">ارزش بازار</th>
                      <th className="text-right py-3 px-4 font-medium">وزن</th>
                      <th className="text-right py-3 px-4 font-medium">سود/زیان</th>
                      <th className="text-right py-3 px-4 font-medium">درصد سود/زیان</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p, i) => (
                      <tr key={`${p.symbol}-${i}`} className="border-b last:border-0 hover:bg-muted/30">
                        <td className="py-3 px-4 font-medium">{p.symbol}</td>
                        <td className="text-right py-3 px-4">{p.quantity.toLocaleString()}</td>
                        <td className="text-right py-3 px-4">{fmt(p.average_cost ?? 0)}</td>
                        <td className="text-right py-3 px-4">{fmt(p.market_value)}</td>
                        <td className="text-right py-3 px-4 text-muted-foreground">{p.weight?.toFixed(1) ?? '—'}%</td>
                        <td className={cn('text-right py-3 px-4 font-medium', p.unrealized_pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400')}>
                          {p.unrealized_pnl >= 0 ? '+' : ''}{fmt(p.unrealized_pnl)}
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
