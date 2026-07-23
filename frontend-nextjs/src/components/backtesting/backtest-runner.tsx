'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { runBacktest } from '@/lib/services/api';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import { BacktestChart } from '@/components/visualization/backtest-chart';
import { Info } from 'lucide-react';

interface BacktestResult {
  final_portfolio_value: number;
  total_return_pct: number;
  portfolio_history: { date: string; value: number }[];
}

export function BacktestRunner() {
  const searchParams = useSearchParams();
  const [symbol, setSymbol] = useState('BTC-USD');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [strategyType, setStrategyType] = useState('');
  const [strategyName, setStrategyName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  // Pre-fill form from URL parameters
  useEffect(() => {
    if (searchParams) {
      const urlStrategyType = searchParams.get('strategy_type');
      const urlStrategyName = searchParams.get('strategy_name');
      const urlSymbols = searchParams.get('symbols');
      const urlInitialCapital = searchParams.get('initial_capital');

      if (urlStrategyType) setStrategyType(urlStrategyType);
      if (urlStrategyName) setStrategyName(urlStrategyName);
      if (urlSymbols) setSymbol(urlSymbols);
      if (urlInitialCapital) setInitialCapital(Number(urlInitialCapital));
    }
  }, [searchParams]);

  const handleBacktest = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const backtestResult = await runBacktest(symbol, startDate, endDate, initialCapital);
      
      if (backtestResult.status === 'success') {
        setResult(backtestResult);
      } else {
        throw new Error(backtestResult.message || 'بک‌تست ناموفق بود');
      }

    } catch (err: any) {
      setError(err.message || 'خطای غیرمنتظره‌ای رخ داد.');
    } finally {
      setIsLoading(false);
    }
  };

  const getStrategyTypeLabel = (type: string) => {
    switch (type) {
      case 'momentum': return 'استراتژی مومنتوم';
      case 'technical': return 'تحلیل تکنیکال';
      case 'risk_aware': return 'استراتژی ریسک‌آگاه';
      case 'mean_reversion': return 'بازگشت به میانگین';
      case 'breakout': return 'استراتژی شکست قیمتی';
      case 'volatility_spread': return 'اسپرد نوسان';
      default: return type;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>بک‌تستر استراتژی</CardTitle>
        {strategyName && (
          <div className="flex items-center gap-2 mt-2">
            <Info className="h-4 w-4 text-blue-500" />
            <span className="text-sm text-muted-foreground">
              بک‌تست استراتژی: <strong>{strategyName}</strong>
            </span>
            {strategyType && (
              <Badge variant="outline">{getStrategyTypeLabel(strategyType)}</Badge>
            )}
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 md:grid-cols-4">
          <div className="space-y-2">
            <Label htmlFor="symbol">نماد(ها)</Label>
            <Input
              id="symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="AAPL,TSLA,MSFT"
            />
            <p className="text-xs text-muted-foreground">
              برای چند نماد، با کاما جدا کنید
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="start-date">تاریخ شروع</Label>
            <Input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="end-date">تاریخ پایان</Label>
            <Input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="initial-capital">سرمایه اولیه ($)</Label>
            <Input
              id="initial-capital"
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              min="1000"
              step="1000"
            />
          </div>
        </div>

        {strategyType && (
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h4 className="font-medium text-blue-900 mb-2">پیکربندی استراتژی</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">نوع:</span> {getStrategyTypeLabel(strategyType)}
              </div>
              <div>
                <span className="font-medium">نام:</span> {strategyName}
              </div>
            </div>
          </div>
        )}

        <Button onClick={handleBacktest} disabled={isLoading} className="w-full">
          {isLoading ? 'در حال اجرای بک‌تست...' : 'اجرای بک‌تست'}
        </Button>

        {error && (
          <div className="text-red-600 bg-red-100 p-3 rounded-md">
            <p className="font-bold">خطا</p>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="space-y-4 pt-4">
            <h3 className="text-xl font-semibold">نتایج بک‌تست</h3>
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader><CardTitle>ارزش نهایی</CardTitle></CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{formatCurrency(result.final_portfolio_value)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader><CardTitle>بازده کل</CardTitle></CardHeader>
                <CardContent>
                  <p className={`text-2xl font-bold ${result.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPercentage(result.total_return_pct / 100)}
                  </p>
                </CardContent>
              </Card>
            </div>
            <Card>
              <CardHeader><CardTitle>عملکرد در طول زمان</CardTitle></CardHeader>
              <CardContent>
                <BacktestChart data={result.portfolio_history} />
              </CardContent>
            </Card>
          </div>
        )}
      </CardContent>
    </Card>
  );
}