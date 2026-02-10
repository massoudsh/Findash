import { Suspense } from 'react';
import { BacktestRunner } from '@/components/backtesting/backtest-runner';

export default function BacktestingPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Strategy Backtesting</h1>
        <p className="text-muted-foreground">
          Test your trading strategies against historical data.
        </p>
      </div>
      <Suspense fallback={<div>Loading backtesting interface...</div>}>
        <BacktestRunner />
      </Suspense>
    </div>
  );
} 