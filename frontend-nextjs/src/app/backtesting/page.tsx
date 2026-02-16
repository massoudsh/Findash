import { Suspense } from 'react';
import { BacktestRunner } from '@/components/backtesting/backtest-runner';
import { BacktestAgentPanel } from '@/components/agents/backtest-agent-panel';

export default function BacktestingPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Strategy Backtesting</h1>
        <p className="text-muted-foreground">
          Test your trading strategies against historical data.
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div>Loading backtesting interface...</div>}>
            <BacktestRunner />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <BacktestAgentPanel />
        </aside>
      </div>
    </div>
  );
} 