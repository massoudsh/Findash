'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { History, CheckCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M10 Backtesting Agent: last run and queue */
const MOCK_BACKTEST = [
  { name: 'مومنتوم SPY', status: 'completed' as const, result: 'شارپ ۱.۲۴' },
  { name: 'بازگشت به میانگین QQQ', status: 'completed' as const, result: 'نرخ برد ۵۸٪' },
  { name: 'استراتژی سفارشی', status: 'queued' as const, result: '—' },
];

export function BacktestAgentPanel() {
  return (
    <AgentPanel
      title="بک‌تست (M10)"
      subtitle="اعتبارسنجی استراتژی و سابقه"
      icon={<History className="h-4 w-4 text-primary" />}
      agentId="M10"
    >
      <ul className="space-y-2 pr-2">
        {MOCK_BACKTEST.map((r) => (
          <li
            key={r.name}
            className={cn(
              'rounded-lg border p-2.5 text-xs',
              r.status === 'completed' && 'border-emerald-500/20 bg-emerald-500/5',
              r.status === 'queued' && 'border-border bg-muted/20'
            )}
          >
            <div className="flex items-center justify-between gap-2 mb-0.5">
              <span className="font-medium truncate">{r.name}</span>
              {r.status === 'completed' ? (
                <CheckCircle className="h-3 w-3 shrink-0 text-emerald-500" />
              ) : (
                <Clock className="h-3 w-3 shrink-0 text-muted-foreground" />
              )}
            </div>
            <p className="text-muted-foreground text-[11px]">{r.result}</p>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
