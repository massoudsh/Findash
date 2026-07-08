'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { Copy, TrendingUp, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M8 Paper Trading Agent: sim performance summary */
const MOCK_SIM = [
  { label: 'سود/زیان امروز', value: '+۱۲۴.۵۰$', positive: true },
  { label: 'موقعیت‌های باز', value: '۵', positive: null },
  { label: 'سفارشات (۲۴ساعت)', value: '۱۲', positive: null },
];

export function PaperTradingAgentPanel() {
  return (
    <AgentPanel
      title="معامله کاغذی (M8)"
      subtitle="اجرای شبیه‌سازی‌شده و عملکرد"
      icon={<Copy className="h-4 w-4 text-primary" />}
      agentId="M8"
    >
      <ul className="space-y-2 pr-2">
        {MOCK_SIM.map((r) => (
          <li
            key={r.label}
            className="rounded-lg border border-border bg-muted/20 p-2.5 text-xs flex items-center justify-between gap-2"
          >
            <span className="text-muted-foreground flex items-center gap-1.5">
              <BarChart3 className="h-3 w-3" />
              {r.label}
            </span>
            <span
              className={cn(
                'font-medium tabular-nums',
                r.positive === true && 'text-emerald-600 dark:text-emerald-400',
                r.positive === false && 'text-red-600 dark:text-red-400'
              )}
            >
              {r.value}
            </span>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
