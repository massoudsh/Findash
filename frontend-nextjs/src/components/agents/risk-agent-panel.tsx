'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M6 Risk Management Agent: risk metrics and compliance status */
const MOCK_RISK = [
  { label: 'VaR پرتفولیو (۹۵٪)', value: '۱.۲٪', status: 'ok' as const },
  { label: 'حداکثر افت', value: '۴.۱٪', status: 'ok' as const },
  { label: 'سقف موقعیت', value: 'در محدوده', status: 'ok' as const },
  { label: 'سقف زیان روزانه', value: '۰.۸٪ مصرف‌شده', status: 'warn' as const },
];

export function RiskAgentPanel() {
  return (
    <AgentPanel
      title="مدیریت ریسک (M6)"
      subtitle="VaR، سقف‌ها و انطباق"
      icon={<Shield className="h-4 w-4 text-primary" />}
      agentId="M6"
    >
      <ul className="space-y-2 pr-2">
        {MOCK_RISK.map((r) => (
          <li
            key={r.label}
            className={cn(
              'rounded-lg border p-2.5 text-xs flex items-center justify-between gap-2',
              r.status === 'warn' && 'border-amber-500/20 bg-amber-500/5',
              r.status === 'ok' && 'border-border bg-muted/20'
            )}
          >
            <span className="text-muted-foreground">{r.label}</span>
            <span className="flex items-center gap-1 font-medium tabular-nums">
              {r.status === 'warn' && <AlertTriangle className="h-3 w-3 text-amber-500" />}
              {r.status === 'ok' && <CheckCircle className="h-3 w-3 text-emerald-500" />}
              {r.value}
            </span>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
