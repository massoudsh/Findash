'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { Shield, AlertTriangle, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M6 Risk Management Agent: risk metrics and compliance status */
const MOCK_RISK = [
  { label: 'Portfolio VaR (95%)', value: '1.2%', status: 'ok' as const },
  { label: 'Max drawdown', value: '4.1%', status: 'ok' as const },
  { label: 'Position limits', value: 'Within', status: 'ok' as const },
  { label: 'Daily loss limit', value: '0.8% used', status: 'warn' as const },
];

export function RiskAgentPanel() {
  return (
    <AgentPanel
      title="Risk Management (M6)"
      subtitle="VaR, limits, and compliance"
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
