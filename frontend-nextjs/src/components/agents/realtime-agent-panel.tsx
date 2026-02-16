'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { Activity, Wifi, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M3 Real-time Processing Agent: stream health and latency */
const MOCK_STREAMS = [
  { name: 'Market quotes', latency: '12 ms', status: 'active' as const },
  { name: 'Order book', latency: '18 ms', status: 'active' as const },
  { name: 'Trades', latency: '8 ms', status: 'active' as const },
  { name: 'Alerts', latency: '—', status: 'idle' as const },
];

export function RealtimeAgentPanel() {
  return (
    <AgentPanel
      title="Real-time Processor (M3)"
      subtitle="Stream health and latency"
      icon={<Activity className="h-4 w-4 text-primary" />}
      agentId="M3"
    >
      <ul className="space-y-2 pr-2">
        {MOCK_STREAMS.map((s) => (
          <li
            key={s.name}
            className={cn(
              'rounded-lg border p-2.5 text-xs flex items-center justify-between gap-2',
              s.status === 'active' && 'border-emerald-500/20 bg-emerald-500/5',
              s.status === 'idle' && 'border-border bg-muted/20'
            )}
          >
            <span className="flex items-center gap-1.5 font-medium">
              {s.status === 'active' ? (
                <Wifi className="h-3 w-3 text-emerald-500" />
              ) : (
                <Zap className="h-3 w-3 text-muted-foreground" />
              )}
              {s.name}
            </span>
            <span className="text-muted-foreground tabular-nums">{s.latency}</span>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
