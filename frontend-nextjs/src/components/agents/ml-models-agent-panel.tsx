'use client';

import { AgentPanel } from '@/components/trading/agent-panel';
import { Brain, Cpu, CheckCircle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M5 ML Models Agent: model status and prediction feed */
const MOCK_MODELS = [
  { name: 'TCN Price', status: 'active' as const },
  { name: 'Ensemble Regime', status: 'active' as const },
  { name: 'Sentiment CNN', status: 'training' as const },
];

export function MLModelsAgentPanel() {
  return (
    <AgentPanel
      title="ML Models (M5)"
      subtitle="Deep learning and prediction models"
      icon={<Brain className="h-4 w-4 text-primary" />}
      agentId="M5"
    >
      <ul className="space-y-2 pr-2">
        {MOCK_MODELS.map((r) => (
          <li
            key={r.name}
            className={cn(
              'rounded-lg border p-2.5 text-xs flex items-center justify-between gap-2',
              r.status === 'active' && 'border-emerald-500/20 bg-emerald-500/5',
              r.status === 'training' && 'border-amber-500/20 bg-amber-500/5'
            )}
          >
            <span className="font-medium flex items-center gap-1.5">
              <Cpu className="h-3 w-3 text-muted-foreground" />
              {r.name}
            </span>
            {r.status === 'active' ? (
              <CheckCircle className="h-3 w-3 shrink-0 text-emerald-500" />
            ) : (
              <Clock className="h-3 w-3 shrink-0 text-amber-500" />
            )}
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
