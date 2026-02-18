'use client';

import { useState, useEffect, useCallback } from 'react';
import { AgentPanel } from './agent-panel';
import { Badge } from '@/components/ui/badge';
import { Target, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M4 Strategy agent: active signals and bot feed (aligned with findash-agents M4). */
interface StrategySignal {
  id: string;
  strategy: string;
  symbol: string;
  side: 'long' | 'short';
  strength: number;
  timestamp: string;
}

const MOCK_SIGNALS: StrategySignal[] = [
  { id: '1', strategy: 'momentum', symbol: 'NVDA', side: 'long', strength: 0.82, timestamp: '2m ago' },
  { id: '2', strategy: 'mean_reversion', symbol: 'SPY', side: 'long', strength: 0.61, timestamp: '5m ago' },
  { id: '3', strategy: 'trend_following', symbol: 'ETH-USD', side: 'short', strength: 0.55, timestamp: '8m ago' },
  { id: '4', strategy: 'momentum', symbol: 'AAPL', side: 'long', strength: 0.71, timestamp: '12m ago' },
];

const API_BASE = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || '' : '';

export function StrategyAgentPanel() {
  const [signals, setSignals] = useState<StrategySignal[]>(MOCK_SIGNALS);

  const fetchSignals = useCallback(async () => {
    if (!API_BASE) return;
    try {
      const res = await fetch(`${API_BASE}/api/agent-panels/strategy-signals`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data?.signals)) setSignals(data.signals);
      }
    } catch {
      // keep mock
    }
  }, []);

  useEffect(() => {
    fetchSignals();
    const t = setInterval(fetchSignals, 30000);
    return () => clearInterval(t);
  }, [fetchSignals]);

  return (
    <AgentPanel
      title="Strategy Agent (M4)"
      subtitle="Active signals and execution feed"
      icon={<Target className="h-4 w-4 text-primary" />}
      agentId="M4"
    >
      <ul className="space-y-2 pr-2">
        {signals.map((s) => (
          <li
            key={s.id}
            className={cn(
              'rounded-lg border p-2.5 text-xs transition-colors hover:bg-muted/50',
              s.side === 'long' && 'border-green-500/20 bg-green-500/5',
              s.side === 'short' && 'border-red-500/20 bg-red-500/5'
            )}
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <span className="font-medium">{s.symbol}</span>
              <Badge variant="outline" className="text-[10px] capitalize">
                {s.strategy.replace('_', ' ')}
              </Badge>
            </div>
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="flex items-center gap-1">
                {s.side === 'long' ? (
                  <TrendingUp className="h-3 w-3 text-green-500" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-red-500" />
                )}
                {s.side.toUpperCase()}
              </span>
              <span>{(s.strength * 100).toFixed(0)}% · {s.timestamp}</span>
            </div>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
