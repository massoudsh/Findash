'use client';

import { useState, useEffect } from 'react';
import { AgentPanel } from './agent-panel';
import { Database, Wifi, WifiOff, CheckCircle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M1 Data Collector agent: data sources status and pipeline health (aligned with findash-data-collector). */
interface DataSource {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'degraded' | 'error';
  lastSync: string;
  recordsToday: number;
}

const MOCK_SOURCES: DataSource[] = [
  { id: 'market', name: 'Market Data', type: 'market_data', status: 'active', lastSync: '12s ago', recordsToday: 15420 },
  { id: 'news', name: 'News Feed', type: 'news', status: 'active', lastSync: '1m ago', recordsToday: 2847 },
  { id: 'social', name: 'Social Sentiment', type: 'social', status: 'degraded', lastSync: '5m ago', recordsToday: 892 },
  { id: 'fundamental', name: 'Fundamental', type: 'fundamental', status: 'active', lastSync: '2m ago', recordsToday: 1205 },
  { id: 'onchain', name: 'On-chain', type: 'on_chain', status: 'active', lastSync: '45s ago', recordsToday: 3421 },
];

export function DataCollectorAgentPanel() {
  const [sources, setSources] = useState<DataSource[]>(MOCK_SOURCES);

  useEffect(() => {
    const t = setInterval(() => {
      setSources((prev) =>
        prev.map((s) => ({
          ...s,
          lastSync:
            s.status === 'active'
              ? `${Math.max(5, Math.floor(Math.random() * 60))}s ago`
              : s.lastSync,
        }))
      );
    }, 20000);
    return () => clearInterval(t);
  }, []);

  return (
    <AgentPanel
      title="Data Collector (M1)"
      subtitle="Pipeline status and ingestion health"
      icon={<Database className="h-4 w-4 text-primary" />}
      agentId="M1"
    >
      <ul className="space-y-2 pr-2">
        {sources.map((src) => (
          <li
            key={src.id}
            className={cn(
              'rounded-lg border p-2.5 text-xs transition-colors',
              src.status === 'active' && 'border-green-500/20 bg-green-500/5',
              src.status === 'degraded' && 'border-amber-500/20 bg-amber-500/5',
              src.status === 'error' && 'border-red-500/20 bg-red-500/5'
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium">{src.name}</span>
              {src.status === 'active' ? (
                <Wifi className="h-3 w-3 text-green-500" />
              ) : src.status === 'error' ? (
                <XCircle className="h-3 w-3 text-red-500" />
              ) : (
                <WifiOff className="h-3 w-3 text-amber-500" />
              )}
            </div>
            <div className="flex justify-between mt-1 text-muted-foreground">
              <span>Last sync: {src.lastSync}</span>
              <span>{src.recordsToday.toLocaleString()} today</span>
            </div>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
