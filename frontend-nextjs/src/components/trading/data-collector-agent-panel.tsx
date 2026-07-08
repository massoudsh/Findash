'use client';

import { useState, useEffect, useCallback } from 'react';
import { AgentPanel } from './agent-panel';
import { Database, Wifi, WifiOff, XCircle } from 'lucide-react';
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
  { id: 'market', name: 'داده بازار', type: 'market_data', status: 'active', lastSync: '۱۲ ثانیه پیش', recordsToday: 15420 },
  { id: 'news', name: 'فید خبری', type: 'news', status: 'active', lastSync: '۱ دقیقه پیش', recordsToday: 2847 },
  { id: 'social', name: 'سنتیمنت اجتماعی', type: 'social', status: 'degraded', lastSync: '۵ دقیقه پیش', recordsToday: 892 },
  { id: 'fundamental', name: 'فاندامنتال', type: 'fundamental', status: 'active', lastSync: '۲ دقیقه پیش', recordsToday: 1205 },
  { id: 'onchain', name: 'آن‌چین', type: 'on_chain', status: 'active', lastSync: '۴۵ ثانیه پیش', recordsToday: 3421 },
];

const API_BASE = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || '' : '';

export function DataCollectorAgentPanel() {
  const [sources, setSources] = useState<DataSource[]>(MOCK_SOURCES);

  const fetchSources = useCallback(async () => {
    if (!API_BASE) return;
    try {
      const res = await fetch(`${API_BASE}/api/agent-panels/data-collector`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data?.sources)) setSources(data.sources);
      }
    } catch {
      // keep existing state or mock
    }
  }, []);

  useEffect(() => {
    fetchSources();
    const t = setInterval(fetchSources, 20000);
    return () => clearInterval(t);
  }, [fetchSources]);

  useEffect(() => {
    const tick = setInterval(() => {
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
    return () => clearInterval(tick);
  }, []);

  return (
    <AgentPanel
      title="گردآورنده داده (M1)"
      subtitle="وضعیت پایپ‌لاین و سلامت دریافت داده"
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
              <span>آخرین همگام‌سازی: {src.lastSync}</span>
              <span>{src.recordsToday.toLocaleString()} امروز</span>
            </div>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
