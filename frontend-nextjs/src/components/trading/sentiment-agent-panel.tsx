'use client';

import { useState, useEffect, useCallback } from 'react';
import { AgentPanel } from './agent-panel';
import { MessageSquare, Smile, Frown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

/** M9 Sentiment Analyzer agent: sentiment summary (aligned with findash-agents M9). */
interface SentimentBucket {
  symbol: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  score: number;
  source: string;
}

const MOCK_SENTIMENT: SentimentBucket[] = [
  { symbol: 'BTC-USD', sentiment: 'positive', score: 0.72, source: 'social' },
  { symbol: 'NVDA', sentiment: 'positive', score: 0.68, source: 'news' },
  { symbol: 'TSLA', sentiment: 'neutral', score: 0.48, source: 'social' },
  { symbol: 'AAPL', sentiment: 'positive', score: 0.61, source: 'news' },
];

const API_BASE = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || '' : '';

export function SentimentAgentPanel() {
  const [items, setItems] = useState<SentimentBucket[]>(MOCK_SENTIMENT);

  const fetchSentiment = useCallback(async () => {
    if (!API_BASE) return;
    try {
      const res = await fetch(`${API_BASE}/api/agent-panels/sentiment`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data?.items)) setItems(data.items);
      }
    } catch {
      // keep mock
    }
  }, []);

  useEffect(() => {
    fetchSentiment();
    const t = setInterval(fetchSentiment, 30000);
    return () => clearInterval(t);
  }, [fetchSentiment]);

  return (
    <AgentPanel
      title="عامل سنتیمنت (M9)"
      subtitle="سنتیمنت اخبار و شبکه‌های اجتماعی"
      icon={<MessageSquare className="h-4 w-4 text-primary" />}
      agentId="M9"
    >
      <ul className="space-y-2 pr-2">
        {items.map((item, i) => (
          <li
            key={`${item.symbol}-${i}`}
            className={cn(
              'rounded-lg border p-2.5 text-xs transition-colors',
              item.sentiment === 'positive' && 'border-green-500/20 bg-green-500/5',
              item.sentiment === 'negative' && 'border-red-500/20 bg-red-500/5',
              item.sentiment === 'neutral' && 'border-border bg-muted/20'
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium">{item.symbol}</span>
              {item.sentiment === 'positive' ? (
                <Smile className="h-3 w-3 text-green-500" />
              ) : item.sentiment === 'negative' ? (
                <Frown className="h-3 w-3 text-red-500" />
              ) : (
                <Minus className="h-3 w-3 text-muted-foreground" />
              )}
            </div>
            <div className="flex justify-between mt-1 text-muted-foreground">
              <span>{(item.score * 100).toFixed(0)}%</span>
              <span className="capitalize">{item.source}</span>
            </div>
          </li>
        ))}
      </ul>
    </AgentPanel>
  );
}
