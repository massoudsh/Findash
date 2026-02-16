'use client';

import { useState } from 'react';
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

export function SentimentAgentPanel() {
  const [items] = useState<SentimentBucket[]>(MOCK_SENTIMENT);

  return (
    <AgentPanel
      title="Sentiment Agent (M9)"
      subtitle="News and social sentiment by asset"
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
