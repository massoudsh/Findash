'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Target,
  Brain,
  TrendingUp,
  Database,
  MessageSquare,
  Cpu,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';

type InsightSource =
  | 'technical'
  | 'fundamental'
  | 'macro'
  | 'on-chain'
  | 'social'
  | 'ai-models';

interface Insight {
  id: string;
  source: InsightSource;
  title: string;
  summary: string;
  signal: 'bullish' | 'bearish' | 'neutral';
  timestamp: string;
  symbol?: string;
}

const SOURCE_CONFIG: Record<
  InsightSource,
  { label: string; icon: React.ComponentType<{ className?: string }>; color: string }
> = {
  technical: { label: 'Technical', icon: Target, color: 'text-blue-500' },
  fundamental: { label: 'Fundamental', icon: Brain, color: 'text-emerald-500' },
  macro: { label: 'Macro', icon: TrendingUp, color: 'text-amber-500' },
  'on-chain': { label: 'On-chain', icon: Database, color: 'text-violet-500' },
  social: { label: 'Social', icon: MessageSquare, color: 'text-pink-500' },
  'ai-models': { label: 'AI Models', icon: Cpu, color: 'text-cyan-500' },
};

// Mock real-time insights stream (replace with API/SSE later)
const MOCK_INSIGHTS: Omit<Insight, 'id' | 'timestamp'>[] = [
  { source: 'technical', title: 'RSI divergence', summary: 'AAPL 4H RSI divergence suggests near-term pullback.', signal: 'bearish', symbol: 'AAPL' },
  { source: 'fundamental', title: 'Earnings beat', summary: 'NVDA forward P/E supportive; institutional flow positive.', signal: 'bullish', symbol: 'NVDA' },
  { source: 'macro', title: 'Rates hold', summary: 'Fed hold priced in; DXY weakness supports risk assets.', signal: 'bullish' },
  { source: 'on-chain', title: 'Whale accumulation', summary: 'Large ETH wallets adding; exchange outflow rising.', signal: 'bullish', symbol: 'ETH' },
  { source: 'social', title: 'Sentiment shift', summary: 'Twitter/X sentiment for BTC turned positive (7d).', signal: 'bullish', symbol: 'BTC' },
  { source: 'ai-models', title: 'Regime: risk-on', summary: 'Ensemble model assigns 0.78 to risk-on regime next 5d.', signal: 'bullish' },
  { source: 'technical', title: 'Support level', summary: 'SPY holding above 580; volume declining on pullbacks.', signal: 'neutral', symbol: 'SPY' },
  { source: 'fundamental', title: 'Sector rotation', summary: 'Fund flows into tech; defensives outflow.', signal: 'bullish' },
  { source: 'macro', title: 'CPI in focus', summary: 'Next CPI print key for rate path; vol elevated.', signal: 'neutral' },
  { source: 'on-chain', title: 'Options flow', summary: 'Unusual call buying in TSLA; OI up 12%.', signal: 'bullish', symbol: 'TSLA' },
  { source: 'social', title: 'Earnings chatter', summary: 'Mentions for AAPL up ahead of event.', signal: 'neutral', symbol: 'AAPL' },
  { source: 'ai-models', title: 'Vol forecast', summary: 'VIX term structure implies lower vol in 2w.', signal: 'neutral' },
];

function generateTimestamp(offsetSeconds: number): string {
  const d = new Date(Date.now() - offsetSeconds * 1000);
  const now = new Date();
  const diff = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

export function AnalysisAgentInsightsPanel() {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const base: Insight[] = MOCK_INSIGHTS.slice(0, 8).map((m, i) => ({
      ...m,
      id: `insight-${i}-${tick}`,
      timestamp: generateTimestamp(i * 120 + tick * 30),
    }));
    setInsights(base);
  }, [tick]);

  useEffect(() => {
    const t = setInterval(() => setTick((k) => k + 1), 15000);
    return () => clearInterval(t);
  }, []);

  return (
    <Card className="flex flex-col h-full min-h-0 border-l bg-card/95">
      <CardHeader className="shrink-0 py-3 px-4 border-b">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            Analysis Agent
          </CardTitle>
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded hover:bg-muted"
            aria-label={collapsed ? 'Expand' : 'Collapse'}
          >
            {collapsed ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">
          Real-time insight from Technical, Fundamental, Macro, On-chain, Social & AI Models
        </p>
      </CardHeader>
      {!collapsed && (
        <CardContent className="p-0 flex-1 min-h-0 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto px-3 py-2">
            <ul className="space-y-2 pr-2">
              {insights.map((insight) => {
                const config = SOURCE_CONFIG[insight.source];
                const Icon = config.icon;
                return (
                  <li
                    key={insight.id}
                    className={cn(
                      'rounded-lg border p-2.5 text-xs transition-colors hover:bg-muted/50',
                      insight.signal === 'bullish' && 'border-green-500/20 bg-green-500/5',
                      insight.signal === 'bearish' && 'border-red-500/20 bg-red-500/5',
                      insight.signal === 'neutral' && 'border-border bg-muted/20'
                    )}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className={cn('font-medium flex items-center gap-1', config.color)}>
                        <Icon className="h-3 w-3 shrink-0" />
                        {config.label}
                      </span>
                      <div className="flex items-center gap-1">
                        {insight.symbol && (
                          <Badge variant="outline" className="text-[10px] px-1 py-0">
                            {insight.symbol}
                          </Badge>
                        )}
                        <span className="text-muted-foreground">{insight.timestamp}</span>
                      </div>
                    </div>
                    <p className="font-medium text-foreground/90">{insight.title}</p>
                    <p className="text-muted-foreground mt-0.5 leading-snug">{insight.summary}</p>
                  </li>
                );
              })}
            </ul>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
