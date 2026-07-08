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
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { AgentCharacter } from '@/components/agents/agent-character';

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
  technical: { label: 'تکنیکال', icon: Target, color: 'text-blue-500' },
  fundamental: { label: 'بنیادی', icon: Brain, color: 'text-emerald-500' },
  macro: { label: 'کلان', icon: TrendingUp, color: 'text-amber-500' },
  'on-chain': { label: 'آن‌چین', icon: Database, color: 'text-violet-500' },
  social: { label: 'اجتماعی', icon: MessageSquare, color: 'text-pink-500' },
  'ai-models': { label: 'مدل‌های هوش مصنوعی', icon: Cpu, color: 'text-cyan-500' },
};

// Mock real-time insights stream (fallback when API unavailable)
const MOCK_INSIGHTS: Omit<Insight, 'id' | 'timestamp'>[] = [
  { source: 'technical', title: 'واگرایی RSI', summary: 'واگرایی RSI در تایم‌فریم ۴ساعته AAPL احتمال اصلاح کوتاه‌مدت را نشان می‌دهد.', signal: 'bearish', symbol: 'AAPL' },
  { source: 'fundamental', title: 'سود بهتر از انتظار', summary: 'P/E آینده NVDA حمایتی است؛ جریان سرمایه نهادی مثبت.', signal: 'bullish', symbol: 'NVDA' },
  { source: 'macro', title: 'ثبات نرخ بهره', summary: 'توقف فد در قیمت‌ها لحاظ شده؛ ضعف DXY از دارایی‌های پرریسک حمایت می‌کند.', signal: 'bullish' },
  { source: 'on-chain', title: 'تجمیع نهنگ‌ها', summary: 'کیف‌پول‌های بزرگ ETH در حال خرید هستند؛ خروج از صرافی‌ها افزایش یافته.', signal: 'bullish', symbol: 'ETH' },
  { source: 'social', title: 'چرخش سنتیمنت', summary: 'سنتیمنت توییتر/X برای BTC در ۷ روز گذشته مثبت شده.', signal: 'bullish', symbol: 'BTC' },
  { source: 'ai-models', title: 'رژیم: ریسک‌پذیر', summary: 'مدل Ensemble احتمال ۰.۷۸ برای رژیم ریسک‌پذیر در ۵ روز آینده تخمین می‌زند.', signal: 'bullish' },
  { source: 'technical', title: 'سطح حمایت', summary: 'SPY بالای ۵۸۰ تثبیت شده؛ حجم در اصلاح‌ها کاهشی است.', signal: 'neutral', symbol: 'SPY' },
  { source: 'fundamental', title: 'چرخش سکتور', summary: 'جریان سرمایه به تکنولوژی؛ خروج از سکتورهای دفاعی.', signal: 'bullish' },
  { source: 'macro', title: 'CPI در کانون توجه', summary: 'گزارش بعدی CPI برای مسیر نرخ بهره کلیدی است؛ نوسانات بالا.', signal: 'neutral' },
  { source: 'on-chain', title: 'جریان اختیارات', summary: 'خرید call غیرعادی در TSLA؛ OI 12% افزایش یافته.', signal: 'bullish', symbol: 'TSLA' },
  { source: 'social', title: 'بحث درباره سود', summary: 'اشاره‌ها به AAPL پیش از رویداد افزایش یافته.', signal: 'neutral', symbol: 'AAPL' },
  { source: 'ai-models', title: 'پیش‌بینی نوسان', summary: 'ساختار ترم VIX نوسان کمتر در ۲ هفته آینده را نشان می‌دهد.', signal: 'neutral' },
];

function generateTimestamp(offsetSeconds: number): string {
  const d = new Date(Date.now() - offsetSeconds * 1000);
  const now = new Date();
  const diff = Math.floor((now.getTime() - d.getTime()) / 1000);
  if (diff < 60) return 'همین الان';
  if (diff < 3600) return `${Math.floor(diff / 60)} دقیقه پیش`;
  return `${Math.floor(diff / 3600)} ساعت پیش`;
}

const API_BASE = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL || '' : '';

export function AnalysisAgentInsightsPanel() {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    async function fetchInsights() {
      if (API_BASE) {
        try {
          const res = await fetch(`${API_BASE}/api/agent-panels/insights`, { credentials: 'include' });
          if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data?.insights) && data.insights.length > 0) {
              setInsights(
                data.insights.map((x: Record<string, unknown>) => ({
                  id: String(x.id ?? `insight-${tick}-${Math.random()}`),
                  source: x.source as InsightSource,
                  title: String(x.title ?? ''),
                  summary: String(x.summary ?? ''),
                  signal: (x.signal as 'bullish' | 'bearish' | 'neutral') ?? 'neutral',
                  timestamp: String(x.timestamp ?? ''),
                  symbol: x.symbol != null ? String(x.symbol) : undefined,
                }))
              );
              return;
            }
          }
        } catch {
          // fallback below
        }
      }
      const base: Insight[] = MOCK_INSIGHTS.slice(0, 8).map((m, i) => ({
        ...m,
        id: `insight-${i}-${tick}`,
        timestamp: generateTimestamp(i * 120 + tick * 30),
      }));
      setInsights(base);
    }
    fetchInsights();
  }, [tick]);

  useEffect(() => {
    const t = setInterval(() => setTick((k) => k + 1), 15000);
    return () => clearInterval(t);
  }, []);

  return (
    <Card className="flex flex-col h-full min-h-0 border-l bg-card/95">
      <CardHeader className="shrink-0 py-3 px-4 border-b">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-2 min-w-0">
            <AgentCharacter agentId="M11" variant="inline" size="sm" />
          </CardTitle>
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded hover:bg-muted"
            aria-label={collapsed ? 'بازکردن' : 'بستن'}
          >
            {collapsed ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">
          بینش بلادرنگ از تکنیکال، بنیادی، کلان، آن‌چین، اجتماعی و هوش مصنوعی
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
