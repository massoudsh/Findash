'use client';

import { useEffect, useState } from 'react';
import { Newspaper, TrendingDown, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useIranTicker, type TickerItem } from '@/lib/hooks/use-iran-ticker';
import type { NewsItem } from '@/app/api/news/route';

const CATEGORY_COLORS: Record<string, string> = {
  gold: 'text-yellow-400',
  currency: 'text-blue-400',
  stock: 'text-green-400',
  crypto: 'text-purple-400',
  macro: 'text-red-400',
  general: 'text-muted-foreground',
};

export function NewsTicker() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const { items: marketItems } = useIranTicker();

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch('/api/news?category=all');
        const json = await res.json();
        if (!cancelled) setNews(json.data ?? []);
      } catch {
        if (!cancelled) setNews([]);
      }
    };
    load();
    const interval = setInterval(load, 120000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (news.length === 0 && marketItems.length === 0) return null;

  const items = [...news];
  const prices = marketItems.filter((item) => item.available && item.price !== null);
  const baseEntries: Array<{ type: 'news'; item: NewsItem } | { type: 'price'; item: TickerItem }> = [
    ...prices.map((item) => ({ type: 'price' as const, item })),
    ...items.map((item) => ({ type: 'news' as const, item })),
  ];
  const tickerEntries = [...baseEntries, ...baseEntries];

  return (
    <div className="flex items-center gap-2 min-w-0 overflow-hidden h-9 border-b border-border/40 bg-card/60 px-3">
      <Newspaper className="h-3.5 w-3.5 text-green-400 shrink-0" />
      <div className="relative flex-1 min-w-0 overflow-hidden">
        <div className="flex w-max whitespace-nowrap animate-news-marquee gap-8">
          {tickerEntries.map((entry, i) => {
            if (entry.type === 'price') {
              const item = entry.item;
              const change = item.change_pct ?? 0;
              return (
                <span key={`${item.symbol}-${i}`} className="text-xs flex items-center gap-1.5" dir="rtl">
                  <span className="font-semibold text-amber-400">{item.icon} {item.label}</span>
                  <span className="font-bold text-foreground" dir="ltr">{new Intl.NumberFormat('fa-IR').format(Math.round(item.price ?? 0))} ریال</span>
                  <span className={cn('flex items-center gap-0.5 font-bold', change >= 0 ? 'text-emerald-400' : 'text-rose-400')} dir="ltr">
                    {change >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                    {change >= 0 ? '+' : ''}{change.toFixed(1)}٪
                  </span>
                </span>
              );
            }
            const item = entry.item;
            return (
              <a
                key={`${item.id}-${i}`}
                href={item.url || undefined}
                target="_blank"
                rel="noopener noreferrer"
                dir={item.lang === 'fa' ? 'rtl' : 'ltr'}
                className="text-xs flex items-center gap-1.5 hover:text-primary transition-colors"
              >
                <span className={cn('font-semibold shrink-0', CATEGORY_COLORS[item.category])}>
                  {item.source}:
                </span>
                <span className="text-muted-foreground">{item.title}</span>
              </a>
            );
          })}
        </div>
      </div>
    </div>
  );
}
