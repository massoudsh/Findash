'use client';

import { useEffect, useState } from 'react';
import { Newspaper } from 'lucide-react';
import { cn } from '@/lib/utils';
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

  if (news.length === 0) return null;

  const items = [...news, ...news]; // duplicated for seamless loop

  return (
    <div className="flex items-center gap-2 min-w-0 overflow-hidden h-8 border-b border-border/40 bg-card/60 px-3">
      <Newspaper className="h-3.5 w-3.5 text-green-400 shrink-0" />
      <div className="relative flex-1 min-w-0 overflow-hidden">
        <div className="flex w-max whitespace-nowrap animate-news-marquee gap-8">
          {items.map((item, i) => (
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
          ))}
        </div>
      </div>
    </div>
  );
}
