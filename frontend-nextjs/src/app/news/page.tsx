'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Newspaper, RefreshCw, ExternalLink, Clock } from 'lucide-react';
import type { NewsItem } from '@/app/api/news/route';

const CATEGORIES = [
  { value: 'all', label: 'همه' },
  { value: 'gold', label: 'طلا & سکه' },
  { value: 'currency', label: 'ارز' },
  { value: 'stock', label: 'بورس' },
  { value: 'crypto', label: 'کریپتو' },
  { value: 'macro', label: 'کلان' },
  { value: 'general', label: 'عمومی' },
];

const CATEGORY_COLORS: Record<string, string> = {
  gold: 'bg-yellow-500/10 text-yellow-700 dark:text-yellow-400',
  currency: 'bg-blue-500/10 text-blue-700 dark:text-blue-400',
  stock: 'bg-green-500/10 text-green-700 dark:text-green-400',
  crypto: 'bg-purple-500/10 text-purple-700 dark:text-purple-400',
  macro: 'bg-red-500/10 text-red-700 dark:text-red-400',
  general: 'bg-gray-500/10 text-gray-700 dark:text-gray-400',
};

function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function NewsPage() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [category, setCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [dataStatus, setDataStatus] = useState<'live' | 'mock' | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetch_news = useCallback(async (cat: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/news?category=${cat}`);
      const json = await res.json();
      setNews(json.data ?? []);
      setDataStatus(json.status);
    } catch {
      setNews([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetch_news(category);
  }, [category, fetch_news]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetch_news(category);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Newspaper className="h-7 w-7" />
            Iran Market News
          </h1>
          <p className="text-muted-foreground mt-1">
            اخبار بازارهای مالی ایران — طلا، ارز، بورس، کریپتو
          </p>
        </div>
        <div className="flex items-center gap-2">
          {dataStatus && (
            <Badge variant={dataStatus === 'live' ? 'default' : 'secondary'}>
              {dataStatus === 'live' ? '🔴 Live' : '📦 Sample'}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={cn('h-4 w-4 mr-1', refreshing && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Category Tabs */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((cat) => (
          <Button
            key={cat.value}
            variant={category === cat.value ? 'default' : 'outline'}
            size="sm"
            onClick={() => setCategory(cat.value)}
            className="font-medium"
          >
            {cat.label}
          </Button>
        ))}
      </div>

      {/* News Grid */}
      {loading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="pt-4 space-y-2">
                <div className="h-4 bg-muted rounded w-3/4" />
                <div className="h-3 bg-muted rounded w-1/2" />
                <div className="h-3 bg-muted rounded w-full" />
                <div className="h-3 bg-muted rounded w-5/6" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : news.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">No news found.</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {news.map((item) => (
            <Card key={item.id} className="hover:border-primary/50 transition-colors group">
              <CardContent className="pt-4">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={cn('text-xs px-2 py-0.5 rounded-full font-medium', CATEGORY_COLORS[item.category])}>
                      {CATEGORIES.find((c) => c.value === item.category)?.label ?? item.category}
                    </span>
                    <span className="text-xs text-muted-foreground">{item.source}</span>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                    <Clock className="h-3 w-3" />
                    {timeAgo(item.publishedAt)}
                  </div>
                </div>

                <h3
                  dir={item.lang === 'fa' ? 'rtl' : 'ltr'}
                  className={cn(
                    'font-semibold text-sm leading-snug mb-2',
                    item.lang === 'fa' && 'font-[Vazirmatn] text-right'
                  )}
                >
                  {item.title}
                </h3>

                {item.description && (
                  <p
                    dir={item.lang === 'fa' ? 'rtl' : 'ltr'}
                    className={cn(
                      'text-xs text-muted-foreground line-clamp-3',
                      item.lang === 'fa' && 'font-[Vazirmatn] text-right'
                    )}
                  >
                    {item.description}
                  </p>
                )}

                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-3 inline-flex items-center gap-1 text-xs text-primary opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    Read more <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
