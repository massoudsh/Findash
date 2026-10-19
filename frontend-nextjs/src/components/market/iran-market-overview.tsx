'use client';

import { getBackendUrl } from '@/lib/backend-url';
import { useEffect, useState, useCallback } from 'react';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';

interface MarketItem {
  currency: 'IRR' | 'IRT';
  symbol: string;
  label: string;
  icon: string;
  category: string;
  price: number | null;
  change_pct: number | null;
  up: boolean | null;
  available: boolean;
}

const API_URL = getBackendUrl();

const CATEGORY_LABELS: Record<string, string> = {
  currency: 'ارز',
  gold: 'طلا',
  coin: 'سکه',
  crypto: 'کریپتو',
  bourse: 'بورس',
};


function formatPrice(price: number | null, currency: 'IRR' | 'IRT'): string {
  if (price === null) return '—';
  return new Intl.NumberFormat('fa-IR', { maximumFractionDigits: 0 }).format(price) + (currency === 'IRR' ? ' ریال' : ' تومان');
}

function PriceCard({ item }: { item: MarketItem }) {
  const isUp = item.up;
  return (
    <div className="persian-card rounded-2xl p-4 flex flex-col gap-2 border border-border/40 hover:border-green-500/20 transition-colors">
      <div className="flex items-center justify-between">
        <span className="text-lg">{item.icon}</span>
        <span className="text-[10px] text-muted-foreground bg-muted/50 rounded-full px-2 py-0.5">
          {CATEGORY_LABELS[item.category] ?? item.category}
        </span>
      </div>
      <div className="text-sm font-semibold text-foreground">{item.label}</div>
      {item.available ? (
        <>
          <div className="text-base font-black tabular-nums" dir="ltr">
            {formatPrice(item.price, item.currency)}
          </div>
          <div
            className={`flex items-center gap-1 text-xs font-medium ${
              isUp ? 'text-green-400' : 'text-red-400'
            }`}
            dir="ltr"
          >
            {isUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
            {item.change_pct !== null
              ? `${isUp ? '+' : ''}${item.change_pct.toFixed(2)}٪`
              : '—'}
          </div>
        </>
      ) : (
        <div className="text-muted-foreground text-sm">در دسترس نیست</div>
      )}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="persian-card rounded-2xl p-4 flex flex-col gap-2 border border-border/40 animate-pulse">
      <div className="h-5 w-8 bg-muted/40 rounded" />
      <div className="h-4 w-24 bg-muted/40 rounded" />
      <div className="h-5 w-32 bg-muted/40 rounded" />
      <div className="h-3 w-16 bg-muted/40 rounded" />
    </div>
  );
}

export function IranMarketOverview() {
  const [items, setItems] = useState<MarketItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>('all');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/iran-market/overview`, { cache: 'no-store', signal: AbortSignal.timeout(15000) });
      if (!res.ok) throw new Error('دریافت قیمت‌ها ناموفق بود.');
      const data = await res.json();
      if (!Array.isArray(data.items) || !data.items.some((item: MarketItem) => item.available)) {
        throw new Error('منابع بازار در دسترس نیستند.');
      }
      setItems(data.items);
      setLastUpdated(new Date(data.cached_at * 1000));
      setError('');
    } catch {
      setError('ارتباط با بازار برقرار نشد. قیمت‌های قبلی، در صورت وجود، قدیمی هستند.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 30_000);
    return () => clearInterval(id);
  }, [fetchData]);

  const categories = ['all', 'currency', 'gold', 'coin', 'crypto', 'bourse'];
  const filtered = activeCategory === 'all'
    ? items
    : items.filter((i) => i.category === activeCategory);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-black">بازار ایران</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            {lastUpdated
              ? `آخرین بروزرسانی: ${lastUpdated.toLocaleTimeString('fa-IR')}`
              : loading ? 'در حال دریافت داده…' : 'هنوز قیمت معتبری دریافت نشده'}
          </p>
        </div>
        <button
          disabled={loading}
          onClick={fetchData}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-green-400 transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          بروزرسانی
        </button>
      </div>

      {error && <p role="alert" className="text-sm text-amber-500">{error}</p>}
      {/* Category filter */}
      <div className="flex gap-2 overflow-x-auto scrollbar-none pb-1">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`shrink-0 text-xs px-3 py-1.5 rounded-full border transition-all ${
              activeCategory === cat
                ? 'bg-green-500/15 border-green-500/40 text-green-400 font-semibold'
                : 'border-border/50 text-muted-foreground hover:border-green-500/20'
            }`}
          >
            {cat === 'all' ? 'همه' : (CATEGORY_LABELS[cat] ?? cat)}
          </button>
        ))}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {loading
          ? Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} />)
          : filtered.map((item) => <PriceCard key={item.symbol} item={item} />)}
      </div>

      {!loading && filtered.length === 0 && (
        <div className="text-center text-muted-foreground text-sm py-8">
          هیچ داده‌ای یافت نشد
        </div>
      )}
    </div>
  );
}
