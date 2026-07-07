'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Maximize2, Minimize2, Download } from 'lucide-react';
import { cn } from '@/lib/utils';

/* ─── Types ─── */
export interface MapItem {
  symbol: string;
  name: string;
  price: number;
  changePercent: number;
  volume: number;
  marketCap?: number;
  category: string;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface LayoutItem extends MapItem {
  rect: Rect;
  size: number;
}

/* ─── Treemap layout (slice-and-dice binary split) ─── */
function layout(
  items: (MapItem & { size: number })[],
  x: number,
  y: number,
  w: number,
  h: number,
  horizontal: boolean,
): LayoutItem[] {
  if (items.length === 0) return [];
  if (items.length === 1) return [{ ...items[0], rect: { x, y, w, h } }];

  const total = items.reduce((s, i) => s + i.size, 0);
  let acc = 0;
  let splitIdx = 0;
  for (let i = 0; i < items.length; i++) {
    acc += items[i].size;
    splitIdx = i;
    if (acc >= total / 2) break;
  }

  const first = items.slice(0, splitIdx + 1);
  const second = items.slice(splitIdx + 1);
  const ratio = first.reduce((s, i) => s + i.size, 0) / total;

  if (horizontal) {
    return [
      ...layout(first, x, y, w * ratio, h, !horizontal),
      ...layout(second, x + w * ratio, y, w * (1 - ratio), h, !horizontal),
    ];
  } else {
    return [
      ...layout(first, x, y, w, h * ratio, !horizontal),
      ...layout(second, x, y + h * ratio, w, h * (1 - ratio), !horizontal),
    ];
  }
}

/* ─── Color helpers ─── */
function changeToColor(pct: number): string {
  const abs = Math.abs(pct);
  if (pct > 0) {
    if (abs >= 5) return 'rgba(22,163,74,0.85)';   // strong green
    if (abs >= 2) return 'rgba(34,197,94,0.70)';    // medium green
    return 'rgba(74,222,128,0.50)';                 // light green
  }
  if (pct < 0) {
    if (abs >= 5) return 'rgba(220,38,38,0.85)';    // strong red
    if (abs >= 2) return 'rgba(239,68,68,0.70)';    // medium red
    return 'rgba(252,165,165,0.50)';                // light red
  }
  return 'rgba(100,116,139,0.50)';                  // neutral
}

function changeToBorder(pct: number): string {
  if (pct > 0) return 'rgba(134,239,172,0.3)';
  if (pct < 0) return 'rgba(252,165,165,0.3)';
  return 'rgba(148,163,184,0.2)';
}

/* ─── Legend ─── */
const LEGEND = [
  { label: '< −5%', color: 'rgba(220,38,38,0.85)' },
  { label: '−2 ~ −5%', color: 'rgba(239,68,68,0.70)' },
  { label: '0 ~ −2%', color: 'rgba(252,165,165,0.50)' },
  { label: 'خنثی', color: 'rgba(100,116,139,0.50)' },
  { label: '0 ~ +2%', color: 'rgba(74,222,128,0.50)' },
  { label: '+2 ~ +5%', color: 'rgba(34,197,94,0.70)' },
  { label: '> +5%', color: 'rgba(22,163,74,0.85)' },
];

const CATEGORY_LABELS: Record<string, string> = {
  all: 'همه',
  stocks: 'سهام',
  crypto: 'کریپتو',
  stablecoins: 'استیبل‌کوین',
  commodities_etfs: 'کالا و ETF',
};

const SIZE_OPTIONS = [
  { value: 'marketCap', label: 'ارزش بازار' },
  { value: 'volume', label: 'حجم معاملات' },
];

const COLOR_OPTIONS = [
  { value: 'changePercent', label: 'بازدهی روزانه' },
];

/* ─── Single tile ─── */
function Tile({ item, containerW, containerH, onHover }: {
  item: LayoutItem;
  containerW: number;
  containerH: number;
  onHover: (item: LayoutItem | null) => void;
}) {
  const { rect, symbol, name, changePercent } = item;
  const px = (rect.x / containerW) * 100;
  const py = (rect.y / containerH) * 100;
  const pw = (rect.w / containerW) * 100;
  const ph = (rect.h / containerH) * 100;

  const area = rect.w * rect.h;
  const showName = area > 1800;
  const showPct = area > 800;
  const fontSize = Math.max(9, Math.min(18, Math.sqrt(area) / 7));

  const pctStr = `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;

  return (
    <div
      style={{
        position: 'absolute',
        left: `${px}%`,
        top: `${py}%`,
        width: `${pw}%`,
        height: `${ph}%`,
        padding: '1px',
        boxSizing: 'border-box',
      }}
      onMouseEnter={() => onHover(item)}
      onMouseLeave={() => onHover(null)}
    >
      <div
        style={{
          width: '100%',
          height: '100%',
          background: changeToColor(changePercent),
          border: `1px solid ${changeToBorder(changePercent)}`,
          borderRadius: '6px',
          backdropFilter: 'blur(4px)',
          WebkitBackdropFilter: 'blur(4px)',
          boxShadow: '0 1px 3px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.1)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          cursor: 'default',
          transition: 'filter 0.15s, transform 0.15s',
          position: 'relative',
        }}
        className="hover:brightness-110 hover:z-10"
      >
        {/* glass highlight */}
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: '40%',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 100%)',
          borderRadius: '5px 5px 0 0',
          pointerEvents: 'none',
        }} />

        {showName && (
          <span style={{
            fontSize: `${Math.max(10, fontSize)}px`,
            fontWeight: 700,
            color: '#fff',
            textShadow: '0 1px 3px rgba(0,0,0,0.5)',
            lineHeight: 1.2,
            textAlign: 'center',
            paddingInline: '4px',
            direction: 'ltr',
          }}>
            {symbol.replace('-USD', '')}
          </span>
        )}

        {showPct && (
          <span style={{
            fontSize: `${Math.max(8, fontSize - 3)}px`,
            fontWeight: 500,
            color: 'rgba(255,255,255,0.9)',
            textShadow: '0 1px 2px rgba(0,0,0,0.5)',
            marginTop: '2px',
            direction: 'ltr',
          }}>
            {pctStr}
          </span>
        )}

        {!showPct && area > 200 && (
          <span style={{
            fontSize: '7px',
            color: 'rgba(255,255,255,0.7)',
            direction: 'ltr',
          }}>{symbol.replace('-USD', '')}</span>
        )}
      </div>
    </div>
  );
}

/* ─── Tooltip ─── */
function Tooltip({ item }: { item: LayoutItem | null }) {
  if (!item) return null;
  return (
    <div className="pointer-events-none absolute z-50 top-2 left-2 rounded-xl border border-white/10 bg-background/80 backdrop-blur-md shadow-xl px-3 py-2 text-xs space-y-1 min-w-[160px]">
      <div className="font-bold text-sm" dir="ltr">{item.symbol}</div>
      <div className="text-muted-foreground">{item.name}</div>
      <div className="flex justify-between gap-4 pt-1">
        <span className="text-muted-foreground">قیمت</span>
        <span className="font-mono">${item.price < 1 ? item.price.toFixed(4) : item.price.toFixed(2)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-muted-foreground">تغییر</span>
        <span className={cn('font-mono font-semibold', item.changePercent >= 0 ? 'text-emerald-400' : 'text-red-400')}>
          {item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%
        </span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-muted-foreground">حجم</span>
        <span className="font-mono">{(item.volume / 1e6).toFixed(1)}M</span>
      </div>
      {item.marketCap && (
        <div className="flex justify-between gap-4">
          <span className="text-muted-foreground">ارزش بازار</span>
          <span className="font-mono">${(item.marketCap / 1e9).toFixed(1)}B</span>
        </div>
      )}
      <div className="flex justify-between gap-4">
        <span className="text-muted-foreground">دسته</span>
        <span>{CATEGORY_LABELS[item.category] ?? item.category}</span>
      </div>
    </div>
  );
}

/* ─── Main component ─── */
export function MarketMap({ data, loading }: { data: MapItem[]; loading?: boolean }) {
  const [category, setCategory] = useState('all');
  const [sizeBy, setSizeBy] = useState('marketCap');
  const [expanded, setExpanded] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<LayoutItem | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 400 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      setDims({ w: entry.contentRect.width, h: entry.contentRect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const categories = useMemo(() => {
    const cats = Array.from(new Set(data.map((d) => d.category)));
    return ['all', ...cats];
  }, [data]);

  const filtered = useMemo(() => {
    return category === 'all' ? data : data.filter((d) => d.category === category);
  }, [data, category]);

  const layoutItems = useMemo((): LayoutItem[] => {
    if (filtered.length === 0 || dims.w === 0 || dims.h === 0) return [];

    const withSize = filtered.map((d) => {
      const raw = sizeBy === 'marketCap' ? (d.marketCap ?? d.volume) : d.volume;
      return { ...d, size: Math.max(raw, 1) };
    });

    // Sort descending
    withSize.sort((a, b) => b.size - a.size);

    // Normalize sizes
    const maxSize = withSize[0]?.size || 1;
    const normalized = withSize.map((d) => ({ ...d, size: (d.size / maxSize) * 10000 }));

    return layout(normalized, 0, 0, dims.w, dims.h, dims.w > dims.h);
  }, [filtered, sizeBy, dims]);

  const gainers = filtered.filter((d) => d.changePercent > 0).length;
  const losers = filtered.filter((d) => d.changePercent < 0).length;
  const avgChange = filtered.length
    ? filtered.reduce((s, d) => s + d.changePercent, 0) / filtered.length
    : 0;

  return (
    <div className={cn(
      'rounded-2xl border border-white/10 bg-background/60 backdrop-blur-sm shadow-xl overflow-hidden',
      expanded && 'fixed inset-4 z-50',
    )}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b border-white/10 bg-white/5">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold">نقشه بازار</span>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-emerald-500 inline-block" />
              {gainers} صعودی
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-red-500 inline-block" />
              {losers} نزولی
            </span>
            <span className={cn(
              'font-semibold',
              avgChange >= 0 ? 'text-emerald-400' : 'text-red-400'
            )}>
              میانگین: {avgChange >= 0 ? '+' : ''}{avgChange.toFixed(2)}%
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Category filter */}
          <Select value={category} onValueChange={setCategory}>
            <SelectTrigger className="h-7 w-32 text-xs bg-white/5 border-white/10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {categories.map((c) => (
                <SelectItem key={c} value={c}>{CATEGORY_LABELS[c] ?? c}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Size by */}
          <Select value={sizeBy} onValueChange={setSizeBy}>
            <SelectTrigger className="h-7 w-36 text-xs bg-white/5 border-white/10">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SIZE_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <button
            onClick={() => setExpanded((e) => !e)}
            className="p-1.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-colors text-muted-foreground hover:text-foreground"
          >
            {expanded ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Map container */}
      <div
        ref={containerRef}
        className="relative bg-black/20"
        style={{ height: expanded ? 'calc(100vh - 160px)' : '420px' }}
      >
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
            در حال بارگذاری...
          </div>
        ) : layoutItems.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
            داده‌ای موجود نیست
          </div>
        ) : (
          <>
            {layoutItems.map((item) => (
              <Tile
                key={item.symbol}
                item={item}
                containerW={dims.w}
                containerH={dims.h}
                onHover={setHoveredItem}
              />
            ))}
            <Tooltip item={hoveredItem} />
          </>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-1.5 px-4 py-2 border-t border-white/10 bg-white/5 flex-wrap">
        {LEGEND.map((l) => (
          <div key={l.label} className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <span className="w-3 h-3 rounded-sm inline-block" style={{ background: l.color, border: '1px solid rgba(255,255,255,0.15)' }} />
            {l.label}
          </div>
        ))}
      </div>
    </div>
  );
}
