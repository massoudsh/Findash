'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Plus, Trash2, TrendingUp, TrendingDown, Wallet } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  AddAssetModal,
  ASSET_PRESETS,
  CATEGORY_LABEL,
  formatToman,
  loadIranAssets,
  saveIranAssets,
  type AssetType,
  type IranPortfolioAsset,
} from './add-asset-modal';

// ─── Colors per asset type ────────────────────────────────────────────────────

const TYPE_COLOR: Record<AssetType, string> = {
  gold:        '#f59e0b',
  silver:      '#94a3b8',
  currency:    '#10b981',
  crypto:      '#f97316',
  stock:       '#3b82f6',
  bond:        '#8b5cf6',
  real_estate: '#ef4444',
  cash:        '#6b7280',
};

// ─── Donut Chart ──────────────────────────────────────────────────────────────

interface DonutSegment {
  type: AssetType;
  label: string;
  value: number;
  pct: number;
  color: string;
}

function DonutChart({ segments, totalValue }: { segments: DonutSegment[]; totalValue: number }) {
  const R = 70;
  const CX = 90;
  const CY = 90;
  const circumference = 2 * Math.PI * R;

  let offset = 0;
  const slices = segments.map((s) => {
    const dashArray = (s.pct / 100) * circumference;
    const dashOffset = circumference - offset * circumference / 100;
    const startOffset = offset;
    offset += s.pct;
    return { ...s, dashArray, dashOffset: circumference - startOffset * circumference / 100 };
  });

  if (segments.length === 0) {
    return (
      <div className="flex items-center justify-center h-[180px]">
        <div className="text-center text-muted-foreground text-sm">
          <Wallet className="h-8 w-8 mx-auto mb-2 opacity-40" />
          <p>هنوز دارایی ثبت نشده</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width="180" height="180" viewBox="0 0 180 180">
        {/* background circle */}
        <circle cx={CX} cy={CY} r={R} fill="none" stroke="currentColor" strokeWidth="2" className="text-muted/30" />

        {slices.map((s, i) => (
          <circle
            key={`${s.type}-${i}`}
            cx={CX}
            cy={CY}
            r={R}
            fill="none"
            stroke={s.color}
            strokeWidth="24"
            strokeDasharray={`${s.dashArray} ${circumference - s.dashArray}`}
            strokeDashoffset={s.dashOffset}
            strokeLinecap="butt"
            style={{ transform: 'rotate(-90deg)', transformOrigin: `${CX}px ${CY}px` }}
          />
        ))}

        {/* center text */}
        <text x={CX} y={CY - 8} textAnchor="middle" fontSize="10" className="fill-muted-foreground">
          ارزش کل
        </text>
        <text x={CX} y={CY + 8} textAnchor="middle" fontSize="9" fontWeight="600" className="fill-foreground">
          {formatToman(totalValue)}
        </text>
        <text x={CX} y={CY + 22} textAnchor="middle" fontSize="8" className="fill-muted-foreground">
          تومان
        </text>
      </svg>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 w-full text-xs">
        {slices.map((s) => (
          <div key={s.type} className="flex items-center gap-1.5 truncate">
            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: s.color }} />
            <span className="text-muted-foreground truncate">{s.label}</span>
            <span className="font-medium mr-auto">{s.pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function IranPortfolioSection() {
  const [assets, setAssets] = useState<IranPortfolioAsset[]>([]);
  const [modalOpen, setModalOpen] = useState(false);

  useEffect(() => {
    setAssets(loadIranAssets());
  }, []);

  const handleAdd = useCallback((asset: IranPortfolioAsset) => {
    setAssets((prev) => {
      const updated = [asset, ...prev];
      saveIranAssets(updated);
      return updated;
    });
  }, []);

  const handleRemove = useCallback((id: string) => {
    setAssets((prev) => {
      const updated = prev.filter((a) => a.id !== id);
      saveIranAssets(updated);
      return updated;
    });
  }, []);

  // Compute positions (net buy - sell per code)
  const positions = useMemo(() => {
    const map: Record<string, {
      name: string; code: string; type: AssetType; qty: number; totalSpent: number; currency: 'IRT' | 'USD';
    }> = {};
    for (const a of assets) {
      const key = a.code;
      if (!map[key]) map[key] = { name: a.name, code: a.code, type: a.type, qty: 0, totalSpent: 0, currency: a.currency };
      if (a.side === 'buy') {
        map[key].qty += a.quantity;
        map[key].totalSpent += a.totalValue;
      } else {
        map[key].qty -= a.quantity;
        map[key].totalSpent -= a.totalValue;
      }
    }
    return Object.values(map).filter((p) => p.qty > 0);
  }, [assets]);

  // Category breakdown for donut
  const donutSegments = useMemo((): DonutSegment[] => {
    const byType: Partial<Record<AssetType, number>> = {};
    for (const p of positions) {
      byType[p.type] = (byType[p.type] ?? 0) + p.totalSpent;
    }
    const total = Object.values(byType).reduce((s: number, v) => s + (v ?? 0), 0) || 1;
    return (Object.entries(byType) as [AssetType, number][])
      .sort(([, a], [, b]) => b - a)
      .map(([type, value]) => ({
        type,
        label: CATEGORY_LABEL[type],
        value,
        pct: (value / total) * 100,
        color: TYPE_COLOR[type],
      }));
  }, [positions]);

  const totalPortfolioValue = useMemo(
    () => positions.reduce((s, p) => s + p.totalSpent, 0),
    [positions]
  );

  const assetIcon = (code: string): string =>
    ASSET_PRESETS.find((p) => p.code === code)?.icon ?? '📦';

  return (
    <div className="space-y-4" dir="rtl">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">دارایی‌های من</h2>
          <p className="text-xs text-muted-foreground">ارزش کل: {formatToman(totalPortfolioValue)} تومان</p>
        </div>
        <Button
          size="sm"
          className="gap-1.5 bg-green-600 hover:bg-green-700 text-white"
          onClick={() => setModalOpen(true)}
        >
          <Plus className="h-4 w-4" />
          افزودن دارایی
        </Button>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-3 gap-3">
        <Card>
          <CardContent className="pt-3 pb-3">
            <p className="text-xs text-muted-foreground mb-0.5">ارزش پرتفوی</p>
            <p className="text-base font-bold leading-tight">
              {formatToman(totalPortfolioValue)}
            </p>
            <p className="text-xs text-muted-foreground">تومان</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-3 pb-3">
            <p className="text-xs text-muted-foreground mb-0.5">تعداد دارایی</p>
            <p className="text-base font-bold">{positions.length}</p>
            <p className="text-xs text-muted-foreground">نماد مختلف</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-3 pb-3">
            <p className="text-xs text-muted-foreground mb-0.5">دسته‌بندی</p>
            <p className="text-base font-bold">{donutSegments.length}</p>
            <p className="text-xs text-muted-foreground">نوع دارایی</p>
          </CardContent>
        </Card>
      </div>

      {/* Chart + Holdings */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Donut */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">تخصیص دارایی</CardTitle>
          </CardHeader>
          <CardContent>
            <DonutChart segments={donutSegments} totalValue={totalPortfolioValue} />
          </CardContent>
        </Card>

        {/* Holdings list */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">ترکیب دارایی‌ها</CardTitle>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground text-sm gap-2">
                <Wallet className="h-8 w-8 opacity-40" />
                <p>دارایی ثبت نشده</p>
                <Button size="sm" variant="outline" onClick={() => setModalOpen(true)}>
                  <Plus className="h-3 w-3 ml-1" />
                  افزودن اولین دارایی
                </Button>
              </div>
            ) : (
              <div className="space-y-2 max-h-56 overflow-y-auto pl-1">
                {positions.map((p) => {
                  const pct = totalPortfolioValue > 0 ? (p.totalSpent / totalPortfolioValue) * 100 : 0;
                  return (
                    <div
                      key={p.code}
                      className="flex items-center gap-2 p-2 rounded-lg border hover:bg-muted/30 transition-colors"
                    >
                      <span className="text-lg flex-shrink-0">{assetIcon(p.code)}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-sm font-medium truncate">{p.name}</span>
                          <span className="text-xs text-muted-foreground flex-shrink-0">{pct.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-muted/40 rounded-full h-1 mt-1">
                          <div
                            className="h-1 rounded-full"
                            style={{ width: `${pct}%`, backgroundColor: TYPE_COLOR[p.type] }}
                          />
                        </div>
                        <div className="flex items-center justify-between mt-0.5">
                          <span className="text-xs text-muted-foreground">
                            تعداد: {p.qty.toLocaleString('fa-IR')}
                          </span>
                          <span className="text-xs font-medium">
                            {formatToman(p.totalSpent)} {p.currency === 'IRT' ? 'ت' : '$'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Transaction history */}
      {assets.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">تاریخچه تراکنش‌ها</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5 max-h-52 overflow-y-auto">
              {assets.map((a) => (
                <div
                  key={a.id}
                  className="flex items-center justify-between gap-2 p-2 rounded-lg border text-sm hover:bg-muted/20"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-base flex-shrink-0">{assetIcon(a.code)}</span>
                    <Badge
                      variant={a.side === 'buy' ? 'default' : 'destructive'}
                      className={cn(
                        'text-xs px-1.5 py-0 flex-shrink-0',
                        a.side === 'buy' && 'bg-green-600 text-white'
                      )}
                    >
                      {a.side === 'buy' ? 'خرید' : 'فروش'}
                    </Badge>
                    <span className="font-medium truncate">{a.name}</span>
                    <span className="text-muted-foreground text-xs flex-shrink-0">
                      ×{a.quantity.toLocaleString('fa-IR')}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <div className="text-left">
                      <p className="text-xs font-medium">{formatToman(a.totalValue)}</p>
                      <p className="text-xs text-muted-foreground">
                        {a.currency === 'IRT' ? 'تومان' : 'USD'} —{' '}
                        {new Date(a.timestamp).toLocaleDateString('fa-IR')}
                      </p>
                    </div>
                    {a.side === 'buy' ? (
                      <TrendingUp className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                    ) : (
                      <TrendingDown className="h-3.5 w-3.5 text-red-500 flex-shrink-0" />
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 flex-shrink-0"
                      onClick={() => handleRemove(a.id)}
                    >
                      <Trash2 className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <AddAssetModal open={modalOpen} onOpenChange={setModalOpen} onAdd={handleAdd} />
    </div>
  );
}
