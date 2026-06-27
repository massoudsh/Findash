"use client";

/**
 * AssetsDashboardWidget
 * A compact widget for the main dashboard that shows:
 * - USD/Toman rate
 * - Top 6 assets (3 biggest gainers + 3 biggest losers)
 * - Link to full /assets page
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { TrendingUp, TrendingDown, ArrowLeft, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Asset, AssetListResponse, fetchAssets, formatToman, formatChange, CATEGORY_ICONS } from "@/lib/assets";

interface TopMover {
  asset: Asset;
  isPositive: boolean;
}

export function AssetsDashboardWidget() {
  const [topMovers, setTopMovers] = useState<TopMover[]>([]);
  const [usdRate, setUsdRate]     = useState<number>(0);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res: AssetListResponse = await fetchAssets();
        setUsdRate(res.usd_to_toman);

        // Sort by abs(change_percent_24h), pick top 3 positive + top 3 negative
        const sorted = [...res.assets].sort(
          (a, b) => Math.abs(b.change_percent_24h) - Math.abs(a.change_percent_24h)
        );
        const gainers = sorted.filter((a) => a.change_percent_24h > 0).slice(0, 3);
        const losers  = sorted.filter((a) => a.change_percent_24h < 0).slice(0, 3);

        setTopMovers([
          ...gainers.map((a) => ({ asset: a, isPositive: true })),
          ...losers.map((a)  => ({ asset: a, isPositive: false })),
        ]);
      } catch {
        // widget fails silently — dashboard should not break
      } finally {
        setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 60_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="bg-[#0f1117] border border-white/5 overflow-hidden" dir="rtl">
      <CardHeader className="pb-2 pt-4 px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-base">📊</span>
            <h3 className="text-sm font-semibold text-white">دارایی‌های ایرانی</h3>
          </div>
          <div className="flex items-center gap-3">
            {usdRate > 0 && (
              <span className="text-xs text-muted-foreground">
                💵 <span className="text-blue-400 font-medium">{formatToman(usdRate)}</span>
              </span>
            )}
            <Link
              href="/assets"
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-white transition-colors"
            >
              مشاهده همه
              <ArrowLeft className="w-3 h-3" />
            </Link>
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-4 pb-4">
        {loading ? (
          <div className="flex items-center justify-center py-6 text-muted-foreground gap-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <span className="text-xs">در حال بارگذاری...</span>
          </div>
        ) : (
          <div className="space-y-1.5">
            {topMovers.map(({ asset, isPositive }) => (
              <div
                key={asset.symbol}
                className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                {/* Name */}
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span className="text-sm leading-none shrink-0">
                    {CATEGORY_ICONS[asset.category]}
                  </span>
                  <span className="text-xs text-white/80 truncate">
                    {asset.name_fa}
                  </span>
                </div>

                {/* Price */}
                <span className="text-xs font-medium text-white tabular-nums mx-3 shrink-0">
                  {formatToman(asset.price_toman)}
                </span>

                {/* Change badge */}
                <span
                  className={`
                    flex items-center gap-0.5 text-[11px] font-medium tabular-nums shrink-0
                    ${isPositive ? "text-emerald-400" : "text-red-400"}
                  `}
                >
                  {isPositive
                    ? <TrendingUp className="w-3 h-3" />
                    : <TrendingDown className="w-3 h-3" />
                  }
                  {formatChange(asset.change_percent_24h)}
                </span>
              </div>
            ))}

            {topMovers.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-4">
                داده‌ای موجود نیست
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
