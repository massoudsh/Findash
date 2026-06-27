"use client";

import { Asset, AssetCategory, CATEGORY_LABELS, fetchAssets } from "@/lib/assets";
import { AssetCard } from "./AssetCard";
import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

interface AssetGridProps {
  category?: AssetCategory;
  onAddToPortfolio?: (symbol: string) => void;
}

export function AssetGrid({ category, onAddToPortfolio }: AssetGridProps) {
  const [assets, setAssets]   = useState<Asset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAssets(category);
      setAssets(res.assets);
    } catch {
      setError("دریافت قیمت‌ها با خطا مواجه شد. لطفاً دوباره تلاش کنید.");
    } finally {
      setLoading(false);
    }
  }, [category]);

  useEffect(() => {
    load();
    // Auto-refresh every 60 seconds
    const interval = setInterval(load, 60_000);
    return () => clearInterval(interval);
  }, [load]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">در حال دریافت قیمت‌ها...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-3">
        <p className="text-sm text-red-400">{error}</p>
        <button
          onClick={load}
          className="text-xs text-white/50 hover:text-white border border-white/10 hover:border-white/20 px-3 py-1.5 rounded-md transition-colors"
        >
          تلاش مجدد
        </button>
      </div>
    );
  }

  if (!assets.length) {
    return (
      <div className="flex items-center justify-center py-20 text-muted-foreground text-sm">
        {category ? `هیچ دارایی در دسته «${CATEGORY_LABELS[category]}» یافت نشد.` : "دارایی‌ای یافت نشد."}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {assets.map((asset) => (
        <AssetCard
          key={asset.symbol}
          asset={asset}
          onAddToPortfolio={onAddToPortfolio}
        />
      ))}
    </div>
  );
}
