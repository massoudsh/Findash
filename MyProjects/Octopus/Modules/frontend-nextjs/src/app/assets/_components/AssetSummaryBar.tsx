"use client";

import { formatToman } from "@/lib/assets";
import { TrendingUp, TrendingDown, DollarSign, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

interface SummaryBarProps {
  usdToToman: number;
  lastUpdated: string;
  totalPositive: number;   // count of assets up today
  totalNegative: number;   // count of assets down today
}

export function AssetSummaryBar({
  usdToToman,
  lastUpdated,
  totalPositive,
  totalNegative,
}: SummaryBarProps) {
  const [timeAgo, setTimeAgo] = useState("");

  useEffect(() => {
    const update = () => {
      const diff = Math.floor((Date.now() - new Date(lastUpdated).getTime()) / 1000);
      if (diff < 60)      setTimeAgo(`${diff} ثانیه پیش`);
      else if (diff < 3600) setTimeAgo(`${Math.floor(diff / 60)} دقیقه پیش`);
      else                setTimeAgo(`${Math.floor(diff / 3600)} ساعت پیش`);
    };
    update();
    const t = setInterval(update, 10_000);
    return () => clearInterval(t);
  }, [lastUpdated]);

  return (
    <div className="flex flex-wrap items-center gap-4 px-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/5 text-sm">
      {/* USD rate */}
      <div className="flex items-center gap-2 text-white/80">
        <DollarSign className="w-4 h-4 text-blue-400" />
        <span className="text-muted-foreground text-xs">دلار:</span>
        <span className="font-semibold tabular-nums">{formatToman(usdToToman)}</span>
      </div>

      <div className="h-4 w-px bg-white/10" />

      {/* Market breadth */}
      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1 text-emerald-400">
          <TrendingUp className="w-3.5 h-3.5" />
          <span className="tabular-nums">{totalPositive}</span>
          <span className="text-muted-foreground text-xs">صعودی</span>
        </span>
        <span className="flex items-center gap-1 text-red-400">
          <TrendingDown className="w-3.5 h-3.5" />
          <span className="tabular-nums">{totalNegative}</span>
          <span className="text-muted-foreground text-xs">نزولی</span>
        </span>
      </div>

      <div className="h-4 w-px bg-white/10 hidden sm:block" />

      {/* Last update */}
      <div className="hidden sm:flex items-center gap-1.5 text-muted-foreground text-xs mr-auto">
        <RefreshCw className="w-3 h-3 animate-[spin_3s_linear_infinite]" />
        آخرین بروزرسانی: {timeAgo}
      </div>
    </div>
  );
}
