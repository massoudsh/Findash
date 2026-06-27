"use client";

import { Asset, formatToman, formatChange, CATEGORY_ICONS } from "@/lib/assets";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { SparklineChart } from "./AssetPriceChart";
import { useState } from "react";

interface AssetCardProps {
  asset: Asset;
  historyData?: { timestamp: string; close: number }[];
  onAddToPortfolio?: (symbol: string) => void;
}

export function AssetCard({ asset, historyData = [], onAddToPortfolio }: AssetCardProps) {
  const [hovered, setHovered] = useState(false);

  const isPositive = asset.change_percent_24h > 0;
  const isNeutral  = asset.change_percent_24h === 0;

  const trendColor = isNeutral
    ? "text-muted-foreground"
    : isPositive
    ? "text-emerald-400"
    : "text-red-400";

  const TrendIcon = isNeutral ? Minus : isPositive ? TrendingUp : TrendingDown;

  return (
    <Card
      className={`
        relative overflow-hidden border border-white/5 bg-[#0f1117]
        hover:border-white/15 hover:bg-[#13161f]
        transition-all duration-200 cursor-pointer group
      `}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* colored left-border accent based on category */}
      <div className={`
        absolute right-0 top-0 h-full w-0.5
        ${asset.category === "gold"        ? "bg-yellow-400/60" : ""}
        ${asset.category === "silver"      ? "bg-slate-400/60"  : ""}
        ${asset.category === "currency"    ? "bg-blue-400/60"   : ""}
        ${asset.category === "real_estate" ? "bg-orange-400/60" : ""}
        ${asset.category === "crypto"      ? "bg-purple-400/60" : ""}
      `} />

      <CardContent className="p-4 space-y-3">
        {/* Header row */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xl leading-none">
              {CATEGORY_ICONS[asset.category]}
            </span>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-white truncate leading-tight">
                {asset.name_fa}
              </p>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                {asset.symbol}
              </p>
            </div>
          </div>
          <Badge
            variant="outline"
            className={`text-[11px] font-medium border-0 px-2 py-0.5 shrink-0 ${
              isNeutral
                ? "bg-white/5 text-muted-foreground"
                : isPositive
                ? "bg-emerald-400/10 text-emerald-400"
                : "bg-red-400/10 text-red-400"
            }`}
          >
            <TrendIcon className="w-3 h-3 ml-1 inline" />
            {formatChange(asset.change_percent_24h)}
          </Badge>
        </div>

        {/* Price */}
        <div>
          <p className="text-xl font-bold text-white tabular-nums">
            {formatToman(asset.price_toman)}
          </p>
          <p className={`text-xs mt-0.5 tabular-nums ${trendColor}`}>
            {isPositive ? "+" : ""}
            {formatToman(Math.abs(asset.change_24h))} نسبت به دیروز
          </p>
        </div>

        {/* Sparkline */}
        {historyData.length > 1 && (
          <div className="h-12 -mx-1">
            <SparklineChart
              data={historyData}
              positive={isPositive}
            />
          </div>
        )}

        {/* 24h range */}
        <div className="flex justify-between text-[11px] text-muted-foreground pt-1 border-t border-white/5">
          <span>
            کف ۲۴h:{" "}
            <span className="text-white/60">{formatToman(asset.low_24h)}</span>
          </span>
          <span>
            سقف ۲۴h:{" "}
            <span className="text-white/60">{formatToman(asset.high_24h)}</span>
          </span>
        </div>

        {/* Add to portfolio button — shows on hover */}
        {onAddToPortfolio && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onAddToPortfolio(asset.symbol);
            }}
            className={`
              w-full text-xs py-1.5 rounded-md font-medium
              bg-white/5 hover:bg-white/10 text-white/60 hover:text-white
              border border-white/5 hover:border-white/15
              transition-all duration-150
              ${hovered ? "opacity-100" : "opacity-0"}
            `}
          >
            + افزودن به پورتفولیو
          </button>
        )}
      </CardContent>
    </Card>
  );
}
