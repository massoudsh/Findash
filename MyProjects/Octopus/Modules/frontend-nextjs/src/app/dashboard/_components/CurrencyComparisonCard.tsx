"use client";

/**
 * TASK-002c — CurrencyComparisonCard
 * مقایسه ارزش دارایی به تومان vs دلار به صورت real-time
 * ورودی: مقدار تومانی + نرخ دلار
 */

import { useState, useEffect } from "react";
import { ArrowLeftRight, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { formatJalali } from "@/lib/locale";

interface AssetComparison {
  name: string;
  icon: string;
  valueToman: number;
}

const SAMPLE_ASSETS: AssetComparison[] = [
  { name: "طلای ۱۸ عیار (۱۰ گرم)", icon: "🥇", valueToman: 0 },
  { name: "سکه بهار آزادی (۱ عدد)", icon: "🪙", valueToman: 0 },
  { name: "دلار (۱۰۰۰ دلار)", icon: "💵", valueToman: 0 },
  { name: "نقره (۱۰۰ گرم)",          icon: "🥈", valueToman: 0 },
];

export function CurrencyComparisonCard() {
  const [usdRate, setUsdRate]     = useState<number>(0);
  const [loading, setLoading]     = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string>("");

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("/api/assets/usd-rate");
        if (res.ok) {
          const data = await res.json();
          setUsdRate(data.usd_to_toman ?? 0);
          setUpdatedAt(new Date().toISOString());
        }
      } catch {
        // fail silently
      } finally {
        setLoading(false);
      }
    };
    load();
    const t = setInterval(load, 120_000);
    return () => clearInterval(t);
  }, []);

  // نرخ‌های نمونه — در production از /api/assets/{symbol} بخوان
  const mockPrices: Record<string, number> = {
    "طلای ۱۸ عیار (۱۰ گرم)":   usdRate > 0 ? usdRate * 0.028 * 10 : 3_500_000,
    "سکه بهار آزادی (۱ عدد)": usdRate > 0 ? usdRate * 0.215 : 40_000_000,
    "دلار (۱۰۰۰ دلار)":         usdRate * 1000,
    "نقره (۱۰۰ گرم)":           usdRate > 0 ? usdRate * 0.0009 * 100 : 150_000,
  };

  return (
    <Card className="bg-[#0f1117] border border-white/5" dir="rtl">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <ArrowLeftRight className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">مقایسه ارزش دارایی</h3>
          </div>
          {updatedAt && (
            <span className="text-[10px] text-muted-foreground">
              {formatJalali(updatedAt, true)}
            </span>
          )}
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-4 gap-2 text-muted-foreground">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <span className="text-xs">در حال دریافت نرخ دلار...</span>
          </div>
        ) : (
          <div className="space-y-2">
            {SAMPLE_ASSETS.map((a) => {
              const toman = mockPrices[a.name] ?? 0;
              const usd   = usdRate > 0 ? toman / usdRate : 0;
              return (
                <div
                  key={a.name}
                  className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-white/5"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{a.icon}</span>
                    <span className="text-xs text-white/80">{a.name}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs tabular-nums">
                    <span className="text-white font-medium">
                      {(toman / 1_000_000).toFixed(1)} م.ت
                    </span>
                    {usdRate > 0 && (
                      <>
                        <span className="text-white/20">|</span>
                        <span className="text-blue-300">
                          ${usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {usdRate > 0 && (
          <p className="text-[10px] text-muted-foreground/50 mt-3 text-center">
            نرخ دلار آزاد:{" "}
            <span className="text-blue-400">
              {new Intl.NumberFormat("fa-IR").format(usdRate)} تومان
            </span>
          </p>
        )}
      </CardContent>
    </Card>
  );
}
