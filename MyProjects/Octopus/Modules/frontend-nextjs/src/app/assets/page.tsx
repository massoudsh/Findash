"use client";

import { useState, useCallback } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AssetCategory, CATEGORY_LABELS, CATEGORY_ICONS, fetchAssets } from "@/lib/assets";
import { AssetGrid } from "./_components/AssetGrid";
import { AssetSummaryBar } from "./_components/AssetSummaryBar";
import { BarChart2 } from "lucide-react";
import { useEffect } from "react";

const ALL_CATEGORIES: AssetCategory[] = [
  "gold",
  "silver",
  "currency",
  "real_estate",
  "crypto",
];

export default function AssetsPage() {
  const [activeTab, setActiveTab] = useState<AssetCategory | "all">("all");
  const [summaryData, setSummaryData] = useState({
    usdToToman:    0,
    lastUpdated:   new Date().toISOString(),
    totalPositive: 0,
    totalNegative: 0,
  });

  // Load summary data once and refresh every 60s
  const loadSummary = useCallback(async () => {
    try {
      const res = await fetchAssets();
      const pos = res.assets.filter((a) => a.change_percent_24h > 0).length;
      const neg = res.assets.filter((a) => a.change_percent_24h < 0).length;
      setSummaryData({
        usdToToman:    res.usd_to_toman,
        lastUpdated:   res.last_updated,
        totalPositive: pos,
        totalNegative: neg,
      });
    } catch {
      // fail silently — individual grids show their own error states
    }
  }, []);

  useEffect(() => {
    loadSummary();
    const interval = setInterval(loadSummary, 60_000);
    return () => clearInterval(interval);
  }, [loadSummary]);

  const handleAddToPortfolio = (symbol: string) => {
    // TODO: open AddToPortfolioModal with symbol pre-filled
    console.log("Add to portfolio:", symbol);
  };

  return (
    <div className="min-h-screen bg-[#0a0b0f] text-white" dir="rtl">
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* Page header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-yellow-400/10 border border-yellow-400/20">
              <BarChart2 className="w-5 h-5 text-yellow-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold">دارایی‌ها</h1>
              <p className="text-xs text-muted-foreground mt-0.5">
                قیمت لحظه‌ای طلا، نقره، ارز، مسکن و کریپتو
              </p>
            </div>
          </div>
        </div>

        {/* Summary bar */}
        <AssetSummaryBar {...summaryData} />

        {/* Tabs */}
        <Tabs
          defaultValue="all"
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as AssetCategory | "all")}
        >
          <TabsList className="bg-white/5 border border-white/10 rounded-xl p-1 gap-0.5 flex-wrap h-auto">
            <TabsTrigger
              value="all"
              className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground"
            >
              🗂 همه
            </TabsTrigger>
            {ALL_CATEGORIES.map((cat) => (
              <TabsTrigger
                key={cat}
                value={cat}
                className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground"
              >
                {CATEGORY_ICONS[cat]} {CATEGORY_LABELS[cat]}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="all" className="mt-4">
            {/* Render all categories in grouped sections */}
            <div className="space-y-8">
              {ALL_CATEGORIES.map((cat) => (
                <section key={cat}>
                  <h2 className="flex items-center gap-2 text-sm font-semibold text-white/70 mb-3">
                    <span>{CATEGORY_ICONS[cat]}</span>
                    <span>{CATEGORY_LABELS[cat]}</span>
                    <div className="flex-1 h-px bg-white/5 mr-2" />
                  </h2>
                  <AssetGrid
                    category={cat}
                    onAddToPortfolio={handleAddToPortfolio}
                  />
                </section>
              ))}
            </div>
          </TabsContent>

          {ALL_CATEGORIES.map((cat) => (
            <TabsContent key={cat} value={cat} className="mt-4">
              <AssetGrid
                category={cat}
                onAddToPortfolio={handleAddToPortfolio}
              />
            </TabsContent>
          ))}
        </Tabs>

      </div>
    </div>
  );
}
