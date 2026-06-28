"use client";

/**
 * Dashboard — صفحه اصلی پلتفرم اختاپوس
 * TASK-001e: AssetsDashboardWidget
 * TASK-002b: IranMacroWidget
 * TASK-002c: CurrencyComparisonCard
 * TASK-004:  CurrencyToggle
 */

import { AssetsDashboardWidget } from "./_components/AssetsDashboardWidget";
import { CurrencyToggle }        from "./_components/CurrencyToggle";
import { IranMacroWidget }       from "./dashboard/_components/IranMacroWidget";
import { CurrencyComparisonCard } from "./dashboard/_components/CurrencyComparisonCard";
import { BarChart2 }             from "lucide-react";

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-[#0a0b0f] text-white" dir="rtl">
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-400/10 border border-blue-400/20">
              <BarChart2 className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold">داشبورد</h1>
              <p className="text-xs text-muted-foreground mt-0.5">
                خلاصه وضعیت بازار و پورتفولیو
              </p>
            </div>
          </div>
          <CurrencyToggle />
        </div>

        {/* Layout: main (2/3) + sidebar (1/3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Sidebar */}
          <div className="order-first lg:order-last space-y-4">
            <AssetsDashboardWidget />
            <CurrencyComparisonCard />
          </div>

          {/* Main */}
          <div className="lg:col-span-2 space-y-4">
            <IranMacroWidget />
            {/* TODO: PortfolioSummaryCard, MarketWatchlist, RecentTrades */}
          </div>

        </div>
      </div>
    </div>
  );
}
