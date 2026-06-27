"use client";

/**
 * Dashboard — صفحه اصلی پلتفرم اختاپوس
 * شامل: خلاصه پورتفولیو، ویجت دارایی‌های ایرانی، منوی سریع
 */

import { AssetsDashboardWidget } from "./_components/AssetsDashboardWidget";

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-[#0a0b0f] text-white" dir="rtl">
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* Page header */}
        <div>
          <h1 className="text-xl font-bold">داشبورد</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            خلاصه وضعیت بازار و پورتفولیو
          </p>
        </div>

        {/* Grid layout: main content + sidebar */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Main content — ۲/۳ عرض */}
          <div className="lg:col-span-2 space-y-4">
            {/* TODO: PortfolioSummaryCard */}
            {/* TODO: MarketWatchlist */}
            {/* TODO: RecentTrades */}
            <div className="h-48 rounded-xl bg-white/[0.02] border border-white/5 flex items-center justify-center text-muted-foreground text-sm">
              خلاصه پورتفولیو — به زودی
            </div>
          </div>

          {/* Sidebar — ۱/۳ عرض */}
          <div className="space-y-4">
            {/* Iranian Assets Widget */}
            <AssetsDashboardWidget />

            {/* TODO: AISignalsWidget */}
            {/* TODO: RiskSummaryWidget */}
          </div>
        </div>

      </div>
    </div>
  );
}
