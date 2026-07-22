import Link from 'next/link';
import { Suspense } from 'react';
import { ReportsContent } from '@/components/reports/reports-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { LlmStatusBadge } from '@/components/reports/llm-status-badge';
import { PieChart } from 'lucide-react';

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">گزارش‌های مبتنی بر هوش مصنوعی</h1>
          <p className="text-muted-foreground mt-1">
            بینش‌های هوشمند و تحلیل جامع مبتنی بر مدل‌های زبانی متن‌باز (Falcon، FinGPT) یا شبیه‌سازی‌شده
          </p>
          <Link
            href="/data?tab=charts"
            className="inline-flex items-center gap-1.5 mt-2 text-xs font-medium text-primary hover:underline"
          >
            <PieChart className="h-3.5 w-3.5" />
            مشاهده نمودارهای پایه این گزارش
          </Link>
        </div>
        <LlmStatusBadge />
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div>در حال بارگذاری سیستم گزارش‌دهی...</div>}>
            <ReportsContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <AnalysisAgentInsightsPanel />
        </aside>
      </div>
    </div>
  );
} 