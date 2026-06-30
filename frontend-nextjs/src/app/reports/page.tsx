import { Suspense } from 'react';
import { ReportsContent } from '@/components/reports/reports-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { LlmStatusBadge } from '@/components/reports/llm-status-badge';

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI-Powered Reports</h1>
          <p className="text-muted-foreground mt-1">
            Intelligent insights and comprehensive analysis powered by open-source LLMs (Falcon, FinGPT) or simulated
          </p>
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