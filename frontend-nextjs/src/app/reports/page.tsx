import { Suspense } from 'react';
import { ReportsContent } from '@/components/reports/reports-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AI-Powered Reports</h1>
        <p className="text-muted-foreground">
          Intelligent insights and comprehensive analysis powered by Llama AI models
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div>Loading AI reporting system...</div>}>
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