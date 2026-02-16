import { Suspense } from 'react';
import { RiskContent } from '@/components/risk/risk-content';
import { RiskAgentPanel } from '@/components/agents/risk-agent-panel';

export default function RiskPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Risk Dashboard</h1>
        <p className="text-muted-foreground">
          Analyze portfolio risk and run stress tests.
        </p>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div>Loading risk analysis...</div>}>
            <RiskContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <RiskAgentPanel />
        </aside>
      </div>
    </div>
  );
} 