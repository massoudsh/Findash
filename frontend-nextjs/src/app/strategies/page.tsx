import { Suspense } from 'react';
import { StrategiesContent } from '@/components/strategies/strategies-content';
import { StrategyAgentPanel } from '@/components/trading/strategy-agent-panel';

export default function StrategiesPage() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
        <div className="min-w-0">
          <Suspense fallback={<div className="text-center">Loading strategies...</div>}>
            <StrategiesContent />
          </Suspense>
        </div>
        <aside className="hidden xl:block min-h-[360px]">
          <StrategyAgentPanel />
        </aside>
      </div>
    </div>
  );
} 