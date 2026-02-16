'use client';

import { TradingBotsContent } from '@/components/trading/trading-bots-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { StrategyAgentPanel } from '@/components/trading/strategy-agent-panel';

export default function TradingBotsPage() {
  return (
    <div className="container mx-auto px-6 py-8">
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6 min-h-[480px]">
        <div className="min-w-0">
          <TradingBotsContent />
        </div>
        <aside className="hidden xl:flex xl:flex-col gap-4 min-h-[360px]">
          <div className="min-h-[280px]">
            <AnalysisAgentInsightsPanel />
          </div>
          <div className="min-h-[220px]">
            <StrategyAgentPanel />
          </div>
        </aside>
      </div>
    </div>
  );
}
