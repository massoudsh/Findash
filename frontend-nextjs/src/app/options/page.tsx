'use client';

import { Suspense, useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { OptionTradingTerminal } from '@/components/options/option-trading-terminal';
import { OptionsStrategiesTab } from '@/components/options/options-strategies-tab';
import { TrendingUp, Target } from 'lucide-react';

export interface OptionsStrategySelection {
  name: string;
  description?: string;
  category?: string;
}

const STRATEGY_PNL_TICK_MS = 4000;
const STRATEGY_PNL_DELTA_MIN = -12;
const STRATEGY_PNL_DELTA_MAX = 18;

function OptionsPageContent() {
  const searchParams = useSearchParams();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<'trade' | 'strategies'>(
    tabParam === 'strategies' ? 'strategies' : 'trade'
  );
  const [selectedStrategyForTerminal, setSelectedStrategyForTerminal] = useState<OptionsStrategySelection | null>(null);
  const [strategyPnl, setStrategyPnl] = useState(0);

  useEffect(() => {
    if (tabParam === 'strategies') setActiveTab('strategies');
  }, [tabParam]);

  // Dynamic PnL updates while a strategy is active (simulated)
  useEffect(() => {
    if (!selectedStrategyForTerminal) return;
    const t = setInterval(() => {
      const delta = STRATEGY_PNL_DELTA_MIN + Math.random() * (STRATEGY_PNL_DELTA_MAX - STRATEGY_PNL_DELTA_MIN);
      setStrategyPnl((prev) => Math.round((prev + delta) * 100) / 100);
    }, STRATEGY_PNL_TICK_MS);
    return () => clearInterval(t);
  }, [selectedStrategyForTerminal]);

  function handleTradeInTerminal(strategy?: OptionsStrategySelection) {
    setSelectedStrategyForTerminal(strategy ?? null);
    if (strategy) setStrategyPnl(0);
    setActiveTab('trade');
  }

  function handleClearStrategy() {
    setSelectedStrategyForTerminal(null);
    setStrategyPnl(0);
  }

  return (
    <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'trade' | 'strategies')} className="flex-1 flex flex-col">
      <div className="border-b bg-background/95 px-4 py-2 shrink-0">
        <TabsList className="grid w-full max-w-full sm:max-w-md grid-cols-2">
          <TabsTrigger value="trade" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Trade
          </TabsTrigger>
          <TabsTrigger value="strategies" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Strategies
          </TabsTrigger>
        </TabsList>
      </div>
      <TabsContent value="trade" className="flex-1 m-0 mt-0 min-h-0">
        <OptionTradingTerminal
          selectedStrategy={selectedStrategyForTerminal}
          strategyPnl={strategyPnl}
          onClearStrategy={handleClearStrategy}
        />
      </TabsContent>
      <TabsContent value="strategies" className="flex-1 m-0 mt-0 overflow-auto">
        <OptionsStrategiesTab onTradeInTerminal={handleTradeInTerminal} />
      </TabsContent>
    </Tabs>
  );
}

export default function OptionsPage() {
  return (
    <div className="w-full h-full min-h-screen flex flex-col">
      <Suspense fallback={<div className="flex flex-1 items-center justify-center text-muted-foreground">Loading options…</div>}>
        <OptionsPageContent />
      </Suspense>
    </div>
  );
}
