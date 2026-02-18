'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { StrategiesContent } from '@/components/strategies/strategies-content';
import { StrategyAgentPanel } from '@/components/trading/strategy-agent-panel';
import { BacktestRunner } from '@/components/backtesting/backtest-runner';
import { BacktestAgentPanel } from '@/components/agents/backtest-agent-panel';
import { Target, FlaskConical } from 'lucide-react';

type StrategiesTab = 'strategies' | 'backtesting';

function StrategiesPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<StrategiesTab>(() => {
    if (tabParam === 'backtesting') return 'backtesting';
    return 'strategies';
  });

  useEffect(() => {
    if (tabParam === 'backtesting' || tabParam === 'strategies') {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  function handleTabChange(value: string) {
    const tab = value as StrategiesTab;
    setActiveTab(tab);
    const params = new URLSearchParams(searchParams.toString());
    if (tab === 'strategies') params.delete('tab');
    else params.set('tab', tab);
    const query = params.toString();
    router.replace(query ? `/strategies?${query}` : '/strategies', { scroll: false });
  }

  return (
    <div className="container mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Strategies</h1>
        <p className="text-muted-foreground">
          Define, manage, and backtest trading strategies.
        </p>
      </div>
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="strategies" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Strategies
          </TabsTrigger>
          <TabsTrigger value="backtesting" className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4" />
            Backtesting
          </TabsTrigger>
        </TabsList>
        <TabsContent value="strategies" className="mt-6">
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
        </TabsContent>
        <TabsContent value="backtesting" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
            <div className="min-w-0">
              <Suspense fallback={<div>Loading backtesting interface...</div>}>
                <BacktestRunner />
              </Suspense>
            </div>
            <aside className="hidden xl:block min-h-[360px]">
              <BacktestAgentPanel />
            </aside>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function StrategiesPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-6 py-8 text-muted-foreground">Loading…</div>}>
      <StrategiesPageContent />
    </Suspense>
  );
}
