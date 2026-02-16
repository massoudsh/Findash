'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { RealtimeContent } from '@/components/realtime/realtime-content';
import TradingCenterPage from '@/app/trades/page';
import { TradingBotsContent } from '@/components/trading/trading-bots-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { DataCollectorAgentPanel } from '@/components/trading/data-collector-agent-panel';
import { StrategyAgentPanel } from '@/components/trading/strategy-agent-panel';
import { SentimentAgentPanel } from '@/components/trading/sentiment-agent-panel';
import { Activity, TrendingUp, Brain } from 'lucide-react';

type TradingTab = 'market' | 'center' | 'bots';

function TradingCenterContent() {
  const searchParams = useSearchParams();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<TradingTab>(() => {
    if (tabParam === 'market' || tabParam === 'bots') return tabParam;
    return 'center'; // default: live trading
  });

  useEffect(() => {
    if (tabParam === 'market' || tabParam === 'center' || tabParam === 'bots') {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Trading Center</h1>
        <p className="text-muted-foreground">
          Market data, live trading, and automated bots in one place
        </p>
      </div>
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as TradingTab)}
        className="w-full"
      >
        <div className="border-b bg-background/95 px-4 py-2 rounded-lg">
          <TabsList className="grid w-full max-w-2xl grid-cols-3">
            <TabsTrigger value="market" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Market
            </TabsTrigger>
            <TabsTrigger value="center" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Live Trading
            </TabsTrigger>
            <TabsTrigger value="bots" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Trading Bots
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="market" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
            <div className="min-w-0">
              <RealtimeContent />
            </div>
            <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px] flex flex-col gap-4 overflow-y-auto">
              <div className="min-h-[280px]">
                <AnalysisAgentInsightsPanel />
              </div>
              <div className="min-h-[200px]">
                <DataCollectorAgentPanel />
              </div>
              <div className="min-h-[200px]">
                <SentimentAgentPanel />
              </div>
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="center" className="mt-0">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4">
            <div className="min-w-0 -mx-4 -mb-6 xl:mx-0 xl:mb-0">
              <TradingCenterPage />
            </div>
            <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px]">
              <AnalysisAgentInsightsPanel />
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="bots" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
            <div className="min-w-0 -mx-4 xl:mx-0">
              <TradingBotsContent />
            </div>
            <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px] flex flex-col gap-4 overflow-y-auto">
              <div className="min-h-[280px]">
                <AnalysisAgentInsightsPanel />
              </div>
              <div className="min-h-[220px]">
                <StrategyAgentPanel />
              </div>
            </aside>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function TradingPage() {
  return (
    <Suspense fallback={<div className="p-6 text-muted-foreground">Loading Trading Center…</div>}>
      <TradingCenterContent />
    </Suspense>
  );
}
