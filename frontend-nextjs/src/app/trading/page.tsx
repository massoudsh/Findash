'use client';

import { useState, useEffect, Suspense, lazy } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TradingBotsContent } from '@/components/trading/trading-bots-content';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { StrategyAgentPanel } from '@/components/trading/strategy-agent-panel';
import { RiskContent } from '@/components/risk/risk-content';
import { RiskAgentPanel } from '@/components/agents/risk-agent-panel';
import { StrategiesContent } from '@/components/strategies/strategies-content';
import { BacktestRunner } from '@/components/backtesting/backtest-runner';
import { BacktestAgentPanel } from '@/components/agents/backtest-agent-panel';
import { Brain, DollarSign, Shield, Target, FlaskConical } from 'lucide-react';
import { ErrorBoundary } from '@/components/error-boundary';

const OptionsPageContent = lazy(() =>
  import('@/app/options/options-page-content').then((m) => ({ default: m.OptionsPageContent }))
);

type TradingTab = 'bots' | 'options' | 'risk' | 'strategies';
type StrategiesSubTab = 'strategies' | 'backtesting';

function TradingCenterContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const subtabParam = searchParams.get('subtab');
  const [activeTab, setActiveTab] = useState<TradingTab>(() => {
    if (tabParam === 'bots' || tabParam === 'options' || tabParam === 'risk' || tabParam === 'strategies') return tabParam;
    return 'options'; // default landing: Options with decision tools
  });
  const [strategiesSubTab, setStrategiesSubTab] = useState<StrategiesSubTab>(() => {
    if (subtabParam === 'backtesting') return 'backtesting';
    return 'strategies';
  });

  useEffect(() => {
    if (tabParam === 'bots' || tabParam === 'options' || tabParam === 'risk' || tabParam === 'strategies') {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  useEffect(() => {
    if (subtabParam === 'backtesting' || subtabParam === 'strategies') {
      setStrategiesSubTab(subtabParam);
    }
  }, [subtabParam]);

  function handleTabChange(value: string) {
    const tab = value as TradingTab;
    setActiveTab(tab);
    const params = new URLSearchParams(searchParams.toString());
    if (tab === 'options') {
      params.delete('tab');
      params.delete('subtab');
    } else {
      params.set('tab', tab);
      if (tab !== 'strategies') params.delete('subtab');
    }
    const query = params.toString();
    router.replace(query ? `/trading?${query}` : '/trading', { scroll: false });
  }

  function handleStrategiesSubTabChange(value: string) {
    const subtab = value as StrategiesSubTab;
    setStrategiesSubTab(subtab);
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', 'strategies');
    if (subtab === 'strategies') params.delete('subtab');
    else params.set('subtab', subtab);
    router.replace(`/trading?${params.toString()}`, { scroll: false });
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">مرکز فرماندهی</h1>
        <p className="text-muted-foreground">
          اختیار معامله، استراتژی‌ها، مدیریت ریسک و ربات‌های معاملاتی در یک بخش واحد
        </p>
      </div>
      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="w-full"
      >
        <div className="border-b bg-background/95 px-4 py-2 rounded-lg">
          <TabsList className="grid w-full max-w-2xl grid-cols-4">
            <TabsTrigger value="options" className="flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              Options
            </TabsTrigger>
            <TabsTrigger value="strategies" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              Strategies
            </TabsTrigger>
            <TabsTrigger value="risk" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Risk
            </TabsTrigger>
            <TabsTrigger value="bots" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Trading Bots
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="options" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
            <div className="min-w-0 -mx-4 xl:mx-0 min-h-[480px] flex flex-col">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">Loading options…</div>}>
                  <OptionsPageContent />
                </Suspense>
              </ErrorBoundary>
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
        <TabsContent value="strategies" className="mt-6">
          <Tabs value={strategiesSubTab} onValueChange={handleStrategiesSubTabChange} className="w-full">
            <TabsList className="grid w-full max-w-md grid-cols-2 mb-4">
              <TabsTrigger value="strategies" className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                Strategies
              </TabsTrigger>
              <TabsTrigger value="backtesting" className="flex items-center gap-2">
                <FlaskConical className="h-4 w-4" />
                Backtesting
              </TabsTrigger>
            </TabsList>
            <TabsContent value="strategies" className="mt-0">
              <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
                <div className="min-w-0">
                  <Suspense fallback={<div className="text-center">در حال بارگذاری استراتژی‌ها...</div>}>
                    <StrategiesContent />
                  </Suspense>
                </div>
                <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px] flex flex-col overflow-y-auto">
                  <StrategyAgentPanel />
                </aside>
              </div>
            </TabsContent>
            <TabsContent value="backtesting" className="mt-0">
              <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
                <div className="min-w-0">
                  <Suspense fallback={<div>در حال بارگذاری بک‌تست...</div>}>
                    <BacktestRunner />
                  </Suspense>
                </div>
                <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px] flex flex-col overflow-y-auto">
                  <BacktestAgentPanel />
                </aside>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>
        <TabsContent value="risk" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4 min-h-[480px]">
            <div className="min-w-0">
              <Suspense fallback={<div>در حال بارگذاری تحلیل ریسک...</div>}>
                <RiskContent />
              </Suspense>
            </div>
            <aside className="h-[calc(100vh-14rem)] min-h-[420px] xl:sticky xl:top-6 max-xl:mt-4 max-xl:max-h-[420px]">
              <RiskAgentPanel />
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
    <Suspense fallback={<div className="p-6 text-muted-foreground">در حال بارگذاری مرکز فرمان…</div>}>
      <TradingCenterContent />
    </Suspense>
  );
}
