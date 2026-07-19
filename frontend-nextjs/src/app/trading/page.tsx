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
import { Brain, DollarSign, Shield, Target, FlaskConical, Activity, LineChart, Database, MessageSquare, Cpu } from 'lucide-react';
import { ErrorBoundary } from '@/components/error-boundary';

const OptionsPageContent = lazy(() =>
  import('@/app/options/options-page-content').then((m) => ({ default: m.OptionsPageContent }))
);
const TechnicalPageContent = lazy(() => import('@/app/technical/page'));
const MacroPageContent = lazy(() => import('@/app/macro/page'));
const FundamentalPageContent = lazy(() => import('@/app/fundamental-data/page'));
const OnChainPageContent = lazy(() => import('@/app/on-chain/page'));
const SocialPageContent = lazy(() => import('@/app/social/page'));
const AIModelsPageContent = lazy(() => import('@/app/ai-models/page'));

type TradingTab = 'bots' | 'options' | 'risk' | 'strategies' | 'analysis';
type StrategiesSubTab = 'strategies' | 'backtesting';
type AnalysisSubTab = 'technical' | 'macro' | 'fundamental' | 'onchain' | 'social' | 'ai-models';
const ANALYSIS_SUBTABS: AnalysisSubTab[] = ['technical', 'macro', 'fundamental', 'onchain', 'social', 'ai-models'];

function TradingCenterContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const subtabParam = searchParams.get('subtab');
  const [activeTab, setActiveTab] = useState<TradingTab>(() => {
    if (tabParam === 'bots' || tabParam === 'options' || tabParam === 'risk' || tabParam === 'strategies' || tabParam === 'analysis') return tabParam;
    return 'options'; // default landing: Options with decision tools
  });
  const [strategiesSubTab, setStrategiesSubTab] = useState<StrategiesSubTab>(() => {
    if (subtabParam === 'backtesting') return 'backtesting';
    return 'strategies';
  });
  const [analysisSubTab, setAnalysisSubTab] = useState<AnalysisSubTab>(() => {
    if (subtabParam && (ANALYSIS_SUBTABS as string[]).includes(subtabParam)) return subtabParam as AnalysisSubTab;
    return 'technical';
  });

  useEffect(() => {
    if (tabParam === 'bots' || tabParam === 'options' || tabParam === 'risk' || tabParam === 'strategies' || tabParam === 'analysis') {
      setActiveTab(tabParam);
    }
  }, [tabParam]);

  useEffect(() => {
    if (subtabParam === 'backtesting' || subtabParam === 'strategies') {
      setStrategiesSubTab(subtabParam);
    }
  }, [subtabParam]);

  useEffect(() => {
    if (subtabParam && (ANALYSIS_SUBTABS as string[]).includes(subtabParam)) {
      setAnalysisSubTab(subtabParam as AnalysisSubTab);
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
      if (tab !== 'strategies' && tab !== 'analysis') params.delete('subtab');
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

  function handleAnalysisSubTabChange(value: string) {
    const subtab = value as AnalysisSubTab;
    setAnalysisSubTab(subtab);
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', 'analysis');
    if (subtab === 'technical') params.delete('subtab');
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
          <TabsList className="grid w-full max-w-3xl grid-cols-5">
            <TabsTrigger value="options" className="flex items-center gap-2">
              <DollarSign className="h-4 w-4" />
              اختیار معامله
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              تحلیل بازار
            </TabsTrigger>
            <TabsTrigger value="strategies" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              استراتژی‌ها
            </TabsTrigger>
            <TabsTrigger value="risk" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              ریسک
            </TabsTrigger>
            <TabsTrigger value="bots" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              ربات‌های معاملاتی
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
        <TabsContent value="analysis" className="mt-6">
          <Tabs value={analysisSubTab} onValueChange={handleAnalysisSubTabChange} className="w-full">
            <TabsList className="grid w-full max-w-3xl grid-cols-3 md:grid-cols-6 mb-4">
              <TabsTrigger value="technical" className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                تکنیکال
              </TabsTrigger>
              <TabsTrigger value="macro" className="flex items-center gap-2">
                <LineChart className="h-4 w-4" />
                کلان
              </TabsTrigger>
              <TabsTrigger value="fundamental" className="flex items-center gap-2">
                <Brain className="h-4 w-4" />
                بنیادی
              </TabsTrigger>
              <TabsTrigger value="onchain" className="flex items-center gap-2">
                <Database className="h-4 w-4" />
                آن‌چین
              </TabsTrigger>
              <TabsTrigger value="social" className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                اجتماعی
              </TabsTrigger>
              <TabsTrigger value="ai-models" className="flex items-center gap-2">
                <Cpu className="h-4 w-4" />
                مدل‌های AI
              </TabsTrigger>
            </TabsList>
            <TabsContent value="technical" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری تحلیل تکنیکال...</div>}>
                  <TechnicalPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="macro" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری تحلیل کلان...</div>}>
                  <MacroPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="fundamental" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری تحلیل بنیادی...</div>}>
                  <FundamentalPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="onchain" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری داده‌های آن‌چین...</div>}>
                  <OnChainPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="social" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری سیگنال‌های اجتماعی...</div>}>
                  <SocialPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="ai-models" className="mt-0">
              <ErrorBoundary>
                <Suspense fallback={<div className="p-6 text-muted-foreground animate-pulse">در حال بارگذاری مدل‌های هوش مصنوعی...</div>}>
                  <AIModelsPageContent />
                </Suspense>
              </ErrorBoundary>
            </TabsContent>
          </Tabs>
        </TabsContent>
        <TabsContent value="strategies" className="mt-6">
          <Tabs value={strategiesSubTab} onValueChange={handleStrategiesSubTabChange} className="w-full">
            <TabsList className="grid w-full max-w-md grid-cols-2 mb-4">
              <TabsTrigger value="strategies" className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                استراتژی‌ها
              </TabsTrigger>
              <TabsTrigger value="backtesting" className="flex items-center gap-2">
                <FlaskConical className="h-4 w-4" />
                بک‌تست
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
