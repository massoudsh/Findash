'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DataExplorer } from '@/components/data/data-explorer';
import { DataExport } from '@/components/ui/data-export';
import { DataCollectorAgentPanel } from '@/components/trading/data-collector-agent-panel';
import { VisualizationContent } from '@/components/visualization/visualization-content';
import { ChartShowcase } from '@/components/ui/chart-showcase';
import { AnalysisAgentInsightsPanel } from '@/components/trading/analysis-agent-insights';
import { ReportsContent } from '@/components/reports/reports-content';
import { LlmStatusBadge } from '@/components/reports/llm-status-badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Database, PieChart, Sparkles } from 'lucide-react';

type DataTab = 'explorer' | 'charts' | 'report';

export default function DataPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<DataTab>(() => {
    if (tabParam === 'charts' || tabParam === 'explorer' || tabParam === 'report') return tabParam;
    return 'explorer';
  });

  useEffect(() => {
    if (tabParam === 'charts' || tabParam === 'explorer' || tabParam === 'report') setActiveTab(tabParam);
  }, [tabParam]);

  function handleTabChange(value: string) {
    const tab = value as DataTab;
    setActiveTab(tab);
    const params = new URLSearchParams(searchParams.toString());
    if (tab === 'explorer') params.delete('tab');
    else params.set('tab', tab);
    const query = params.toString();
    router.replace(query ? `/data?${query}` : '/data', { scroll: false });
  }

  return (
    <div className="container mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">داده و نمودارها</h1>
        <p className="text-muted-foreground">
          داده‌های معاملاتی خود را کاوش، خروجی‌گیری و مصورسازی کنید
        </p>
      </div>
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full max-w-lg grid-cols-3">
          <TabsTrigger value="explorer" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            کاوشگر
          </TabsTrigger>
          <TabsTrigger value="charts" className="flex items-center gap-2">
            <PieChart className="h-4 w-4" />
            نمودارها
          </TabsTrigger>
          <TabsTrigger value="report" className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            گزارش هوش مصنوعی
          </TabsTrigger>
        </TabsList>
        <TabsContent value="explorer" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
            <div className="min-w-0">
              <div className="space-y-6">
                <DataExport
                  title="خروجی‌گیری از داده‌های معاملاتی"
                  filename="trading-data"
                  className="mb-6"
                />
                <DataExplorer />
              </div>
            </div>
            <aside className="hidden xl:block min-h-[360px]">
              <DataCollectorAgentPanel />
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="charts" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
            <div className="min-w-0">
              <Suspense fallback={<div className="text-center text-muted-foreground">در حال بارگذاری نمودارها...</div>}>
                <ChartShowcase />
                <VisualizationContent />
              </Suspense>
              <Card className="mt-6 bg-card border-border">
                <CardHeader>
                  <CardTitle>تحلیل معاملات</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">مجموع معاملات</h3>
                      <div className="text-2xl font-bold">۱,۲۴۷</div>
                      <p className="text-xs text-green-500">۱۲%+ این ماه</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">نرخ برد</h3>
                      <div className="text-2xl font-bold">۶۸.۳%</div>
                      <p className="text-xs text-green-500">۲.۱%+ این ماه</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">میانگین بازده</h3>
                      <div className="text-2xl font-bold">۲.۴%</div>
                      <p className="text-xs text-red-500">۰.۳%- این ماه</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">نسبت شارپ</h3>
                      <div className="text-2xl font-bold">۱.۸۵</div>
                      <p className="text-xs text-green-500">۰.۱۲+ این ماه</p>
                    </div>
                  </div>
                  <div className="mt-6 p-4 bg-muted/30 rounded-lg">
                    <h3 className="text-lg font-semibold mb-4">نمای کلی عملکرد</h3>
                    <div className="text-center text-muted-foreground border-2 border-dashed border-border rounded-lg h-48 flex items-center justify-center">
                      تحلیل معاملاتی بلادرنگ (اتصال به Grafana/TradingView)
                    </div>
                  </div>
                </CardContent>
              </Card>
              <button
                type="button"
                onClick={() => handleTabChange('report')}
                className="mt-4 w-full flex items-center justify-center gap-2 rounded-lg border border-dashed border-primary/40 bg-primary/5 px-4 py-3 text-sm font-medium text-primary hover:bg-primary/10 transition-colors"
              >
                <Sparkles className="h-4 w-4" />
                این داده‌ها را به گزارش هوش مصنوعی تبدیل کن
              </button>
            </div>
            <aside className="hidden xl:block min-h-[360px]">
              <AnalysisAgentInsightsPanel />
            </aside>
          </div>
        </TabsContent>
        <TabsContent value="report" className="mt-6">
          <div className="flex items-center justify-end mb-2">
            <LlmStatusBadge />
          </div>
          <Suspense fallback={<div className="text-center text-muted-foreground">در حال بارگذاری سیستم گزارش‌دهی...</div>}>
            <ReportsContent />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}
