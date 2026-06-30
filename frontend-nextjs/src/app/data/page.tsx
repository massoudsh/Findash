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
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Database, PieChart } from 'lucide-react';

type DataTab = 'explorer' | 'charts';

export default function DataPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<DataTab>(() => {
    if (tabParam === 'charts' || tabParam === 'explorer') return tabParam;
    return 'explorer';
  });

  useEffect(() => {
    if (tabParam === 'charts' || tabParam === 'explorer') setActiveTab(tabParam);
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
        <h1 className="text-3xl font-bold tracking-tight">Data & Charts</h1>
        <p className="text-muted-foreground">
          Explore, export, and visualize your trading data
        </p>
      </div>
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="explorer" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Explorer
          </TabsTrigger>
          <TabsTrigger value="charts" className="flex items-center gap-2">
            <PieChart className="h-4 w-4" />
            Charts
          </TabsTrigger>
        </TabsList>
        <TabsContent value="explorer" className="mt-6">
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-6">
            <div className="min-w-0">
              <div className="space-y-6">
                <DataExport
                  title="Export Trading Data"
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
                  <CardTitle>Trading Analytics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Total Trades</h3>
                      <div className="text-2xl font-bold">1,247</div>
                      <p className="text-xs text-green-500">+12% this month</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Win Rate</h3>
                      <div className="text-2xl font-bold">68.3%</div>
                      <p className="text-xs text-green-500">+2.1% this month</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Avg. Return</h3>
                      <div className="text-2xl font-bold">2.4%</div>
                      <p className="text-xs text-red-500">-0.3% this month</p>
                    </div>
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Sharpe Ratio</h3>
                      <div className="text-2xl font-bold">1.85</div>
                      <p className="text-xs text-green-500">+0.12 this month</p>
                    </div>
                  </div>
                  <div className="mt-6 p-4 bg-muted/30 rounded-lg">
                    <h3 className="text-lg font-semibold mb-4">Performance Overview</h3>
                    <div className="text-center text-muted-foreground border-2 border-dashed border-border rounded-lg h-48 flex items-center justify-center">
                      Real-time trading analytics (Grafana/TradingView integration)
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            <aside className="hidden xl:block min-h-[360px]">
              <AnalysisAgentInsightsPanel />
            </aside>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
