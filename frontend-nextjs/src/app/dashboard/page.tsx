'use client';

import { useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense } from 'react';
import { DashboardContent } from '@/components/dashboard/dashboard-content';
import { PortfolioContent } from '@/components/portfolio/portfolio-content';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Card, CardContent, GlassCard } from '@/components/ui/card';
import { BarChart3, Briefcase, Bell, TrendingUp, Users } from 'lucide-react';

type DashboardTab = 'overview' | 'portfolio';

function DashboardPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<DashboardTab>(() => {
    if (tabParam === 'portfolio') return 'portfolio';
    return 'overview';
  });

  useEffect(() => {
    if (tabParam === 'overview' || tabParam === 'portfolio') setActiveTab(tabParam);
  }, [tabParam]);

  function handleTabChange(value: string) {
    const tab = value as DashboardTab;
    setActiveTab(tab);
    const params = new URLSearchParams(searchParams.toString());
    if (tab === 'overview') params.delete('tab');
    else params.set('tab', tab);
    const query = params.toString();
    router.replace(query ? `/dashboard?${query}` : '/dashboard', { scroll: false });
  }

  const currentTime = new Date().toLocaleString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'America/New_York',
  });

  return (
    <div className="min-h-screen bg-background">
      <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
        {/* Header: Command center + Overview/Portfolio tabs */}
        <div className="border-b border-white/30 dark:border-white/20 bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
          <div className="container mx-auto px-4 py-6">
            <h1 className="text-center text-3xl sm:text-4xl font-bold tracking-tight text-foreground antialiased font-serif mb-4 uppercase tracking-wider">
              Command center
            </h1>
            <div className="flex justify-center mb-4">
              <TabsList className="grid w-full max-w-md grid-cols-2">
                <TabsTrigger value="overview" className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Overview
                </TabsTrigger>
                <TabsTrigger value="portfolio" className="flex items-center gap-2">
                  <Briefcase className="h-4 w-4" />
                  Portfolio
                </TabsTrigger>
              </TabsList>
            </div>

            {/* Market Status Bar */}
            <GlassCard className="bg-gradient-to-r from-green-500/10 via-blue-500/10 to-green-500/10 border border-white/30 dark:border-white/20">
              <CardContent className="p-5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full animate-pulse shadow-lg shadow-green-500/50" />
                      <span className="text-sm font-semibold bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400 bg-clip-text text-transparent">
                        Market Open
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground font-medium">
                      {currentTime} EST
                    </div>
                    <div className="flex items-center gap-4 text-xs">
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-green-500/10 border border-white/30 dark:border-white/20">
                        <TrendingUp className="h-3 w-3 text-green-600" />
                        <span className="font-semibold">S&P 500: +0.8%</span>
                      </div>
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-green-500/10 border border-white/30 dark:border-white/20">
                        <TrendingUp className="h-3 w-3 text-green-600" />
                        <span className="font-semibold">NASDAQ: +1.2%</span>
                      </div>
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-blue-500/10 border border-white/30 dark:border-white/20">
                        <Users className="h-3 w-3 text-blue-600" />
                        <span className="font-semibold">247 Active Traders</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" className="hover:bg-primary/10">
                      <Bell className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </GlassCard>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-8">
          <TabsContent value="overview" className="mt-0">
            <DashboardContent />
          </TabsContent>
          <TabsContent value="portfolio" className="mt-0">
            <Suspense fallback={<div className="text-muted-foreground">Loading portfolio…</div>}>
              <PortfolioContent />
            </Suspense>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center text-muted-foreground">Loading…</div>}>
      <DashboardPageContent />
    </Suspense>
  );
} 