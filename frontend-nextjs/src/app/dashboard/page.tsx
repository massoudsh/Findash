'use client';

import { Suspense, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { PortfolioContent } from '@/components/portfolio/portfolio-content';
import { TradeTracker } from '@/components/portfolio/trade-tracker';
import { IranMarketOverview } from '@/components/market/iran-market-overview';
import { AnalyticsOverview } from '@/components/analytics/analytics-overview';
import { HelpCenter } from '@/components/help/help-center';
import { OverviewDashboard, BlueTickerBar } from '@/components/dashboard/overview-dashboard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  Briefcase,
  ClipboardList,
  Globe,
  HelpCircle,
  Landmark,
  LineChart,
} from 'lucide-react';

type Tab = 'overview' | 'portfolio' | 'market' | 'trades' | 'analytics' | 'help';
const VALID_TABS: Tab[] = ['overview', 'portfolio', 'market', 'trades', 'analytics', 'help'];

const TABS = [
  { value: 'overview',   icon: BarChart3,    label: 'نمای کلی' },
  { value: 'portfolio',  icon: Briefcase,    label: 'پرتفولیو' },
  { value: 'market',     icon: Globe,        label: 'بازار' },
  { value: 'trades',     icon: ClipboardList,label: 'معاملات' },
  { value: 'analytics',  icon: LineChart,    label: 'تحلیل' },
  { value: 'help',       icon: HelpCircle,   label: 'راهنما' },
] as const;


function DashboardPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const tabParam = searchParams.get('tab') as Tab | null;
  const [activeTab, setActiveTab] = useState<Tab>(() => (tabParam && VALID_TABS.includes(tabParam) ? tabParam : 'overview'));
  const [riskValue, setRiskValue] = useState(34);

  useEffect(() => {
    if (tabParam && VALID_TABS.includes(tabParam)) setActiveTab(tabParam);
  }, [tabParam]);

  useEffect(() => {
    const id = setInterval(() => {
      setRiskValue((v) => Math.min(95, Math.max(5, v + (Math.random() - 0.48) * 3)));
    }, 3000);
    return () => clearInterval(id);
  }, []);

  function handleTabChange(value: string) {
    const tab = value as Tab;
    setActiveTab(tab);
    const p = new URLSearchParams(searchParams.toString());
    if (tab === 'overview') p.delete('tab'); else p.set('tab', tab);
    router.replace(p.toString() ? `/dashboard?${p}` : '/dashboard', { scroll: false });
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#020617] text-slate-100">
      <div className="pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(circle_at_10%_0%,rgba(59,130,246,0.26),transparent_32%),radial-gradient(circle_at_90%_10%,rgba(14,165,233,0.16),transparent_30%),linear-gradient(180deg,rgba(15,23,42,0)_0%,rgba(2,6,23,1)_70%)]" />
      <div className="relative z-10 mx-auto max-w-[1500px] px-3 pb-10 pt-3 sm:px-5 lg:px-7">
        <div className="sticky top-0 z-30 -mx-3 border-b border-white/10 bg-[#020617]/80 px-3 backdrop-blur-2xl sm:-mx-5 sm:px-5 lg:-mx-7 lg:px-7">
          <BlueTickerBar />
        </div>

        <header className="py-6 sm:py-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1.5 text-xs font-bold text-blue-300">
                <Landmark className="h-3.5 w-3.5" />
                داشبورد مالی بازار ایران
              </div>
              <h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl lg:text-5xl">
                نمای یکپارچه سرمایه‌گذاری
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-400">
                رصد دارایی‌ها، ریسک، معاملات، جریان نقدی و فعالیت‌های مهم در یک صفحه مدرن، شبکه‌ای و موبایل‌محور.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button className="rounded-2xl border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm font-bold text-slate-200 transition hover:border-blue-400/30 hover:text-blue-300">
                گزارش امروز
              </button>
              <button className="rounded-2xl bg-[#3B82F6] px-4 py-2.5 text-sm font-black text-white shadow-lg shadow-blue-950/30 transition hover:bg-blue-400">
                افزودن دارایی
              </button>
            </div>
          </div>
        </header>

        <Tabs value={activeTab} onValueChange={handleTabChange}>
          <div className="mb-5 overflow-x-auto rounded-[24px] border border-white/10 bg-white/[0.05] p-1.5 backdrop-blur-xl">
            <TabsList className="flex h-12 min-w-[620px] gap-1 bg-transparent p-0">
              {TABS.map(({ value, icon: Icon, label }) => (
                <TabsTrigger
                  key={value}
                  value={value}
                  className="flex-1 rounded-[18px] text-xs font-black text-slate-400 transition data-[state=active]:bg-[#3B82F6] data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-blue-950/30"
                >
                  <Icon className="ml-1.5 h-4 w-4" />
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          <TabsContent value="overview" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <OverviewDashboard riskValue={riskValue} />
          </TabsContent>

          <TabsContent value="portfolio" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl">
              <Suspense fallback={<div className="p-8 text-center text-sm text-slate-400">در حال بارگذاری پرتفولیو…</div>}>
                <PortfolioContent />
              </Suspense>
            </div>
          </TabsContent>

          <TabsContent value="market" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl">
              <IranMarketOverview />
            </div>
          </TabsContent>

          <TabsContent value="trades" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl">
              <TradeTracker />
            </div>
          </TabsContent>

          <TabsContent value="analytics" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-4 backdrop-blur-xl">
              <AnalyticsOverview />
            </div>
          </TabsContent>

          <TabsContent value="help" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-300">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-5 backdrop-blur-xl">
              <HelpCenter />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-[#020617] text-sm text-slate-400">در حال بارگذاری داشبورد…</div>}>
      <DashboardPageContent />
    </Suspense>
  );
}
