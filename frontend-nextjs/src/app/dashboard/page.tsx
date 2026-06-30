'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { DashboardContent } from '@/components/dashboard/dashboard-content';
import { PortfolioContent } from '@/components/portfolio/portfolio-content';
import { TradeTracker } from '@/components/portfolio/trade-tracker';
import { RiskGauge } from '@/components/dashboard/risk-gauge';
import { CreditScore } from '@/components/dashboard/credit-score';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  Briefcase,
  ClipboardList,
  Globe,
  LineChart,
  TrendingUp,
  TrendingDown,
} from 'lucide-react';

// ── Types ────────────────────────────────────────────────────────────────
type Tab = 'overview' | 'portfolio' | 'market' | 'trades' | 'analytics';
const VALID_TABS: Tab[] = ['overview', 'portfolio', 'market', 'trades', 'analytics'];

const TABS = [
  { value: 'overview',   icon: BarChart3,     label: 'نمای کلی' },
  { value: 'portfolio',  icon: Briefcase,     label: 'پرتفولیو' },
  { value: 'market',     icon: Globe,         label: 'بازار' },
  { value: 'trades',     icon: ClipboardList, label: 'معاملات' },
  { value: 'analytics',  icon: LineChart,     label: 'تحلیل' },
] as const;

// ── Live ticker data ──────────────────────────────────────────────────────
const TICKERS = [
  { label: 'بیت‌کوین',   value: '۶۷,۴۲۰',   change: '+۱.۸٪', up: true },
  { label: 'طلا',        value: '$۲,۳۴۵',    change: '+۰.۴٪', up: true },
  { label: 'دلار',       value: '۶۱,۲۰۰',   change: '-۰.۲٪', up: false },
  { label: 'شاخص کل',   value: '۲,۱۸۶,۴۴۰', change: '+۰.۹٪', up: true },
  { label: 'نفت برنت',   value: '$۸۳.۲',    change: '-۰.۶٪', up: false },
];

// ── Lazy market placeholder ───────────────────────────────────────────────
function MarketPlaceholder() {
  return (
    <div className="rounded-2xl border border-dashed border-border flex items-center justify-center h-64 text-muted-foreground text-sm">
      داده‌های بازار به‌زودی
    </div>
  );
}

function AnalyticsPlaceholder() {
  return (
    <div className="rounded-2xl border border-dashed border-border flex items-center justify-center h-64 text-muted-foreground text-sm">
      تحلیل پیشرفته به‌زودی
    </div>
  );
}

// ── Ticker bar ────────────────────────────────────────────────────────────
function TickerBar() {
  return (
    <div className="flex items-center gap-1 overflow-x-auto scrollbar-none px-1 py-2 flex-nowrap">
      {/* Live dot */}
      <div className="flex items-center gap-1.5 shrink-0 ml-2">
        <span className="h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
        <span className="text-[10px] text-green-400 font-medium">زنده</span>
      </div>
      <div className="h-3 w-px bg-border mx-1 shrink-0" />
      {TICKERS.map((t) => (
        <div key={t.label} className="flex items-center gap-1.5 shrink-0 px-2.5 py-1 rounded-lg bg-muted/50 border border-border/50">
          <span className="text-[10px] text-muted-foreground">{t.label}</span>
          <span className="text-[11px] font-bold tabular-nums text-foreground" dir="ltr">{t.value}</span>
          <span className={['text-[10px] font-semibold flex items-center gap-0.5', t.up ? 'text-green-400' : 'text-red-400'].join(' ')} dir="ltr">
            {t.up ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
            {t.change}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────
function DashboardPageContent() {
  const router      = useRouter();
  const searchParams = useSearchParams();
  const tabParam    = searchParams.get('tab') as Tab | null;

  const [activeTab, setActiveTab] = useState<Tab>(() =>
    tabParam && VALID_TABS.includes(tabParam) ? tabParam : 'overview'
  );
  const [riskValue, setRiskValue] = useState(34);

  // Keep tab in URL
  useEffect(() => {
    if (tabParam && VALID_TABS.includes(tabParam)) setActiveTab(tabParam);
  }, [tabParam]);

  // Simulate slow drift in risk value
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
    <div className="min-h-screen">
      {/* ── Sticky tab bar ───────────────────────────────────────────── */}
      <div className="sticky top-0 z-30 bg-background/90 backdrop-blur-xl border-b border-border/50">
        {/* Ticker */}
        <TickerBar />

        {/* Tab list */}
        <div className="px-2 pb-2">
          <Tabs value={activeTab} onValueChange={handleTabChange}>
            <TabsList className="flex w-full h-10 bg-muted/40 border border-border/50 rounded-xl p-1 gap-0.5">
              {TABS.map(({ value, icon: Icon, label }) => (
                <TabsTrigger
                  key={value}
                  value={value}
                  className="flex-1 flex items-center justify-center gap-1.5 text-xs font-medium rounded-lg
                             data-[state=active]:bg-card data-[state=active]:text-primary
                             data-[state=active]:shadow-[0_0_0_1px_hsl(var(--primary)/0.3),0_2px_8px_rgba(34,197,94,0.12)]
                             data-[state=inactive]:text-muted-foreground transition-all duration-200"
                >
                  <Icon className="h-3.5 w-3.5 shrink-0" />
                  <span className="hidden sm:inline">{label}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {/* ── Tab contents ─────────────────────────────────────── */}
            <div className="pt-4">

              {/* Overview — 3-column layout on large screens */}
              <TabsContent value="overview" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_280px] gap-5">
                  {/* Main content */}
                  <div className="min-w-0">
                    <DashboardContent />
                  </div>

                  {/* Right sidebar — Risk Gauge + Credit Score */}
                  <div className="flex flex-col gap-4">
                    <RiskGauge value={riskValue} live />
                    <CreditScore score={712} />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="portfolio" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_280px] gap-5">
                  <div className="min-w-0">
                    <Suspense fallback={<div className="text-muted-foreground text-sm">در حال بارگذاری…</div>}>
                      <PortfolioContent />
                    </Suspense>
                  </div>
                  <div className="flex flex-col gap-4">
                    <RiskGauge value={riskValue} live />
                    <CreditScore score={712} />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="market" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
                <MarketPlaceholder />
              </TabsContent>

              <TabsContent value="trades" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
                <TradeTracker />
              </TabsContent>

              <TabsContent value="analytics" className="mt-0 animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
                <AnalyticsPlaceholder />
              </TabsContent>

            </div>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center text-muted-foreground text-sm">
        در حال بارگذاری داشبورد…
      </div>
    }>
      <DashboardPageContent />
    </Suspense>
  );
}
