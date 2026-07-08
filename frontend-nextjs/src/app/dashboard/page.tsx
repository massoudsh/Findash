'use client';

import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { PortfolioContent } from '@/components/portfolio/portfolio-content';
import { TradeTracker } from '@/components/portfolio/trade-tracker';
import { RiskGauge } from '@/components/dashboard/risk-gauge';
import { CreditScore } from '@/components/dashboard/credit-score';
import { IranMarketOverview } from '@/components/market/iran-market-overview';
import { AnalyticsOverview } from '@/components/analytics/analytics-overview';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useIranTicker } from '@/lib/hooks/use-iran-ticker';
import {
  Activity,
  ArrowDownLeft,
  ArrowUpLeft,
  BarChart3,
  Bell,
  Briefcase,
  CalendarClock,
  CheckCircle2,
  ChevronLeft,
  ClipboardList,
  CreditCard,
  Globe,
  Landmark,
  LineChart,
  MoreHorizontal,
  PieChart,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  WalletCards,
} from 'lucide-react';

type Tab = 'overview' | 'portfolio' | 'market' | 'trades' | 'analytics';
const VALID_TABS: Tab[] = ['overview', 'portfolio', 'market', 'trades', 'analytics'];

const TABS = [
  { value: 'overview', icon: BarChart3, label: 'نمای کلی' },
  { value: 'portfolio', icon: Briefcase, label: 'پرتفولیو' },
  { value: 'market', icon: Globe, label: 'بازار' },
  { value: 'trades', icon: ClipboardList, label: 'معاملات' },
  { value: 'analytics', icon: LineChart, label: 'تحلیل' },
] as const;

const stats = [
  {
    label: 'ارزش کل دارایی',
    value: '۱۲.۸۴ میلیارد',
    suffix: 'تومان',
    change: '+۱۸.۲٪',
    up: true,
    icon: WalletCards,
    accent: 'from-blue-500 to-sky-400',
  },
  {
    label: 'سود امروز',
    value: '۳۸۷.۵ میلیون',
    suffix: 'تومان',
    change: '+۴.۷٪',
    up: true,
    icon: TrendingUp,
    accent: 'from-emerald-500 to-teal-400',
  },
  {
    label: 'ریسک فعال',
    value: '۳۴',
    suffix: 'از ۱۰۰',
    change: '-۲.۱٪',
    up: true,
    icon: ShieldCheck,
    accent: 'from-indigo-500 to-blue-400',
  },
  {
    label: 'هشدارهای باز',
    value: '۹',
    suffix: 'مورد',
    change: '+۳ جدید',
    up: false,
    icon: Bell,
    accent: 'from-amber-500 to-orange-400',
  },
];

const allocation = [
  { label: 'کریپتو', value: 42, color: '#3B82F6' },
  { label: 'طلا و سکه', value: 28, color: '#F59E0B' },
  { label: 'ارز', value: 18, color: '#10B981' },
  { label: 'بورس', value: 12, color: '#8B5CF6' },
];

const performance = [46, 52, 49, 64, 58, 72, 68, 79, 76, 88, 83, 96];

const positions = [
  { asset: 'تتر', symbol: 'USDT-IRT', amount: '۳۲,۴۰۰', value: '۲.۰۸ میلیارد', pnl: '+۲.۴٪', up: true },
  { asset: 'طلای ۱۸ عیار', symbol: 'GOLD18-IRT', amount: '۱۴۸ گرم', value: '۱.۴۱ میلیارد', pnl: '+۵.۸٪', up: true },
  { asset: 'بیت‌کوین', symbol: 'BTC-IRT', amount: '۰.۰۸ BTC', value: '۱.۱۲ میلیارد', pnl: '-۱.۳٪', up: false },
  { asset: 'سکه تمام', symbol: 'COIN-IRT', amount: '۲۱ عدد', value: '۹۶۵ میلیون', pnl: '+۳.۱٪', up: true },
];

const activities = [
  { title: 'خرید تتر ثبت شد', meta: '۳۲,۴۰۰ USDT · نوبیتکس', time: '۵ دقیقه پیش', type: 'buy' },
  { title: 'هشدار ریسک فعال شد', meta: 'افت بیت‌کوین بیشتر از ۱.۲٪', time: '۱۸ دقیقه پیش', type: 'risk' },
  { title: 'پرداخت اشتراک تأیید شد', meta: 'زرین‌پال · پلن حرفه‌ای', time: '۱ ساعت پیش', type: 'payment' },
  { title: 'پرتفولیو rebalance شد', meta: 'کاهش وزن کریپتو به ۴۲٪', time: 'امروز ۱۰:۳۰', type: 'rebalance' },
];

function TickerItem({ t }: { t: ReturnType<typeof useIranTicker>['items'][number] }) {
  const [flash, setFlash] = useState<'up' | 'down' | null>(null);
  const prevPrice = useRef<number | null>(null);

  useEffect(() => {
    if (t.price === null) return;
    if (prevPrice.current !== null && prevPrice.current !== t.price) {
      setFlash(t.price > prevPrice.current ? 'up' : 'down');
      const id = setTimeout(() => setFlash(null), 800);
      prevPrice.current = t.price;
      return () => clearTimeout(id);
    }
    prevPrice.current = t.price;
  }, [t.price]);

  return (
    <div
      key={t.symbol}
      className={`flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 transition-colors duration-500 ${
        flash === 'up'
          ? 'border-emerald-400/40 bg-emerald-500/10'
          : flash === 'down'
          ? 'border-rose-400/40 bg-rose-500/10'
          : 'border-white/10 bg-white/[0.04]'
      }`}
    >
      <span className="text-[11px] text-slate-400">{t.label}</span>
      <span className="text-xs font-black text-white" dir="ltr">
        {t.price ? new Intl.NumberFormat('fa-IR').format(Math.round(t.price)) : '—'}
      </span>
      {t.change_pct !== null && (
        <span className={`flex items-center gap-0.5 text-[10px] font-bold ${t.up ? 'text-emerald-400' : 'text-rose-400'}`} dir="ltr">
          {t.up ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
          {t.up ? '+' : ''}{t.change_pct.toFixed(1)}٪
        </span>
      )}
    </div>
  );
}

function BlueTickerBar() {
  const { items, loading } = useIranTicker();

  return (
    <div className="flex items-center gap-2 overflow-x-auto scrollbar-none px-2 py-2.5 flex-nowrap">
      <div className="flex items-center gap-2 shrink-0 rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1.5 text-blue-300">
        <span className={`h-2 w-2 rounded-full ${loading ? 'bg-amber-400' : 'bg-blue-400 animate-pulse'}`} />
        <span className="text-[11px] font-bold">{loading ? 'در حال دریافت' : 'بازار زنده'}</span>
      </div>

      {loading
        ? Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-8 w-32 shrink-0 animate-pulse rounded-full bg-white/5" />
          ))
        : items.map((t) => <TickerItem key={t.symbol} t={t} />)}
    </div>
  );
}

function StatCard({ item }: { item: (typeof stats)[number] }) {
  const Icon = item.icon;
  return (
    <div className="group relative overflow-hidden rounded-[28px] border border-white/10 bg-white/[0.06] p-5 shadow-[0_24px_80px_rgba(15,23,42,0.28)] backdrop-blur-xl transition duration-300 hover:-translate-y-1 hover:border-blue-400/35 hover:bg-white/[0.08]">
      <div className={`absolute -left-8 -top-10 h-28 w-28 rounded-full bg-gradient-to-br ${item.accent} opacity-20 blur-2xl transition group-hover:opacity-35`} />
      <div className="relative flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-slate-400">{item.label}</p>
          <div className="mt-3 flex items-end gap-2">
            <h3 className="text-2xl font-black tracking-tight text-white">{item.value}</h3>
            <span className="pb-1 text-[11px] text-slate-400">{item.suffix}</span>
          </div>
        </div>
        <div className={`rounded-2xl bg-gradient-to-br ${item.accent} p-3 text-white shadow-lg shadow-blue-950/30`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <div className="relative mt-5 flex items-center justify-between">
        <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-bold ${item.up ? 'bg-emerald-500/10 text-emerald-300' : 'bg-amber-500/10 text-amber-300'}`}>
          {item.up ? <ArrowUpLeft className="h-3.5 w-3.5" /> : <ArrowDownLeft className="h-3.5 w-3.5" />}
          {item.change}
        </span>
        <span className="text-[11px] text-slate-500">نسبت به دیروز</span>
      </div>
    </div>
  );
}

function PerformanceChart() {
  const points = useMemo(() => {
    const max = Math.max(...performance);
    const min = Math.min(...performance);
    return performance
      .map((v, i) => {
        const x = (i / (performance.length - 1)) * 100;
        const y = 100 - ((v - min) / (max - min)) * 82 - 9;
        return `${x},${y}`;
      })
      .join(' ');
  }, []);

  return (
    <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-5 shadow-[0_24px_80px_rgba(15,23,42,0.25)] backdrop-blur-xl lg:col-span-2">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-black text-white">روند عملکرد پرتفولیو</h2>
          <p className="mt-1 text-xs text-slate-400">نمای ۱۲ ماهه با تمرکز روی بازار ایران</p>
        </div>
        <button className="rounded-full border border-white/10 bg-white/[0.04] p-2 text-slate-300 hover:border-blue-400/30 hover:text-blue-300">
          <MoreHorizontal className="h-5 w-5" />
        </button>
      </div>

      <div className="relative h-72 overflow-hidden rounded-[24px] border border-white/10 bg-gradient-to-br from-slate-950/80 to-blue-950/30 p-4">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(59,130,246,0.22),transparent_28%),radial-gradient(circle_at_80%_0%,rgba(14,165,233,0.14),transparent_32%)]" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="relative h-full w-full overflow-visible">
          {[20, 40, 60, 80].map((y) => (
            <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="rgba(148,163,184,0.14)" strokeDasharray="2 2" />
          ))}
          <defs>
            <linearGradient id="blueLine" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#38BDF8" />
              <stop offset="100%" stopColor="#3B82F6" />
            </linearGradient>
            <linearGradient id="blueFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(59,130,246,0.32)" />
              <stop offset="100%" stopColor="rgba(59,130,246,0)" />
            </linearGradient>
          </defs>
          <polygon points={`0,100 ${points} 100,100`} fill="url(#blueFill)" />
          <polyline points={points} fill="none" stroke="url(#blueLine)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
        </svg>
        <div className="absolute bottom-4 left-4 right-4 flex justify-between text-[10px] text-slate-500">
          {['فر', 'ار', 'خر', 'تیر', 'مر', 'شه'].map((m) => <span key={m}>{m}</span>)}
        </div>
      </div>
    </div>
  );
}

function AllocationWidget() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-5 shadow-[0_24px_80px_rgba(15,23,42,0.22)] backdrop-blur-xl">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-black text-white">ترکیب دارایی‌ها</h2>
          <p className="mt-1 text-xs text-slate-400">تخصیص سرمایه</p>
        </div>
        <PieChart className="h-5 w-5 text-blue-300" />
      </div>
      <div className="relative mx-auto mb-6 flex h-40 w-40 items-center justify-center rounded-full" style={{ background: `conic-gradient(${allocation.map((a, i) => `${a.color} ${allocation.slice(0, i).reduce((s, x) => s + x.value, 0)}% ${allocation.slice(0, i + 1).reduce((s, x) => s + x.value, 0)}%`).join(', ')})` }}>
        <div className="flex h-24 w-24 flex-col items-center justify-center rounded-full bg-slate-950 text-center shadow-inner">
          <span className="text-2xl font-black text-white">۴۲٪</span>
          <span className="text-[10px] text-slate-400">کریپتو</span>
        </div>
      </div>
      <div className="space-y-3">
        {allocation.map((a) => (
          <div key={a.label} className="flex items-center gap-3">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: a.color }} />
            <span className="flex-1 text-xs text-slate-300">{a.label}</span>
            <span className="text-xs font-black text-white">{a.value}٪</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PositionsTable() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-5 shadow-[0_24px_80px_rgba(15,23,42,0.22)] backdrop-blur-xl lg:col-span-2">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-black text-white">دارایی‌های مهم</h2>
          <p className="mt-1 text-xs text-slate-400">جدول موقعیت‌های فعال</p>
        </div>
        <button className="inline-flex items-center gap-1 rounded-full bg-blue-500/12 px-3 py-1.5 text-xs font-bold text-blue-300 hover:bg-blue-500/18">
          مشاهده کامل
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-right text-sm">
          <thead>
            <tr className="border-b border-white/10 text-xs text-slate-500">
              <th className="pb-3 font-medium">دارایی</th>
              <th className="pb-3 font-medium">مقدار</th>
              <th className="pb-3 font-medium">ارزش</th>
              <th className="pb-3 font-medium">سود/زیان</th>
              <th className="pb-3 font-medium">وضعیت</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {positions.map((p) => (
              <tr key={p.symbol} className="text-slate-300">
                <td className="py-4">
                  <div className="font-bold text-white">{p.asset}</div>
                  <div className="mt-0.5 text-[11px] text-slate-500" dir="ltr">{p.symbol}</div>
                </td>
                <td className="py-4 text-xs">{p.amount}</td>
                <td className="py-4 text-xs font-bold text-white">{p.value}</td>
                <td className={`py-4 text-xs font-black ${p.up ? 'text-emerald-400' : 'text-rose-400'}`}>{p.pnl}</td>
                <td className="py-4">
                  <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-bold ${p.up ? 'bg-emerald-500/10 text-emerald-300' : 'bg-rose-500/10 text-rose-300'}`}>
                    {p.up ? <CheckCircle2 className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                    {p.up ? 'مثبت' : 'نیازمند توجه'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ActivityTimeline() {
  const iconMap = {
    buy: TrendingUp,
    risk: ShieldCheck,
    payment: CreditCard,
    rebalance: Activity,
  } as const;

  return (
    <div className="rounded-[32px] border border-white/10 bg-white/[0.06] p-5 shadow-[0_24px_80px_rgba(15,23,42,0.22)] backdrop-blur-xl">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <h2 className="text-base font-black text-white">خط‌زمانی فعالیت</h2>
          <p className="mt-1 text-xs text-slate-400">آخرین رویدادهای حساب</p>
        </div>
        <CalendarClock className="h-5 w-5 text-blue-300" />
      </div>
      <div className="space-y-4">
        {activities.map((a, index) => {
          const Icon = iconMap[a.type as keyof typeof iconMap];
          return (
            <div key={a.title} className="relative flex gap-3">
              {index !== activities.length - 1 && <span className="absolute right-[17px] top-9 h-10 w-px bg-white/10" />}
              <div className="relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-blue-500/12 text-blue-300 ring-1 ring-blue-400/20">
                <Icon className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-bold text-white">{a.title}</p>
                  <span className="shrink-0 text-[10px] text-slate-500">{a.time}</span>
                </div>
                <p className="mt-1 text-xs text-slate-400">{a.meta}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function InsightCard() {
  return (
    <div className="relative overflow-hidden rounded-[32px] border border-blue-400/20 bg-gradient-to-br from-blue-600/25 via-blue-500/12 to-slate-900 p-5 shadow-[0_24px_80px_rgba(37,99,235,0.22)]">
      <div className="absolute -left-8 -top-8 h-28 w-28 rounded-full bg-blue-400/25 blur-2xl" />
      <div className="relative flex items-start gap-3">
        <div className="rounded-2xl bg-blue-400/20 p-3 text-blue-100 ring-1 ring-blue-200/20">
          <Sparkles className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-base font-black text-white">پیشنهاد هوشمند امروز</h2>
          <p className="mt-2 text-sm leading-7 text-blue-50/80">
            وزن کریپتو کمی بالاست. برای کاهش نوسان، ۶٪ از BTC-IRT را به طلا یا تتر منتقل کن.
          </p>
        </div>
      </div>
      <button className="relative mt-5 w-full rounded-2xl bg-blue-500 px-4 py-3 text-sm font-black text-white shadow-lg shadow-blue-950/30 transition hover:bg-blue-400">
        اجرای سناریو
      </button>
    </div>
  );
}

function OverviewDashboard({ riskValue }: { riskValue: number }) {
  return (
    <div className="space-y-5">
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((item) => <StatCard key={item.label} item={item} />)}
      </section>

      <section className="grid gap-5 lg:grid-cols-3">
        <PerformanceChart />
        <AllocationWidget />
      </section>

      <section className="grid gap-5 xl:grid-cols-[1fr_320px]">
        <div className="grid gap-5 lg:grid-cols-2">
          <PositionsTable />
          <div className="grid gap-5 lg:hidden">
            <InsightCard />
            <ActivityTimeline />
          </div>
        </div>
        <aside className="space-y-5">
          <RiskGauge value={riskValue} live className="border-white/10 bg-white/[0.06]" />
          <CreditScore score={724} className="border-white/10 bg-white/[0.06]" />
          <div className="hidden xl:block"><InsightCard /></div>
          <div className="hidden xl:block"><ActivityTimeline /></div>
        </aside>
      </section>
    </div>
  );
}

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
