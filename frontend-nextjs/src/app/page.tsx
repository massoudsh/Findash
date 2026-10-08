'use client';

import Link from 'next/link';
import {
  ArrowLeft,
  BarChart3,
  BellRing,
  ChevronDown,
  CreditCard,
  FileText,
  LineChart,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Wallet,
  Zap,
} from 'lucide-react';

const marketCards = [
  { title: 'شاخص کل', value: '۲,۱۸۶,۴۴۰', change: '+۰.۹٪', up: true },
  { title: 'دلار آزاد', value: '۶۱,۲۰۰', change: '-۰.۲٪', up: false },
  { title: 'طلا ۱۸ عیار', value: '۳,۴۵۰,۰۰۰', change: '+۱.۱٪', up: true },
];

const researchDirectory = [
  {
    title: 'بخش‌های بازار',
    items: [
      ['فناوری و سهام', '/technical'],
      ['مالی و اقتصاد', '/macro'],
      ['انرژی و کالا', '/fundamental-data'],
      ['سلامت و مصرف‌کننده', '/fundamental-data'],
      ['صنایع و مواد', '/data'],
      ['خدمات عمومی و املاک', '/macro'],
      ['ارتباطات', '/social'],
    ],
  },
  {
    title: 'صفحات کاربردی',
    items: [
      ['اقتصاد امروز', '/macro'],
      ['نمودارهای کلان', '/data'],
      ['ابزارک‌ها و داشبورد', '/dashboard'],
    ],
  },
  {
    title: 'پژوهش‌ها و تحلیل‌ها',
    items: [
      ['تحلیل بنیادی بازار', '/fundamental-data'],
      ['تحلیل تکنیکال', '/technical'],
      ['تحلیل اجتماعی و احساسات', '/social'],
      ['مدل‌های هوش مصنوعی', '/ai-models'],
    ],
  },
] as const;

export default function HomePage() {
  return (
    <main className="min-h-screen persian-pattern-bg">
      {/* Hero */}
      <section className="relative overflow-hidden px-4 py-10 sm:py-16">
        <div className="absolute inset-x-0 top-0 h-48 bg-gradient-to-b from-green-500/15 to-transparent pointer-events-none" />
        <div className="mx-auto max-w-6xl relative">
          <div className="grid gap-8 lg:grid-cols-[1fr_420px] items-center">
            <div className="text-center lg:text-right">
              <div className="inline-flex items-center gap-2 rounded-full border border-green-500/20 bg-green-500/10 px-4 py-1.5 text-xs text-green-400 mb-5">
                <Sparkles className="h-3.5 w-3.5" />
                پلتفرم هوشمند معاملاتی برای بازار ایران
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-[1.15] mb-5">
                اختاپوس؛
                <span className="block gradient-text">تصمیم‌گیری سریع‌تر، ریسک کمتر</span>
              </h1>

              <p className="text-base sm:text-lg text-muted-foreground leading-8 max-w-2xl mx-auto lg:mx-0 mb-8">
                داشبورد موبایل‌محور برای تحلیل بازار، مدیریت پرتفوی، امتیاز اعتباری معاملاتی و پایش ریسک زنده.
              </p>

              <div className="flex flex-col sm:flex-row gap-3 justify-center lg:justify-start">
                <Link href="/demo" className="btn-persian h-12 px-6 flex items-center justify-center gap-2 rounded-2xl">
                  مشاهده دمو بدون ثبت‌نام
                  <ArrowLeft className="h-4 w-4" />
                </Link>
                <Link href="/payment/checkout" className="h-12 px-6 flex items-center justify-center gap-2 rounded-2xl border border-green-500/25 bg-card/70 text-foreground hover:bg-green-500/10 transition-colors">
                  فعال‌سازی اشتراک
                  <CreditCard className="h-4 w-4" />
                </Link>
              </div>
            </div>

            {/* Phone mock */}
            <div className="relative mx-auto w-full max-w-sm">
              <div className="absolute -inset-6 rounded-[3rem] bg-green-500/20 blur-3xl" />
              <div className="relative rounded-[2.5rem] border border-green-500/25 bg-slate-950 p-3 shadow-2xl persian-border">
                <div className="rounded-[2rem] bg-background overflow-hidden border border-white/10">
                  <div className="h-8 flex items-center justify-center">
                    <div className="h-1.5 w-20 rounded-full bg-muted" />
                  </div>
                  <div className="p-4 space-y-4">
                    <div className="persian-card p-4">
                      <div className="flex items-center justify-between mb-4">
                        <span className="text-xs text-muted-foreground">ارزش پرتفوی</span>
                        <Wallet className="h-4 w-4 text-green-400" />
                      </div>
                      <div className="text-2xl font-black tabular-nums">۲,۸۴۷,۲۳۲</div>
                      <div className="text-xs text-green-400 mt-1 flex items-center gap-1">
                        <TrendingUp className="h-3 w-3" />
                        +۱۲.۴٪ این ماه
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-2xl border border-green-500/20 bg-green-500/10 p-3">
                        <ShieldCheck className="h-5 w-5 text-green-400 mb-2" />
                        <div className="text-xl font-bold">۳۴</div>
                        <div className="text-[10px] text-muted-foreground">ریسک زنده</div>
                      </div>
                      <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3">
                        <Zap className="h-5 w-5 text-amber-400 mb-2" />
                        <div className="text-xl font-bold">۷۱۲</div>
                        <div className="text-[10px] text-muted-foreground">امتیاز اعتباری</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {marketCards.map((m) => (
                        <div key={m.title} className="flex items-center justify-between rounded-xl bg-muted/50 p-3">
                          <span className="text-xs text-muted-foreground">{m.title}</span>
                          <div className="text-left">
                            <div className="text-sm font-bold tabular-nums">{m.value}</div>
                            <div className={m.up ? 'text-[10px] text-green-400' : 'text-[10px] text-red-400'}>{m.change}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Research directory inspired by the reference information architecture */}
      <section className="px-4 pb-16">
        <div className="mx-auto max-w-6xl rounded-[2rem] border border-border/80 bg-card/70 p-5 shadow-sm backdrop-blur sm:p-8">
          <div className="mb-7 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="mb-2 inline-flex items-center gap-2 text-xs font-semibold text-green-400">
                <FileText className="h-4 w-4" />
                فهرست بازار و پژوهش
              </div>
              <h2 className="text-2xl font-black sm:text-3xl">از هر زاویه‌ای بازار را بررسی کنید</h2>
              <p className="mt-2 max-w-2xl text-sm leading-7 text-muted-foreground">
                بخش‌ها، صفحات تحلیلی و پژوهش‌های Findash را در یک نمای ساده پیدا کنید.
              </p>
            </div>
            <Link href="/workflow" className="text-sm font-semibold text-green-400 hover:text-green-300">
              راهنمای جریان کار ←
            </Link>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {researchDirectory.map((group) => (
              <details key={group.title} open className="group rounded-2xl border border-border/70 bg-background/50 p-4">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 font-bold marker:hidden">
                  {group.title}
                  <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
                </summary>
                <nav aria-label={group.title} className="mt-4 space-y-1 border-t border-border/60 pt-3">
                  {group.items.map(([label, href]) => (
                    <Link key={`${label}-${href}`} href={href} className="block rounded-xl px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-green-500/10 hover:text-green-300">
                      {label}
                    </Link>
                  ))}
                </nav>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-4 pb-16">
        <div className="mx-auto max-w-6xl">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((f) => {
              const Icon = f.icon;
              return (
                <div key={f.title} className="persian-card persian-corner p-5 card-hover">
                  <div className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20">
                    <Icon className="h-5 w-5 text-green-400" />
                  </div>
                  <h3 className="font-bold mb-2">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-6">{f.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 pb-20">
        <div className="mx-auto max-w-4xl persian-card p-6 sm:p-8 text-center persian-border">
          <LineChart className="h-10 w-10 text-green-400 mx-auto mb-4" />
          <h2 className="text-2xl font-black mb-3">یک داشبورد تمیز برای تصمیم‌های جدی</h2>
          <p className="text-muted-foreground mb-6">از تحلیل تا مدیریت ریسک، همه‌چیز برای تجربه موبایل و بازار ایران بازطراحی شده است.</p>
          <Link href="/auth/signup" className="btn-persian inline-flex items-center gap-2 rounded-2xl">
            شروع کنید
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </main>
  );
}
