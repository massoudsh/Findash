'use client';

import Link from 'next/link';
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  ShieldCheck,
  Sparkles,
  Target,
  UserRound,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { WorkflowFlowChart } from '@/components/workflow/workflow-flow-chart';

const stages = [
  {
    number: '۰۱',
    title: 'داده را یک‌جا ببینید',
    description: 'قیمت‌ها، خبرها و داده‌های تکمیلی دریافت، اعتبارسنجی و به‌روز می‌شوند.',
    icon: Database,
    accent: 'text-sky-600 dark:text-sky-400',
    surface: 'bg-sky-500/10',
    agents: ['Nexus · M1', 'Vault · M2', 'Pulse · M3', 'Echo · M9'],
  },
  {
    number: '۰۲',
    title: 'تحلیل و ریسک را بررسی کنید',
    description: 'عامل‌ها داده را به پیش‌بینی، سناریو، سیگنال و محدودیت‌های ریسک تبدیل می‌کنند.',
    icon: Brain,
    accent: 'text-violet-600 dark:text-violet-400',
    surface: 'bg-violet-500/10',
    agents: ['Neuron · M5', 'Oracle · M7', 'Atlas · M4', 'Guardian · M6'],
  },
  {
    number: '۰۳',
    title: 'تصمیم نهایی با شماست',
    description: 'پیشنهادها و ریسک را در مرکز فرماندهی مقایسه کنید، سپس تأیید، رد یا اصلاح کنید.',
    icon: UserRound,
    accent: 'text-amber-600 dark:text-amber-400',
    surface: 'bg-amber-500/10',
    agents: ['مرکز فرماندهی', 'تأیید یا اصلاح شما'],
  },
  {
    number: '۰۴',
    title: 'اجرا و نتیجه را دنبال کنید',
    description: 'اجرای آزمایشی یا زنده، بک‌تست و گزارش‌ها به شما کمک می‌کنند نتیجه را بسنجید.',
    icon: BarChart3,
    accent: 'text-emerald-600 dark:text-emerald-400',
    surface: 'bg-emerald-500/10',
    agents: ['Shadow · M8', 'Chronicle · M10', 'Lens · M11'],
  },
];

const agentGroups = [
  {
    title: 'داده و پایش بازار',
    description: 'دریافت، ذخیره، پخش زنده و سنجش احساسات بازار',
    items: ['Nexus M1 — دریافت داده', 'Vault M2 — اعتبارسنجی و ذخیره', 'Pulse M3 — پخش زنده', 'Echo M9 — تحلیل احساسات'],
    color: 'border-sky-500/25 bg-sky-500/5',
  },
  {
    title: 'تحلیل و مدیریت ریسک',
    description: 'تبدیل داده به پیش‌بینی، سیگنال و محدودهٔ امن تصمیم',
    items: ['Neuron M5 — مدل‌های پیش‌بینی', 'Oracle M7 — سناریوی قیمت', 'Atlas M4 — ترکیب سیگنال‌ها', 'Guardian M6 — ریسک و حجم مجاز'],
    color: 'border-violet-500/25 bg-violet-500/5',
  },
  {
    title: 'اجرا و گزارش‌دهی',
    description: 'اجرای سفارش، اعتبارسنجی ایده و نمایش نتیجه',
    items: ['Shadow M8 — اجرای آزمایشی یا زنده', 'Chronicle M10 — بک‌تست', 'Lens M11 — گزارش و تجسم'],
    color: 'border-emerald-500/25 bg-emerald-500/5',
  },
];

export default function WorkflowPage() {
  return (
    <main dir="rtl" className="mx-auto max-w-6xl space-y-10 px-4 py-8 sm:px-6 lg:py-12">
      <section className="relative overflow-hidden rounded-3xl border border-primary/15 bg-gradient-to-bl from-primary/15 via-card to-card px-6 py-10 sm:px-10 sm:py-14">
        <div className="absolute -left-20 -top-24 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
        <div className="relative max-w-3xl">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-background/70 px-3 py-1.5 text-sm font-medium text-primary">
            <Sparkles className="h-4 w-4" />
            راهنمای جریان کار اختاپوس
          </div>
          <h1 className="text-3xl font-bold leading-tight tracking-tight text-foreground sm:text-5xl">
            داده را به تصمیمی آگاهانه تبدیل کنید.
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-8 text-muted-foreground sm:text-lg">
            اختاپوس اطلاعات بازار را جمع‌آوری و تحلیل می‌کند، ریسک را شفاف نشان می‌دهد و نتیجه را گزارش می‌کند؛ اما کنترل و تصمیم نهایی همیشه با شماست.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild size="lg" className="gap-2">
              <Link href="/trading">
                رفتن به مرکز فرماندهی
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/dashboard">مشاهدهٔ داشبورد</Link>
            </Button>
          </div>
        </div>
      </section>

      <section aria-labelledby="workflow-steps-title">
        <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-medium text-primary">چهار مرحلهٔ روشن</p>
            <h2 id="workflow-steps-title" className="mt-1 text-2xl font-bold">از بازار تا گزارش</h2>
          </div>
          <p className="text-sm text-muted-foreground">هر مرحله خروجی مشخصی برای مرحلهٔ بعدی دارد.</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {stages.map(({ number, title, description, icon: Icon, accent, surface, agents }) => (
            <article key={number} className="relative rounded-2xl border border-border bg-card p-5 shadow-sm transition-shadow hover:shadow-md">
              <span className="text-sm font-bold text-muted-foreground">{number}</span>
              <div className={`mt-4 flex h-11 w-11 items-center justify-center rounded-xl ${surface} ${accent}`}>
                <Icon className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-lg font-bold">{title}</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">{description}</p>
              <div className="mt-5 flex flex-wrap gap-1.5">
                {agents.map((agent) => (
                  <span key={agent} className="rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground">{agent}</span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]" aria-labelledby="flow-title">
        <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
          <div className="border-b border-border px-5 py-5 sm:px-6">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-primary/10 p-2 text-primary"><Activity className="h-5 w-5" /></div>
              <div>
                <h2 id="flow-title" className="text-lg font-bold">نمای کامل جریان داده</h2>
                <p className="mt-1 text-sm text-muted-foreground">برای بررسی جزئیات، جابه‌جایی یا بزرگ‌نمایی کنید.</p>
              </div>
            </div>
          </div>
          <div className="p-3 sm:p-5"><WorkflowFlowChart /></div>
        </div>

        <aside className="rounded-2xl border border-amber-500/25 bg-amber-500/5 p-6">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-500/15 text-amber-700 dark:text-amber-400"><UserRound className="h-5 w-5" /></div>
          <h2 className="mt-5 text-xl font-bold">شما در حلقهٔ تصمیم هستید</h2>
          <p className="mt-3 text-sm leading-7 text-muted-foreground">
            عامل‌ها پیشنهاد می‌دهند، اما هیچ سفارش یا تغییری بدون انتخاب شما نباید مبنای تصمیم‌گیری باشد.
          </p>
          <ul className="mt-6 space-y-3 text-sm">
            {['سیگنال و ریسک را کنار هم مقایسه کنید', 'حجم و محدودیت‌های پیشنهادی را بازبینی کنید', 'تأیید، اصلاح یا رد کنید'].map((item) => (
              <li key={item} className="flex items-start gap-2"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />{item}</li>
            ))}
          </ul>
        </aside>
      </section>

      <section className="rounded-2xl border border-border bg-card p-5 sm:p-7" aria-labelledby="agents-title">
        <div className="mb-6">
          <p className="text-sm font-medium text-primary">مرجع سریع</p>
          <h2 id="agents-title" className="mt-1 text-2xl font-bold">عامل‌ها چه می‌کنند؟</h2>
        </div>
        <div className="grid gap-4 lg:grid-cols-3">
          {agentGroups.map((group) => (
            <article key={group.title} className={`rounded-xl border p-5 ${group.color}`}>
              <h3 className="font-bold">{group.title}</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">{group.description}</p>
              <ul className="mt-5 space-y-2.5 text-sm">
                {group.items.map((item) => <li key={item} className="flex gap-2"><Zap className="mt-0.5 h-4 w-4 shrink-0 text-primary" />{item}</li>)}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-5 rounded-2xl border border-primary/20 bg-primary/5 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-6 w-6 shrink-0 text-primary" />
          <div><h2 className="font-bold">تصمیم‌گیری آگاهانه، نه سیگنال‌فروشی</h2><p className="mt-1 text-sm leading-6 text-muted-foreground">این ابزار برای شفاف‌سازی داده و ریسک است و توصیهٔ خرید یا فروش محسوب نمی‌شود.</p></div>
        </div>
        <Button asChild variant="outline" className="shrink-0"><Link href="/reports">مشاهدهٔ گزارش‌ها</Link></Button>
      </section>
    </main>
  );
}
