'use client';

import { useState } from 'react';
import {
  BookOpen,
  ChevronLeft,
  HelpCircle,
  Lightbulb,
  Play,
  PlayCircle,
  Rocket,
  Search,
  TrendingUp,
  Shield,
  BarChart3,
  Wallet,
  Bell,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────
type Category = 'getting-started' | 'tutorials' | 'faq' | 'tips';

interface VideoCard {
  id: string;
  title: string;
  duration: string;
  thumbnail?: string;
  url?: string;
  tag: string;
}

interface FaqItem {
  q: string;
  a: string;
}

// ─── Data ─────────────────────────────────────────────────────────────────────
const CATEGORIES = [
  { id: 'getting-started' as Category, icon: Rocket,     label: 'شروع به کار' },
  { id: 'tutorials'       as Category, icon: BookOpen,   label: 'آموزش‌ها' },
  { id: 'faq'             as Category, icon: HelpCircle, label: 'سوالات متکرر' },
  { id: 'tips'            as Category, icon: Lightbulb,  label: 'نکات مفید' },
];

const VIDEOS: VideoCard[] = [
  { id: 'v1', title: 'ورود و راه‌اندازی حساب',               duration: '۳:۲۰', tag: 'شروع به کار', url: '#' },
  { id: 'v2', title: 'آشنایی با داشبورد اصلی',              duration: '۵:۴۵', tag: 'شروع به کار', url: '#' },
  { id: 'v3', title: 'افزودن دارایی به پرتفولیو',            duration: '۴:۱۰', tag: 'پرتفولیو',    url: '#' },
  { id: 'v4', title: 'خواندن تیکر بازار زنده',              duration: '۲:۳۰', tag: 'بازار',       url: '#' },
  { id: 'v5', title: 'استراتژی کاورد کال — مقدماتی',        duration: '۸:۵۵', tag: 'آپشن',        url: '#' },
  { id: 'v6', title: 'کَش‌سکیورد پوت — خرید ارزان‌تر سهام', duration: '۷:۱۵', tag: 'آپشن',        url: '#' },
  { id: 'v7', title: 'کالر — بیمه رایگان برای سهام',        duration: '۹:۴۰', tag: 'آپشن',        url: '#' },
  { id: 'v8', title: 'مدیریت ریسک با گیج ریسک',             duration: '۶:۲۰', tag: 'ریسک',        url: '#' },
];

const FAQS: FaqItem[] = [
  {
    q: 'چطور وارد داشبورد شوم؟',
    a: 'از صفحه ورود با ایمیل و رمز عبور وارد شوید. اگر حساب ندارید، روی «ثبت‌نام کنید» کلیک کنید. برای تست سریع می‌توانید از حساب‌های آزمایشی استفاده کنید.',
  },
  {
    q: 'داده‌های بازار هر چند وقت یک بار بروزرسانی می‌شوند؟',
    a: 'قیمت‌های بازار (دلار، طلا، سکه، کریپتو) هر ۶۰ ثانیه یک بار از tgju.org و Nobitex دریافت می‌شوند. تیکر بالای صفحه وضعیت «بازار زنده» را نشان می‌دهد.',
  },
  {
    q: 'آیا اطلاعاتم ذخیره می‌شود؟',
    a: 'دارایی‌های اضافه‌شده در مرورگر شما (localStorage) ذخیره می‌شوند. در نسخه‌های آینده امکان sync با سرور اضافه خواهد شد.',
  },
  {
    q: 'تفاوت OTM، ATM و ITM در آپشن چیست؟',
    a: 'OTM (Out-of-the-Money): قیمت اعمال بالاتر از قیمت جاری — پرمیوم کمتر، ریسک اعمال کمتر. ATM (At-the-Money): قیمت اعمال نزدیک به قیمت جاری. ITM (In-the-Money): قیمت اعمال پایین‌تر از قیمت جاری — پرمیوم بیشتر، احتمال اعمال زیاد.',
  },
  {
    q: 'چرا برخی قیمت‌ها «در دسترس نیست» نشان می‌دهد؟',
    a: 'شاخص کل بورس (TEDPIX) و برخی دارایی‌های خاص هنوز منبع داده زنده ندارند و به‌زودی اضافه می‌شوند.',
  },
  {
    q: 'آیا اپ موبایل دارد؟',
    a: 'فین‌دَش یک اپ PWA است — از مرورگر موبایل (Chrome یا Safari) وارد شوید و گزینه «افزودن به صفحه اصلی» را انتخاب کنید تا مانند یک اپ نصب شود.',
  },
];

const GETTING_STARTED_STEPS = [
  {
    icon: Rocket,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10 border-blue-500/20',
    title: 'ورود به حساب',
    desc: 'با ایمیل و رمز عبور وارد شوید. در صفحه ورود، دکمه‌های آزمایشی برای ورود سریع وجود دارند.',
  },
  {
    icon: BarChart3,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10 border-emerald-500/20',
    title: 'نمای کلی داشبورد',
    desc: 'در تب «نمای کلی» ارزش کل دارایی، سود روزانه، ریسک زنده و هشدارها را یک‌جا می‌بینید.',
  },
  {
    icon: Wallet,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10 border-amber-500/20',
    title: 'افزودن دارایی',
    desc: 'به تب «پرتفولیو» بروید و دکمه «افزودن دارایی» را بزنید. نوع دارایی، مقدار و قیمت خرید را وارد کنید.',
  },
  {
    icon: TrendingUp,
    color: 'text-purple-400',
    bg: 'bg-purple-500/10 border-purple-500/20',
    title: 'پایش بازار زنده',
    desc: 'تب «بازار» قیمت ارز، طلا، سکه و کریپتو را از منابع معتبر ایرانی نمایش می‌دهد.',
  },
  {
    icon: Shield,
    color: 'text-rose-400',
    bg: 'bg-rose-500/10 border-rose-500/20',
    title: 'مدیریت ریسک',
    desc: 'گیج ریسک در سایدبار میزان ریسک کل پرتفولیو را در بازه ۰ تا ۱۰۰ نمایش می‌دهد.',
  },
  {
    icon: Bell,
    color: 'text-orange-400',
    bg: 'bg-orange-500/10 border-orange-500/20',
    title: 'هشدارها',
    desc: 'تعداد هشدارهای باز در کارت بالا نمایش داده می‌شود. به‌زودی امکان تنظیم هشدار قیمتی اضافه می‌شود.',
  },
];

const TIPS = [
  {
    icon: RefreshCw,
    color: 'text-blue-400',
    title: 'بروزرسانی دستی بازار',
    body: 'در تب «بازار» دکمه «بروزرسانی» گوشه راست بالا وجود دارد. اگر داده‌ها قدیمی به نظر می‌رسند، روی آن کلیک کنید.',
  },
  {
    icon: TrendingUp,
    color: 'text-emerald-400',
    title: 'انتخاب قیمت اعمال در کاورد کال',
    body: 'برای بازار ایران، OTM با فاصله ۵-۱۰٪ از قیمت جاری تعادل خوبی بین پرمیوم و ریسک فروش سهام دارد.',
  },
  {
    icon: Shield,
    color: 'text-amber-400',
    title: 'ریسک نقدشوندگی آپشن ایران',
    body: 'قبل از ورود به هر موقعیت اختیار، حجم معاملات آن قرارداد را بررسی کنید. قراردادهای ATM با سررسید ماهانه معمولاً نقدشوندگی بهتری دارند.',
  },
  {
    icon: CheckCircle2,
    color: 'text-purple-400',
    title: 'رول کردن موقعیت',
    body: 'وقتی قیمت اختیار به ۲۰٪ ارزش اولیه رسید، آن را بازخرید کرده و برای ماه بعد Roll forward کنید — سود بیشتر با همان دارایی.',
  },
  {
    icon: AlertTriangle,
    color: 'text-rose-400',
    title: 'قبل از مجمع آپشن ببندید',
    body: 'تقسیم سود نقدی (DPS) قیمت سهام را کاهش می‌دهد و احتمال اعمال زودهنگام اختیار را بالا می‌برد. قبل از تاریخ مجمع، موقعیت را رول یا ببندید.',
  },
  {
    icon: Lightbulb,
    color: 'text-sky-400',
    title: 'نصب اپ روی موبایل',
    body: 'در Chrome موبایل از منو «Add to Home Screen» را انتخاب کنید. فین‌دَش مانند یک اپ نصب می‌شود — بدون App Store.',
  },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function VideoCardItem({ v }: { v: VideoCard }) {
  return (
    <a
      href={v.url ?? '#'}
      className="group flex flex-col gap-3 rounded-2xl border border-white/10 bg-white/[0.04] p-4 transition hover:border-blue-400/30 hover:bg-white/[0.07]"
    >
      <div className="relative flex h-32 items-center justify-center rounded-xl bg-gradient-to-br from-slate-800 to-slate-900 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.15),transparent_70%)]" />
        <PlayCircle className="h-12 w-12 text-white/40 transition group-hover:text-blue-400 group-hover:scale-110" />
        <span className="absolute bottom-2 right-2 rounded-lg bg-black/60 px-2 py-0.5 text-[10px] font-bold text-white" dir="ltr">
          {v.duration}
        </span>
        <span className="absolute top-2 left-2 rounded-full bg-blue-500/20 border border-blue-500/30 px-2 py-0.5 text-[10px] text-blue-300">
          {v.tag}
        </span>
      </div>
      <p className="text-sm font-bold text-white group-hover:text-blue-300 transition">{v.title}</p>
    </a>
  );
}

function FaqRow({ item }: { item: FaqItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-5 py-4 text-right hover:bg-white/[0.03] transition"
      >
        <span className="text-sm font-bold text-white">{item.q}</span>
        <ChevronLeft className={`h-4 w-4 shrink-0 text-slate-400 transition-transform duration-200 ${open ? '-rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-white/10 px-5 py-4 text-sm leading-7 text-slate-300">
          {item.a}
        </div>
      )}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export function HelpCenter() {
  const [active, setActive] = useState<Category>('getting-started');
  const [search, setSearch] = useState('');

  const filteredFaqs = FAQS.filter(
    (f) => !search || f.q.includes(search) || f.a.includes(search)
  );

  return (
    <div className="flex gap-5 min-h-[600px]">

      {/* Sidebar */}
      <aside className="hidden md:flex flex-col gap-1 w-52 shrink-0">
        <p className="mb-2 px-3 text-[10px] font-bold uppercase tracking-widest text-slate-500">راهنما</p>
        {CATEGORIES.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-bold transition ${
              active === id
                ? 'bg-blue-500/15 border border-blue-500/30 text-blue-300'
                : 'text-slate-400 hover:bg-white/[0.04] hover:text-white border border-transparent'
            }`}
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </button>
        ))}

        <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.03] p-4 space-y-2">
          <p className="text-[11px] font-bold text-slate-400">نیاز به کمک دارید؟</p>
          <p className="text-[11px] text-slate-500 leading-5">از طریق پنل پشتیبانی با ما در ارتباط باشید.</p>
          <a
            href="mailto:support@findash.ir"
            className="flex items-center gap-1 text-[11px] text-blue-400 hover:underline"
          >
            <ExternalLink className="h-3 w-3" />
            ارتباط با پشتیبانی
          </a>
        </div>
      </aside>

      {/* Mobile category bar */}
      <div className="flex md:hidden gap-2 overflow-x-auto scrollbar-none pb-1 absolute top-0 left-0 right-0">
        {CATEGORIES.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`shrink-0 flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-bold border transition ${
              active === id ? 'bg-blue-500/15 border-blue-500/30 text-blue-300' : 'border-white/10 text-slate-400'
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 space-y-5">

        {/* ── شروع به کار ── */}
        {active === 'getting-started' && (
          <div className="space-y-5">
            <div>
              <h2 className="text-xl font-black text-white">شروع به کار با فین‌دَش</h2>
              <p className="mt-1 text-sm text-slate-400">در چند دقیقه داشبورد معاملاتی خود را راه‌اندازی کنید.</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {GETTING_STARTED_STEPS.map((step, i) => {
                const Icon = step.icon;
                return (
                  <div key={i} className={`flex gap-4 rounded-2xl border p-4 ${step.bg}`}>
                    <div className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/[0.05]`}>
                      <Icon className={`h-5 w-5 ${step.color}`} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black text-slate-500">۰{i + 1}</span>
                        <p className="text-sm font-black text-white">{step.title}</p>
                      </div>
                      <p className="mt-1 text-xs leading-6 text-slate-400">{step.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Quick video */}
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
              <p className="mb-3 text-sm font-black text-white">ویدیوهای پیشنهادی برای شروع</p>
              <div className="grid gap-3 sm:grid-cols-2">
                {VIDEOS.filter((v) => v.tag === 'شروع به کار').map((v) => (
                  <VideoCardItem key={v.id} v={v} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── آموزش‌ها ── */}
        {active === 'tutorials' && (
          <div className="space-y-5">
            <div>
              <h2 className="text-xl font-black text-white">آموزش‌های تصویری</h2>
              <p className="mt-1 text-sm text-slate-400">از اصول تا استراتژی‌های پیشرفته بازار ایران.</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {VIDEOS.map((v) => <VideoCardItem key={v.id} v={v} />)}
            </div>

            {/* Strategy article cards */}
            <div className="space-y-3">
              <p className="text-sm font-black text-white">مقالات استراتژی آپشن</p>
              {[
                {
                  title: 'استراتژی کاورد کال — راهنمای کامل',
                  desc: 'چطور از سهام موجود در سبد، هر ماه درآمد پرمیوم کسب کنید. شامل مثال عددی با «فملی» و ریسک‌های خاص بازار ایران.',
                  tags: ['کاورد کال', 'آپشن', 'بورس ایران'],
                  color: 'border-blue-500/20 bg-blue-500/[0.06]',
                  tagColor: 'bg-blue-500/10 border-blue-500/20 text-blue-300',
                  btnColor: 'border-blue-500/30 bg-blue-500/10 text-blue-300 hover:bg-blue-500/20',
                  url: 'https://github.com/massoudsh/Findash/blob/main/docs/guides/option-strategies/covered-call.md',
                },
                {
                  title: 'کَش‌سکیورد پوت — خرید ارزان‌تر سهام',
                  desc: 'نقدینگی دارید و می‌خواهید سهامی بخرید؟ پرمیوم بگیرید تا سهام را ارزان‌تر وارد پرتفولیو کنید.',
                  tags: ['Cash-Secured Put', 'آپشن', 'بورس ایران'],
                  color: 'border-emerald-500/20 bg-emerald-500/[0.06]',
                  tagColor: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300',
                  btnColor: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20',
                  url: 'https://github.com/massoudsh/Findash/blob/main/docs/guides/option-strategies/cash-secured-put.md',
                },
                {
                  title: 'کالر — بیمه رایگان برای سهام',
                  desc: 'سهام دارید و نگران ریزش هستید؟ کالر هم کف زیان می‌گذارد هم سقف سود — و می‌تواند Zero-Cost باشد.',
                  tags: ['Collar', 'آپشن', 'مدیریت ریسک'],
                  color: 'border-amber-500/20 bg-amber-500/[0.06]',
                  tagColor: 'bg-amber-500/10 border-amber-500/20 text-amber-300',
                  btnColor: 'border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20',
                  url: 'https://github.com/massoudsh/Findash/blob/main/docs/guides/option-strategies/collar.md',
                },
                {
                  title: 'کانورژن — آربیتراژ بدون ریسک بازار',
                  desc: 'وقتی بازار قیمت‌گذاری اشتباه می‌کند. سه معامله همزمان — خرید سهام، فروش Call، خرید Put — و سود تضمینی از Put-Call Parity.',
                  tags: ['Conversion', 'آربیتراژ', 'آپشن پیشرفته'],
                  color: 'border-violet-500/20 bg-violet-500/[0.06]',
                  tagColor: 'bg-violet-500/10 border-violet-500/20 text-violet-300',
                  btnColor: 'border-violet-500/30 bg-violet-500/10 text-violet-300 hover:bg-violet-500/20',
                  url: 'https://github.com/massoudsh/Findash/blob/main/docs/guides/option-strategies/conversion.md',
                },
                {
                  title: 'لانگ استرادل — شرط روی نوسان، نه جهت',
                  desc: 'می‌دانید بازار تکان می‌خورد ولی نمی‌دانید کجا؟ Call و Put بخرید — از هر کدام برنده شود سود می‌برید.',
                  tags: ['Long Straddle', 'نوسان', 'رویداد خبری'],
                  color: 'border-rose-500/20 bg-rose-500/[0.06]',
                  tagColor: 'bg-rose-500/10 border-rose-500/20 text-rose-300',
                  btnColor: 'border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20',
                  url: 'https://github.com/massoudsh/Findash/blob/main/docs/guides/option-strategies/long-straddle.md',
                },
              ].map((article) => (
                <div key={article.title} className={`rounded-2xl border p-5 ${article.color}`}>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex gap-3">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/[0.05]">
                        <BookOpen className="h-5 w-5 text-white/60" />
                      </div>
                      <div>
                        <p className="font-black text-white">{article.title}</p>
                        <p className="mt-1 text-xs text-slate-400 leading-5">{article.desc}</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {article.tags.map((tag) => (
                            <span key={tag} className={`rounded-full border px-2.5 py-0.5 text-[10px] ${article.tagColor}`}>
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                    <a
                      href={article.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`flex shrink-0 items-center gap-1 rounded-xl border px-3 py-2 text-xs font-bold transition ${article.btnColor}`}
                    >
                      <Play className="h-3.5 w-3.5" />
                      مطالعه
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── سوالات متکرر ── */}
        {active === 'faq' && (
          <div className="space-y-5">
            <div>
              <h2 className="text-xl font-black text-white">سوالات متکرر</h2>
              <p className="mt-1 text-sm text-slate-400">پاسخ سریع به رایج‌ترین سوال‌ها.</p>
            </div>

            {/* Search */}
            <div className="relative">
              <Search className="absolute right-4 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="جستجو در سوالات..."
                className="w-full h-11 rounded-2xl border border-white/10 bg-white/[0.04] px-4 pr-11 text-sm text-white placeholder:text-slate-500 outline-none focus:border-blue-500/40 focus:ring-1 focus:ring-blue-500/20"
              />
            </div>

            <div className="space-y-2">
              {filteredFaqs.length > 0
                ? filteredFaqs.map((f, i) => <FaqRow key={i} item={f} />)
                : (
                  <div className="rounded-2xl border border-white/10 p-8 text-center text-slate-400 text-sm">
                    نتیجه‌ای برای «{search}» پیدا نشد
                  </div>
                )}
            </div>
          </div>
        )}

        {/* ── نکات مفید ── */}
        {active === 'tips' && (
          <div className="space-y-5">
            <div>
              <h2 className="text-xl font-black text-white">نکات مفید</h2>
              <p className="mt-1 text-sm text-slate-400">ترفندهایی که کار با داشبورد را آسان‌تر می‌کنند.</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {TIPS.map((tip, i) => {
                const Icon = tip.icon;
                return (
                  <div key={i} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 flex gap-4">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/[0.05]">
                      <Icon className={`h-5 w-5 ${tip.color}`} />
                    </div>
                    <div>
                      <p className="text-sm font-black text-white">{tip.title}</p>
                      <p className="mt-1 text-xs leading-6 text-slate-400">{tip.body}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
