'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BookOpen,
  BarChart3,
  TrendingUp,
  Target,
  Brain,
  Shield,
  Database,
  MessageSquare,
  Activity,
  History,
  PieChart,
  Newspaper,
  BellRing,
  GitBranch,
  Cpu,
  ChevronDown,
  ChevronRight,
  LineChart,
  Bot,
  Layers,
  Zap,
  Settings,
  User,
  FileText,
  Code,
  Info,
  Keyboard,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';

interface Section {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
  color: string;
  description: string;
  steps: string[];
  agentId?: string;
  agentName?: string;
}

interface FAQItem {
  q: string;
  a: string;
}

const APP_SECTIONS: Section[] = [
  {
    id: 'dashboard',
    title: 'داشبورد',
    icon: BarChart3,
    href: '/dashboard',
    color: 'text-blue-500',
    description: 'نمای کلی از وضعیت پرتفولیو، بازار، و عملکرد روزانه.',
    steps: [
      'از سایدبار چپ روی «داشبورد» کلیک کن.',
      'کارت‌های KPI (ارزش پرتفولیو، سود/زیان، تعداد موقعیت‌های باز) در بالا نمایش داده می‌شوند.',
      'نمودار عملکرد تاریخی را در بخش مرکزی ببین.',
      'تب‌های پایین‌تر: «پرتفولیو»، «بازار»، «تحلیل» را بررسی کن.',
      'برای تعامل با عاملان هوش مصنوعی، پنل سمت راست را باز کن.',
    ],
    agentId: 'M11',
    agentName: 'لنز',
  },
  {
    id: 'trading',
    title: 'مرکز فرماندهی',
    icon: TrendingUp,
    href: '/trading',
    color: 'text-emerald-500',
    description: 'مرکز اصلی معامله: اختیار معامله، استراتژی‌ها، ریسک، و ربات‌ها.',
    steps: [
      'از سایدبار چپ روی «مرکز فرماندهی» کلیک کن.',
      'تب «اختیار معامله»: ابزارهای تصمیم‌گیری برای معاملات آپشن.',
      'تب «استراتژی‌ها»: لیست استراتژی‌های فعال و امکان بک‌تست.',
      'تب «ریسک»: داشبورد مدیریت ریسک با VaR و سقف موقعیت.',
      'تب «ربات‌های معاملاتی»: تنظیم، راه‌اندازی، و پایش ربات‌ها.',
      'پنل عاملان سمت راست هر تب: بینش‌های بلادرنگ هوش مصنوعی.',
    ],
    agentId: 'M4',
    agentName: 'اطلس',
  },
  {
    id: 'news',
    title: 'اخبار بازار ایران',
    icon: Newspaper,
    href: '/dashboard',
    color: 'text-orange-500',
    description: 'اخبار بلادرنگ بازار ایران به‌صورت نوار متحرک در بالای صفحه نمایش داده می‌شود.',
    steps: [
      'نوار خبری متحرک همیشه در بالای صفحه (کنار لوگو) در حال اسکرول است.',
      'اخبار به‌صورت بلادرنگ از منابع ایرانی دریافت می‌شود.',
      'روی هر خبر کلیک کن تا منبع اصلی در تب جدید باز شود.',
    ],
    agentId: 'M9',
    agentName: 'پژواک',
  },
  {
    id: 'alerts',
    title: 'هشدار قیمت',
    icon: BellRing,
    href: '/alerts',
    color: 'text-yellow-500',
    description: 'تعیین آستانه قیمتی و دریافت اعلان هنگام رسیدن به هدف.',
    steps: [
      'از سایدبار چپ روی «هشدار قیمت» کلیک کن.',
      'روی «افزودن هشدار جدید» کلیک کن.',
      'نماد مورد نظر را جستجو و انتخاب کن.',
      'قیمت هدف، جهت (بالاتر/پایین‌تر)، و روش اعلان را تنظیم کن.',
      'هشدار ذخیره می‌شود و هنگام فعال‌شدن اعلان دریافت می‌کنی.',
    ],
  },
  {
    id: 'technical',
    title: 'تکنیکال',
    icon: Target,
    href: '/technical',
    color: 'text-blue-600',
    description: 'تحلیل تکنیکال با اندیکاتورها، الگوها، و سیگنال‌های خودکار.',
    steps: [
      'از سایدبار چپ روی «تکنیکال» کلیک کن.',
      'نماد مورد نظر را در نوار جستجو وارد کن.',
      'تایم‌فریم دلخواه را از منوی بالای نمودار انتخاب کن.',
      'اندیکاتورها را از پنل راست اضافه کن (RSI، MACD، BB و ...).',
      'سیگنال‌های اتوماتیک را در پنل «سیگنال‌ها» زیر نمودار ببین.',
    ],
    agentId: 'M7',
    agentName: 'پیشگو',
  },
  {
    id: 'fundamental-data',
    title: 'تحلیل بنیادی',
    icon: Brain,
    href: '/fundamental-data',
    color: 'text-violet-500',
    description: 'صورت‌های مالی، نسبت‌های بنیادی، و تحلیل کیفی شرکت‌ها.',
    steps: [
      'از سایدبار چپ روی «تحلیل بنیادی» کلیک کن.',
      'نماد شرکت را جستجو کن.',
      'تب‌ها: «خلاصه»، «صورت مالی»، «نسبت‌ها»، «تاریخچه سود» را بررسی کن.',
      'مقایسه با صنعت در پنل «مقایسه بخشی» انجام می‌شود.',
    ],
    agentId: 'M5',
    agentName: 'نورون',
  },
  {
    id: 'macro',
    title: 'کلان',
    icon: LineChart,
    href: '/macro',
    color: 'text-amber-500',
    description: 'شاخص‌های اقتصاد کلان، تورم، نرخ بهره، و داده‌های ارزی.',
    steps: [
      'از سایدبار چپ روی «کلان» کلیک کن.',
      'شاخص‌های کلیدی (تورم، رشد GDP، نرخ دلار) در کارت‌های بالا.',
      'نمودارهای سری زمانی را برای هر شاخص باز کن.',
      'ارتباط بین شاخص‌ها و بازار سهام در تب «همبستگی» قابل بررسی است.',
    ],
  },
  {
    id: 'on-chain',
    title: 'آن‌چین',
    icon: Database,
    href: '/on-chain',
    color: 'text-violet-600',
    description: 'داده‌های زنجیره‌ای رمزارزها: تراکنش‌ها، آدرس‌های فعال، جریان صرافی.',
    steps: [
      'از سایدبار چپ روی «آن‌چین» کلیک کن.',
      'شبکه مورد نظر (BTC، ETH، ...) را انتخاب کن.',
      'متریک‌های کلیدی: تراکنش‌های روزانه، هزینه گس، آدرس‌های فعال.',
      'جریان ورود/خروج از صرافی‌ها در تب «جریان صرافی» قابل مشاهده است.',
    ],
    agentId: 'M1',
    agentName: 'پیوند',
  },
  {
    id: 'social',
    title: 'سیگنال‌های اجتماعی',
    icon: MessageSquare,
    href: '/social',
    color: 'text-pink-500',
    description: 'سنتیمنت اجتماعی از توییتر، ردیت، شاخص ترس و طمع.',
    steps: [
      'از سایدبار چپ روی «سیگنال‌های اجتماعی» کلیک کن.',
      'شاخص «ترس و طمع» را در بالای صفحه ببین.',
      'نمودار سنتیمنت نمادهای مختلف در بخش مرکزی.',
      'تب «ردیت/توییتر» آخرین پست‌های مرتبط با هر نماد را نشان می‌دهد.',
    ],
    agentId: 'M9',
    agentName: 'پژواک',
  },
  {
    id: 'ai-models',
    title: 'مدل‌های هوش مصنوعی',
    icon: Cpu,
    href: '/ai-models',
    color: 'text-cyan-500',
    description: 'مدیریت، آموزش، و پیش‌بینی مدل‌های یادگیری ماشین.',
    steps: [
      'از سایدبار چپ روی «مدل‌های هوش مصنوعی» کلیک کن.',
      'لیست مدل‌های فعال با وضعیت (آماده/در حال آموزش) نمایش داده می‌شود.',
      'روی هر مدل کلیک کن تا جزئیات، متریک‌های ارزیابی، و تاریخچه آموزش ببینی.',
      'برای اجرای پیش‌بینی: نماد انتخاب کن، افق زمانی تعیین کن، «پیش‌بینی» را بزن.',
    ],
    agentId: 'M5',
    agentName: 'نورون',
  },
];

const AGENTS = [
  { id: 'M1', name: 'پیوند', fullName: 'عامل گردآوری داده', emoji: '📡', color: 'text-sky-500', tagline: 'هر فید را جاری نگه می‌دارم', desc: 'داده‌های بازار، اخبار، و داده‌های آن‌چین را از تمام منابع گردآوری می‌کند.' },
  { id: 'M2', name: 'خزانه', fullName: 'عامل انبار داده', emoji: '🗄️', color: 'text-violet-500', tagline: 'داده‌هایت، سازمان‌یافته', desc: 'داده‌های تاریخی را ذخیره و سرویس‌دهی می‌کند.' },
  { id: 'M3', name: 'نبض', fullName: 'عامل پردازش بلادرنگ', emoji: '⚡', color: 'text-amber-500', tagline: 'داده زنده، بدون تأخیر', desc: 'قیمت‌های لحظه‌ای و اطلاعات بازار را در زمان واقعی پردازش می‌کند.' },
  { id: 'M4', name: 'اطلس', fullName: 'عامل استراتژی', emoji: '🎯', color: 'text-emerald-500', tagline: 'سیگنال‌هایی که بازار را می‌جنبانند', desc: 'سیگنال معاملاتی تولید می‌کند و اجرای استراتژی را مدیریت می‌کند.' },
  { id: 'M5', name: 'نورون', fullName: 'عامل مدل‌های هوش مصنوعی', emoji: '🧠', color: 'text-fuchsia-500', tagline: 'یادگیری عمیق، مزیت عمیق‌تر', desc: 'مدل‌های پیش‌بینی و یادگیری ماشین را آموزش داده و اجرا می‌کند.' },
  { id: 'M6', name: 'نگهبان', fullName: 'عامل مدیریت ریسک', emoji: '🛡️', color: 'text-rose-500', tagline: 'ریسک زیر کنترل', desc: 'VaR، سقف موقعیت، و انطباق پرتفولیو را پایش می‌کند.' },
  { id: 'M7', name: 'پیشگو', fullName: 'عامل پیش‌بینی قیمت', emoji: '🔮', color: 'text-cyan-500', tagline: 'قیمت بعدی کجا می‌رود', desc: 'پیش‌بینی قیمت مبتنی بر سری زمانی و مدل‌های ترکیبی.' },
  { id: 'M8', name: 'سایه', fullName: 'عامل معامله کاغذی', emoji: '📋', color: 'text-slate-500', tagline: 'تمرین بدون فشار', desc: 'اجرای شبیه‌سازی‌شده معاملات بدون ریسک سرمایه واقعی.' },
  { id: 'M9', name: 'پژواک', fullName: 'عامل سنتیمنت بازار', emoji: '💬', color: 'text-pink-500', tagline: 'جمع چه احساسی دارد', desc: 'سنتیمنت اخبار و شبکه‌های اجتماعی را تحلیل می‌کند.' },
  { id: 'M10', name: 'تاریخ‌نگار', fullName: 'عامل بک‌تست', emoji: '📜', color: 'text-orange-500', tagline: 'تاریخ تکرار می‌شود، ما اندازه‌اش می‌گیریم', desc: 'استراتژی‌ها را روی داده‌های تاریخی آزمایش می‌کند.' },
  { id: 'M11', name: 'لنز', fullName: 'عامل نمایش داده', emoji: '📊', color: 'text-indigo-500', tagline: 'تصویر کامل را ببین', desc: 'نمودارها، داشبوردها، و بینش‌های گزارش را پشتیبانی می‌کند.' },
];

const FAQS: FAQItem[] = [
  { q: 'از کجا شروع کنم؟', a: 'از داشبورد شروع کن. کارت‌های KPI، نمودار عملکرد، و موقعیت‌های باز را ببین. سپس به «مرکز فرماندهی» برو تا با ابزارهای معاملاتی آشنا بشی.' },
  { q: 'عاملان هوش مصنوعی چی هستند؟', a: 'پلتفرم ۱۱ عامل هوش مصنوعی دارد که هر کدام مسئولیت بخشی از پردازش داده و تحلیل را دارند. هر صفحه پنل عامل مربوطه را در سمت راست نمایش می‌دهد.' },
  { q: 'معامله کاغذی چیست؟', a: 'معامله کاغذی (Paper Trading) شبیه‌سازی معاملات بدون پول واقعی است. عامل «سایه» (M8) این محیط را مدیریت می‌کند. از «مرکز فرماندهی» به سراغش برو.' },
  { q: 'چطور بک‌تست بگیرم؟', a: 'در «مرکز فرماندهی» تب «استراتژی‌ها» را باز کن، سپس تب فرعی «بک‌تست» را انتخاب کن. استراتژی، بازه زمانی، و سرمایه اولیه را تعیین کن و «اجرا» را بزن.' },
  { q: 'مرکز فرماندهی چه تب‌هایی دارد؟', a: 'چهار تب: «اختیار معامله» (ابزارهای آپشن)، «استراتژی‌ها» (سیگنال و بک‌تست)، «ریسک» (مدیریت ریسک)، «ربات‌های معاملاتی» (تنظیم بات‌ها).' },
  { q: 'چطور هشدار قیمت بسازم؟', a: 'به بخش «هشدار قیمت» در سایدبار چپ برو. روی «افزودن هشدار» کلیک کن، نماد، قیمت هدف، و جهت را تعیین کن. هنگام فعال‌شدن اعلان می‌گیری.' },
  { q: 'سایدبار راست چیست؟', a: 'سایدبار راست شامل «داده و نمودارها»، «گزارش‌ها»، «API Playground»، «اعلان‌ها»، «مدیریت»، «حساب»، «گردش کار»، و «راهنما» می‌شود.' },
  { q: 'چطور زبان پلتفرم را تغییر دهم؟', a: 'در پایین سایدبار چپ، کنار تم‌سوئیچر، گزینه زبان قرار دارد. می‌توانی بین فارسی، انگلیسی، و اسپانیایی جابجا شوی.' },
  { q: 'داده‌های بازار از کجا می‌آیند؟', a: 'عامل «پیوند» (M1) داده را از منابع متعدد گردآوری می‌کند و عامل «نبض» (M3) پردازش بلادرنگ را انجام می‌دهد. در بخش «داده و نمودارها» جزئیات منابع قابل مشاهده است.' },
  { q: 'API Playground چیست؟', a: 'یک محیط تعاملی برای تست endpoint‌های API پلتفرم. می‌توانی درخواست‌های REST بفرستی و پاسخ‌ها را مستقیم ببینی.' },
];

const SHORTCUTS = [
  { key: '⌘K', desc: 'باز کردن Command Palette — جستجوی سریع در کل پلتفرم' },
  { key: '⌘/', desc: 'نمایش راهنمای کلیدهای میانبر' },
  { key: 'G → D', desc: 'رفتن به داشبورد' },
  { key: 'G → T', desc: 'رفتن به مرکز فرماندهی' },
  { key: 'G → N', desc: 'رفتن به اخبار' },
  { key: 'G → H', desc: 'رفتن به راهنما' },
  { key: 'Esc', desc: 'بستن دیالوگ یا Command Palette' },
];

export default function HelpPage() {
  const [activeTab, setActiveTab] = useState('manual');
  const [expandedSection, setExpandedSection] = useState<string | null>('dashboard');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
          <BookOpen className="w-8 h-8 text-emerald-500" />
          راهنمای پلتفرم
        </h1>
        <p className="text-muted-foreground mt-1">
          مرجع کامل استفاده از بخش‌ها، عاملان هوش مصنوعی، و ابزارهای پلتفرم
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="manual" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            راهنمای بخش‌ها
          </TabsTrigger>
          <TabsTrigger value="agents" className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            عاملان هوش مصنوعی
          </TabsTrigger>
          <TabsTrigger value="faq" className="flex items-center gap-2">
            <Info className="h-4 w-4" />
            سوالات متداول
          </TabsTrigger>
          <TabsTrigger value="shortcuts" className="flex items-center gap-2">
            <Keyboard className="h-4 w-4" />
            کلیدهای میانبر
          </TabsTrigger>
        </TabsList>

        {/* ── Tab 1: Manual ── */}
        <TabsContent value="manual" className="space-y-4 mt-4">
          {/* Quick Nav */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {APP_SECTIONS.map((s) => {
              const Icon = s.icon;
              return (
                <button
                  key={s.id}
                  onClick={() => setExpandedSection(expandedSection === s.id ? null : s.id)}
                  className={cn(
                    'flex flex-col items-center gap-1.5 p-3 rounded-lg border text-xs font-medium transition-colors hover:bg-muted/60',
                    expandedSection === s.id ? 'bg-muted border-primary/40' : 'bg-card'
                  )}
                >
                  <Icon className={cn('h-5 w-5', s.color)} />
                  <span className="text-center leading-tight">{s.title}</span>
                </button>
              );
            })}
          </div>

          {/* Section Detail */}
          {APP_SECTIONS.map((section) => {
            const Icon = section.icon;
            const isOpen = expandedSection === section.id;
            return (
              <Card key={section.id} className={cn('transition-all', isOpen && 'ring-1 ring-primary/20')}>
                <button
                  className="w-full text-right"
                  onClick={() => setExpandedSection(isOpen ? null : section.id)}
                >
                  <CardHeader className="py-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Icon className={cn('h-5 w-5', section.color)} />
                        <div className="text-right">
                          <CardTitle className="text-sm">{section.title}</CardTitle>
                          <p className="text-xs text-muted-foreground mt-0.5">{section.description}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {section.agentId && (
                          <Badge variant="outline" className="text-[10px]">
                            {section.agentId} · {section.agentName}
                          </Badge>
                        )}
                        <Link
                          href={section.href}
                          onClick={(e) => e.stopPropagation()}
                          className="text-xs text-primary hover:underline px-2"
                        >
                          رفتن به بخش
                        </Link>
                        {isOpen ? (
                          <ChevronDown className="h-4 w-4 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>
                    </div>
                  </CardHeader>
                </button>
                {isOpen && (
                  <CardContent className="pt-0 pb-4">
                    <ol className="space-y-2 pr-4">
                      {section.steps.map((step, i) => (
                        <li key={i} className="flex gap-3 text-sm text-muted-foreground">
                          <span className={cn('shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white', section.color.replace('text-', 'bg-'))}>
                            {i + 1}
                          </span>
                          {step}
                        </li>
                      ))}
                    </ol>
                  </CardContent>
                )}
              </Card>
            );
          })}

          {/* Right sidebar note */}
          <Card className="bg-muted/30">
            <CardHeader className="py-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="h-4 w-4 text-muted-foreground" />
                سایدبار ابزار و سیستم (راست)
              </CardTitle>
            </CardHeader>
            <CardContent className="pb-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                {[
                  { icon: PieChart, label: 'داده و نمودارها', href: '/data' },
                  { icon: FileText, label: 'گزارش‌ها', href: '/reports' },
                  { icon: Code, label: 'API Playground', href: '/api-playground' },
                  { icon: BellRing, label: 'اعلان‌ها', href: '/notifications' },
                  { icon: Settings, label: 'مدیریت', href: '/admin' },
                  { icon: User, label: 'حساب', href: '/account' },
                  { icon: GitBranch, label: 'گردش کار', href: '/workflow' },
                  { icon: BookOpen, label: 'راهنما', href: '/help' },
                ].map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="flex items-center gap-1.5 p-2 rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <Icon className="h-3.5 w-3.5 shrink-0" />
                      {item.label}
                    </Link>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Tab 2: Agents ── */}
        <TabsContent value="agents" className="space-y-4 mt-4">
          <Card className="bg-muted/30">
            <CardContent className="py-4 text-sm text-muted-foreground">
              پلتفرم دارای <strong className="text-foreground">۱۱ عامل هوش مصنوعی</strong> است که هر کدام یک بخش از پردازش داده و تحلیل را پوشش می‌دهند.
              هر صفحه پنل عامل مربوطه را در سمت راست نشان می‌دهد. عاملان با داده‌های بلادرنگ بینش تولید می‌کنند.
            </CardContent>
          </Card>
          <div className="grid gap-3 sm:grid-cols-2">
            {AGENTS.map((agent) => (
              <Card key={agent.id} className="hover:shadow-md transition-shadow">
                <CardContent className="py-4">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">{agent.emoji}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={cn('font-semibold text-sm', agent.color)}>{agent.name}</span>
                        <Badge variant="outline" className="text-[10px]">{agent.id}</Badge>
                        <span className="text-xs text-muted-foreground">{agent.fullName}</span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 italic">«{agent.tagline}»</p>
                      <p className="text-xs text-muted-foreground mt-1.5">{agent.desc}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* ── Tab 3: FAQ ── */}
        <TabsContent value="faq" className="space-y-3 mt-4">
          {FAQS.map((faq, i) => (
            <Card key={i}>
              <button
                className="w-full text-right px-4 py-3 flex items-center justify-between hover:bg-muted/40 rounded-lg transition-colors"
                onClick={() => setExpandedFaq(expandedFaq === i ? null : i)}
              >
                <span className="font-medium text-sm">{faq.q}</span>
                {expandedFaq === i ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0 ml-2" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0 ml-2" />
                )}
              </button>
              {expandedFaq === i && (
                <CardContent className="pt-0 pb-4 text-sm text-muted-foreground border-t mx-4">
                  <p className="pt-3">{faq.a}</p>
                </CardContent>
              )}
            </Card>
          ))}
        </TabsContent>

        {/* ── Tab 4: Shortcuts ── */}
        <TabsContent value="shortcuts" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Zap className="h-4 w-4 text-yellow-500" />
                کلیدهای میانبر
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {SHORTCUTS.map((s, i) => (
                  <div key={i} className="flex items-center gap-4 py-2 border-b last:border-0">
                    <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono shrink-0 min-w-16 text-center">
                      {s.key}
                    </kbd>
                    <span className="text-sm text-muted-foreground">{s.desc}</span>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
                <strong className="text-foreground">نکته:</strong> Command Palette (⌘K) سریع‌ترین راه برای جابجایی بین بخش‌ها است. کافی است نام بخش مورد نظر را تایپ کنی.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
