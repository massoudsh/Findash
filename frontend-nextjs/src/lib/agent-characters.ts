/**
 * Agent characters for the platform.
 * Each agent has a persona used across مرکز فرماندهی, Risk, Portfolio, Backtesting, etc.
 * Aligned with backend: intelligence_orchestrator (M1–M11).
 */

export const AGENT_IDS = [
  'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11',
] as const;

export type AgentId = typeof AGENT_IDS[number];

export interface AgentCharacter {
  id: AgentId;
  name: string;
  shortName: string;
  tagline: string;
  description: string;
  /** Lucide icon name for UI */
  icon: string;
  /** Tailwind classes for avatar ring / accent (e.g. ring-emerald-500/40) */
  colorClass: string;
  /** Emoji used as avatar fallback or flair */
  emoji: string;
  /** Pages/sections this agent is responsible for */
  responsibleFor: string[];
}

export const AGENT_CHARACTERS: Record<AgentId, AgentCharacter> = {
  M1: {
    id: 'M1',
    name: 'عامل گردآوری داده',
    shortName: 'پیوند',
    tagline: 'هر فید را جاری نگه می‌دارم',
    description: 'داده‌های بازار، اخبار و داده‌های جایگزین را از تمام منابع گردآوری و نرمال‌سازی می‌کند.',
    icon: 'Database',
    colorClass: 'ring-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
    emoji: '📡',
    responsibleFor: ['Market Data', 'Data Explorer', 'مرکز فرماندهی (data pipeline)'],
  },
  M2: {
    id: 'M2',
    name: 'عامل انبار داده',
    shortName: 'خزانه',
    tagline: 'داده‌هایت، سازمان‌یافته',
    description: 'داده‌های تاریخی و بلادرنگ را ذخیره، اعتبارسنجی و سرویس‌دهی می‌کند.',
    icon: 'Archive',
    colorClass: 'ring-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300',
    emoji: '🗄️',
    responsibleFor: ['Data Explorer', 'Exports', 'Historical data'],
  },
  M3: {
    id: 'M3',
    name: 'عامل پردازش بلادرنگ',
    shortName: 'نبض',
    tagline: 'داده زنده، بدون تأخیر',
    description: 'داده‌های جاری بازار را پردازش کرده و تحلیل‌های لحظه‌ای و هشدارها را پشتیبانی می‌کند.',
    icon: 'Activity',
    colorClass: 'ring-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
    emoji: '⚡',
    responsibleFor: ['Real-time', 'Live Trading', 'Alerts'],
  },
  M4: {
    id: 'M4',
    name: 'عامل استراتژی',
    shortName: 'اطلس',
    tagline: 'سیگنال‌هایی که بازار را می‌جنبانند',
    description: 'سیگنال‌های معاملاتی تولید کرده و اجرای استراتژی را با بک‌تست ترکیب می‌کند.',
    icon: 'Target',
    colorClass: 'ring-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    emoji: '🎯',
    responsibleFor: ['مرکز فرماندهی', 'Strategies', 'Trading Bots', 'Signals'],
  },
  M5: {
    id: 'M5',
    name: 'عامل مدل‌های هوش مصنوعی',
    shortName: 'نورون',
    tagline: 'یادگیری عمیق، مزیت عمیق‌تر',
    description: 'مدل‌های پیش‌بینی، دسته‌بندی و یادگیری عمیق را اجرا می‌کند.',
    icon: 'Brain',
    colorClass: 'ring-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300',
    emoji: '🧠',
    responsibleFor: ['AI Models', 'Training', 'Predictions'],
  },
  M6: {
    id: 'M6',
    name: 'عامل مدیریت ریسک',
    shortName: 'نگهبان',
    tagline: 'ریسک زیر کنترل',
    description: 'ریسک، حجم موقعیت، VaR و انطباق پرتفولیو را ارزیابی می‌کند.',
    icon: 'Shield',
    colorClass: 'ring-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300',
    emoji: '🛡️',
    responsibleFor: ['Risk Dashboard', 'Portfolio risk', 'Compliance'],
  },
  M7: {
    id: 'M7',
    name: 'عامل پیش‌بینی قیمت',
    shortName: 'پیشگو',
    tagline: 'قیمت بعدی کجا می‌رود',
    description: 'مدل‌های پیش‌بینی سری زمانی و قیمت را اجرا می‌کند.',
    icon: 'TrendingUp',
    colorClass: 'ring-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300',
    emoji: '🔮',
    responsibleFor: ['Predictions', 'Technical analysis', 'Forecasts'],
  },
  M8: {
    id: 'M8',
    name: 'عامل معامله کاغذی',
    shortName: 'سایه',
    tagline: 'تمرین بدون فشار',
    description: 'اجرا را شبیه‌سازی کرده و عملکرد پرتفولیوی کاغذی را ردیابی می‌کند.',
    icon: 'Copy',
    colorClass: 'ring-slate-500/40 bg-slate-500/10 text-slate-700 dark:text-slate-300',
    emoji: '📋',
    responsibleFor: ['Paper Trading', 'Portfolio (sim)', 'Execution sim'],
  },
  M9: {
    id: 'M9',
    name: 'عامل سنتیمنت بازار',
    shortName: 'پژواک',
    tagline: 'جمع چه احساسی دارد',
    description: 'اخبار و سنتیمنت اجتماعی را برای دارایی‌ها و تم‌ها تحلیل می‌کند.',
    icon: 'MessageSquare',
    colorClass: 'ring-pink-500/40 bg-pink-500/10 text-pink-700 dark:text-pink-300',
    emoji: '💬',
    responsibleFor: ['Social', 'Sentiment', 'مرکز فرماندهی (sentiment panel)'],
  },
  M10: {
    id: 'M10',
    name: 'عامل بک‌تست',
    shortName: 'تاریخ‌نگار',
    tagline: 'تاریخ تکرار می‌شود، ما اندازه‌اش می‌گیریم',
    description: 'بک‌تست‌های تاریخی و اعتبارسنجی استراتژی را اجرا می‌کند.',
    icon: 'History',
    colorClass: 'ring-orange-500/40 bg-orange-500/10 text-orange-700 dark:text-orange-300',
    emoji: '📜',
    responsibleFor: ['Backtesting', 'Strategy validation'],
  },
  M11: {
    id: 'M11',
    name: 'عامل نمایش داده',
    shortName: 'لنز',
    tagline: 'تصویر کامل را ببین',
    description: 'نمودارها، داشبوردها و بینش‌های گزارش مبتنی بر هوش مصنوعی را پشتیبانی می‌کند.',
    icon: 'BarChart3',
    colorClass: 'ring-indigo-500/40 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300',
    emoji: '📊',
    responsibleFor: ['Reports', 'Visualization', 'مرکز فرماندهی (insights)', 'Dashboards'],
  },
};

export function getAgentCharacter(id: AgentId): AgentCharacter {
  return AGENT_CHARACTERS[id];
}

export function getAgentsForPage(page: string): AgentId[] {
  const pageLower = page.toLowerCase();
  const out: AgentId[] = [];
  (AGENT_IDS as unknown as AgentId[]).forEach((id) => {
    const a = AGENT_CHARACTERS[id];
    if (a.responsibleFor.some((r) => r.toLowerCase().includes(pageLower) || pageLower.includes(r.toLowerCase().split(' ')[0])))
      out.push(id);
  });
  return out.length ? out : [];
}
