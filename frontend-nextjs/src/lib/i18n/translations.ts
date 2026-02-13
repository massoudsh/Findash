export type Locale = 'en' | 'fa';

export const translations: Record<Locale, Record<string, string>> = {
  en: {
    // Nav group names
    'nav.group.Trading': 'Trading',
    'nav.group.Portfolio': 'Portfolio',
    'nav.group.Analysis': 'Analysis',
    'nav.group.Tools & Data': 'Tools & Data',
    'nav.group.System': 'System',
    // Nav items - Left
    'nav.item.Dashboard': 'Dashboard',
    'nav.item.Market': 'Market',
    'nav.item.Options': 'Options',
    'nav.item.Trading Center': 'Trading Center',
    'nav.item.Trading Bots': 'Trading Bots',
    'nav.item.Paper Trading & Backtesting': 'Paper Trading & Backtesting',
    'nav.item.Portfolio': 'Portfolio',
    'nav.item.Strategies': 'Strategies',
    'nav.item.Risk Assessment': 'Risk Assessment',
    // Nav items - Right
    'nav.item.Technical': 'Technical',
    'nav.item.Fundamental Research': 'Fundamental Research',
    'nav.item.Macro': 'Macro',
    'nav.item.On-chain': 'On-chain',
    'nav.item.Social Signals': 'Social Signals',
    'nav.item.AI Models': 'AI Models',
    'nav.item.Data Explorer': 'Data Explorer',
    'nav.item.Visualization': 'Visualization',
    'nav.item.Reports': 'Reports',
    'nav.item.API Playground': 'API Playground',
    'nav.item.Notifications': 'Notifications',
    'nav.item.Admin Panel': 'Admin Panel',
    'nav.item.Audit Log': 'Audit Log',
    'nav.item.Profile': 'Profile',
    'nav.item.Settings': 'Settings',
    'nav.item.Help': 'Help',
    // Header
    'nav.title.Analysis & Tools': 'Analysis & Tools',
    'common.search': 'Search commands...',
    'common.searchShort': 'Search...',
    'common.searchPlaceholder': 'Type a command or search...',
    'common.language': 'Language',
    'common.english': 'English',
    'common.persian': 'فارسی',
    'app.title': 'Octopus Trading Platform',
  },
  fa: {
    'nav.group.Trading': 'معاملات',
    'nav.group.Portfolio': 'پرتفوی',
    'nav.group.Analysis': 'تحلیل',
    'nav.group.Tools & Data': 'ابزار و داده',
    'nav.group.System': 'سیستم',
    'nav.item.Dashboard': 'داشبورد',
    'nav.item.Market': 'بازار',
    'nav.item.Options': 'اختیار معامله',
    'nav.item.Trading Center': 'مرکز معاملات',
    'nav.item.Trading Bots': 'ربات‌های معاملاتی',
    'nav.item.Paper Trading & Backtesting': 'معاملات کاغذی و بک‌تست',
    'nav.item.Portfolio': 'پرتفوی',
    'nav.item.Strategies': 'استراتژی‌ها',
    'nav.item.Risk Assessment': 'ارزیابی ریسک',
    'nav.item.Technical': 'تکنیکال',
    'nav.item.Fundamental Research': 'تحلیل بنیادی',
    'nav.item.Macro': 'کلان',
    'nav.item.On-chain': 'آن‌چین',
    'nav.item.Social Signals': 'سیگنال‌های اجتماعی',
    'nav.item.AI Models': 'مدل‌های هوش مصنوعی',
    'nav.item.Data Explorer': 'کاوشگر داده',
    'nav.item.Visualization': 'نمایش داده',
    'nav.item.Reports': 'گزارش‌ها',
    'nav.item.API Playground': 'محیط API',
    'nav.item.Notifications': 'اعلان‌ها',
    'nav.item.Admin Panel': 'پنل مدیریت',
    'nav.item.Audit Log': 'لاگ ممیزی',
    'nav.item.Profile': 'پروفایل',
    'nav.item.Settings': 'تنظیمات',
    'nav.item.Help': 'راهنما',
    'nav.title.Analysis & Tools': 'تحلیل و ابزار',
    'common.search': 'جستجوی دستورات...',
    'common.searchShort': 'جستجو...',
    'common.searchPlaceholder': 'دستور یا جستجو را بنویسید...',
    'common.language': 'زبان',
    'common.english': 'English',
    'common.persian': 'فارسی',
    'app.title': 'پلتفرم معاملاتی اکتپوس',
  },
};

const LOCALE_STORAGE_KEY = 'findash-locale';

export function getStoredLocale(): Locale {
  if (typeof window === 'undefined') return 'en';
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
  return stored === 'fa' ? 'fa' : 'en';
}

export function setStoredLocale(locale: Locale): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(LOCALE_STORAGE_KEY, locale);
}

export function t(locale: Locale, key: string): string {
  const dict = translations[locale];
  return dict[key] ?? translations.en[key] ?? key;
}

export function getDir(locale: Locale): 'ltr' | 'rtl' {
  return locale === 'fa' ? 'rtl' : 'ltr';
}
