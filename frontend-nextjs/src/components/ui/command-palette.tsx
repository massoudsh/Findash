'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useTranslations } from '@/lib/i18n/locale-context';
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Search,
  BarChart3, 
  Briefcase, 
  Target, 
  TrendingUp,
  Activity,
  Brain,
  Shield,
  MessageSquare,
  PieChart,
  FileText,
  User,
  Settings as SettingsIcon,
  FlaskConical,
  Bell,
  ServerCog,
  BookOpen,
  ListChecks,
  Database,
  DollarSign,
  Moon,
  Sun,
  Zap,
  Download,
  Upload,
  Copy,
  RefreshCw,
  LogOut,
  Calculator,
  Calendar,
  Clock,
  Filter,
  SortAsc,
  ArrowRight,
  Command as CommandIcon,
  Keyboard,
  Home,
  Plus,
  Edit,
  Trash2,
  Eye,
  Share,
  Star,
  Bookmark,
  Archive,
  Mail,
  Phone
} from 'lucide-react';

interface CommandAction {
  id: string;
  title: string;
  description?: string;
  action: () => void;
  icon: any;
  category: string;
  keywords: string[];
  shortcut?: string;
  badge?: string;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface PlatformSearchResult {
  id: string;
  title: string;
  path: string;
  type: string;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [platformResults, setPlatformResults] = useState<PlatformSearchResult[]>([]);
  const t = useTranslations();

  // Platform search (Elasticsearch-backed when available)
  useEffect(() => {
    if (!search || search.length < 2) {
      setPlatformResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(search)}`);
        if (res.ok) {
          const data = await res.json();
          setPlatformResults(data.results || []);
        } else {
          setPlatformResults([]);
        }
      } catch {
        setPlatformResults([]);
      }
    }, 150);
    return () => clearTimeout(timer);
  }, [search]);

  // Navigation actions
  const navigationActions: CommandAction[] = [
    {
      id: 'nav-dashboard',
      title: 'داشبورد',
      description: 'رفتن به داشبورد اصلی',
      action: () => router.push('/'),
      icon: BarChart3,
      category: 'ناوبری',
      keywords: ['dashboard', 'home', 'overview', 'داشبورد'],
      shortcut: 'Ctrl+D'
    },
    {
      id: 'nav-portfolio',
      title: 'پرتفوی',
      description: 'مشاهده و مدیریت پرتفوی سرمایه‌گذاری',
      action: () => router.push('/dashboard?tab=portfolio'),
      icon: Briefcase,
      category: 'ناوبری',
      keywords: ['portfolio', 'investments', 'assets', 'پرتفوی']
    },
    {
      id: 'nav-options',
      title: 'معاملات اختیار',
      description: 'دسترسی به پلتفرم معاملات اختیار',
      action: () => router.push('/options'),
      icon: DollarSign,
      category: 'ناوبری',
      keywords: ['options', 'trading', 'derivatives', 'اختیار']
    },
    {
      id: 'nav-strategies',
      title: 'استراتژی‌های معاملاتی',
      description: 'مدیریت استراتژی‌های معاملاتی شما',
      action: () => router.push('/strategies'),
      icon: Target,
      category: 'ناوبری',
      keywords: ['strategies', 'algorithms', 'trading', 'استراتژی']
    },
    {
      id: 'nav-trades',
      title: 'تاریخچه معاملات',
      description: 'مشاهده معاملات گذشته و جاری',
      action: () => router.push('/trades'),
      icon: TrendingUp,
      category: 'ناوبری',
      keywords: ['trades', 'history', 'orders', 'معاملات']
    },
    {
      id: 'nav-realtime',
      title: 'داده بلادرنگ',
      description: 'داده و تحلیل زنده بازار',
      action: () => router.push('/realtime'),
      icon: Activity,
      category: 'ناوبری',
      keywords: ['realtime', 'live', 'market', 'data', 'بلادرنگ']
    },
    {
      id: 'nav-ai-models',
      title: 'مدل‌های هوش مصنوعی',
      description: 'مدل‌های یادگیری ماشین و آموزش آن‌ها',
      action: () => router.push('/ai-models'),
      icon: Brain,
      category: 'ناوبری',
      keywords: ['ai', 'ml', 'models', 'training', 'هوش مصنوعی']
    },
    {
      id: 'nav-risk',
      title: 'مدیریت ریسک',
      description: 'تحلیل ریسک و استرس‌تست',
      action: () => router.push('/trading?tab=risk'),
      icon: Shield,
      category: 'ناوبری',
      keywords: ['risk', 'analysis', 'var', 'stress', 'ریسک']
    },
    {
      id: 'nav-visualization',
      title: 'نمودار و نمایش داده',
      description: 'نمودارسازی پیشرفته و مصورسازی داده',
      action: () => router.push('/visualization'),
      icon: PieChart,
      category: 'ناوبری',
      keywords: ['charts', 'visualization', 'graphs', 'نمودار']
    },
    {
      id: 'nav-reports',
      title: 'گزارش‌ها',
      description: 'تولید و مشاهده گزارش‌ها',
      action: () => router.push('/reports'),
      icon: FileText,
      category: 'ناوبری',
      keywords: ['reports', 'analytics', 'pdf', 'گزارش']
    }
  ];

  // Trading actions
  const tradingActions: CommandAction[] = [
    {
      id: 'trade-buy',
      title: 'خرید سهم',
      description: 'ثبت سفارش خرید',
      action: () => {
        onOpenChange(false);
        // Mock action - would open buy dialog
        console.log('Opening buy order dialog');
      },
      icon: TrendingUp,
      category: 'معاملات',
      keywords: ['buy', 'purchase', 'long', 'order', 'خرید'],
      shortcut: 'Ctrl+B',
      badge: 'سریع'
    },
    {
      id: 'trade-sell',
      title: 'فروش سهم',
      description: 'ثبت سفارش فروش',
      action: () => {
        onOpenChange(false);
        console.log('Opening sell order dialog');
      },
      icon: TrendingUp,
      category: 'معاملات',
      keywords: ['sell', 'short', 'order', 'فروش'],
      shortcut: 'Ctrl+S'
    },
    {
      id: 'trade-options',
      title: 'معامله اختیار',
      description: 'باز کردن رابط معاملات اختیار',
      action: () => router.push('/options'),
      icon: Target,
      category: 'معاملات',
      keywords: ['options', 'calls', 'puts', 'derivatives', 'اختیار']
    },
    {
      id: 'trade-portfolio-rebalance',
      title: 'بازتوازن پرتفوی',
      description: 'بازتوازن خودکار پرتفوی شما',
      action: () => {
        onOpenChange(false);
        console.log('Starting portfolio rebalancing');
      },
      icon: Briefcase,
      category: 'معاملات',
      keywords: ['rebalance', 'portfolio', 'allocation', 'بازتوازن']
    }
  ];

  // Analysis actions
  const analysisActions: CommandAction[] = [
    {
      id: 'analysis-stock',
      title: 'تحلیل سهم',
      description: 'اجرای تحلیل بنیادی و تکنیکال',
      action: () => {
        onOpenChange(false);
        console.log('Opening stock analysis');
      },
      icon: BarChart3,
      category: 'تحلیل',
      keywords: ['analyze', 'stock', 'fundamental', 'technical', 'تحلیل']
    },
    {
      id: 'analysis-backtest',
      title: 'بک‌تست استراتژی',
      description: 'آزمایش استراتژی روی داده‌های تاریخی',
      action: () => router.push('/strategies?tab=backtesting'),
      icon: FlaskConical,
      category: 'تحلیل',
      keywords: ['backtest', 'strategy', 'historical', 'simulation', 'بک‌تست']
    },
    {
      id: 'analysis-risk',
      title: 'ارزیابی ریسک',
      description: 'تحلیل معیارهای ریسک پرتفوی',
      action: () => router.push('/trading?tab=risk'),
      icon: Shield,
      category: 'تحلیل',
      keywords: ['risk', 'var', 'stress', 'assessment', 'ریسک']
    },
    {
      id: 'analysis-generate-report',
      title: 'تولید گزارش',
      description: 'ساخت گزارش کامل معاملاتی',
      action: () => {
        onOpenChange(false);
        console.log('Generating report');
      },
      icon: FileText,
      category: 'تحلیل',
      keywords: ['report', 'generate', 'pdf', 'analytics', 'گزارش']
    }
  ];

  // System actions
  const systemActions: CommandAction[] = [
    {
      id: 'system-settings',
      title: 'تنظیمات',
      description: 'باز کردن تنظیمات برنامه',
      action: () => router.push('/settings'),
      icon: SettingsIcon,
      category: 'سیستم',
      keywords: ['settings', 'preferences', 'config', 'تنظیمات']
    },
    {
      id: 'system-profile',
      title: 'پروفایل',
      description: 'مشاهده و ویرایش پروفایل شما',
      action: () => router.push('/profile'),
      icon: User,
      category: 'سیستم',
      keywords: ['profile', 'account', 'user', 'پروفایل']
    },
    {
      id: 'system-notifications',
      title: 'اعلان‌ها',
      description: 'مشاهده اعلان‌های سیستم',
      action: () => router.push('/notifications'),
      icon: Bell,
      category: 'سیستم',
      keywords: ['notifications', 'alerts', 'messages', 'اعلان']
    },
    {
      id: 'system-help',
      title: 'راهنما و مستندات',
      description: 'دسترسی به راهنما و مستندات',
      action: () => router.push('/help'),
      icon: BookOpen,
      category: 'سیستم',
      keywords: ['help', 'docs', 'documentation', 'support', 'راهنما']
    },
    {
      id: 'system-logout',
      title: 'خروج',
      description: 'خروج از حساب کاربری',
      action: () => {
        onOpenChange(false);
        console.log('Logging out');
      },
      icon: LogOut,
      category: 'سیستم',
      keywords: ['logout', 'signout', 'exit', 'خروج']
    }
  ];

  // Quick actions
  const quickActions: CommandAction[] = [
    {
      id: 'quick-refresh',
      title: 'به‌روزرسانی داده',
      description: 'به‌روزرسانی تمام داده‌های بازار',
      action: () => {
        onOpenChange(false);
        console.log('Refreshing data');
      },
      icon: RefreshCw,
      category: 'اقدامات سریع',
      keywords: ['refresh', 'reload', 'update', 'به‌روزرسانی'],
      shortcut: 'Ctrl+R'
    },
    {
      id: 'quick-export',
      title: 'خروجی‌گیری داده',
      description: 'خروجی‌گیری از داده‌های پرتفوی',
      action: () => {
        onOpenChange(false);
        console.log('Exporting data');
      },
      icon: Download,
      category: 'اقدامات سریع',
      keywords: ['export', 'download', 'csv', 'pdf', 'خروجی']
    },
    {
      id: 'quick-calculator',
      title: 'ماشین‌حساب اختیار معامله',
      description: 'باز کردن ماشین‌حساب قیمت‌گذاری اختیار',
      action: () => {
        onOpenChange(false);
        console.log('Opening calculator');
      },
      icon: Calculator,
      category: 'اقدامات سریع',
      keywords: ['calculator', 'options', 'pricing', 'greeks', 'ماشین‌حساب']
    },
    {
      id: 'quick-watchlist',
      title: 'افزودن به دیده‌بان',
      description: 'افزودن نماد جاری به لیست دیده‌بان',
      action: () => {
        onOpenChange(false);
        console.log('Adding to watchlist');
      },
      icon: Star,
      category: 'اقدامات سریع',
      keywords: ['watchlist', 'favorite', 'star', 'add', 'دیده‌بان']
    }
  ];

  const allActions = [
    ...navigationActions,
    ...tradingActions,
    ...analysisActions,
    ...systemActions,
    ...quickActions
  ];

  // Filter actions based on search
  const filteredActions = allActions.filter(action => {
    if (!search) return true;
    const searchLower = search.toLowerCase();
    return (
      action.title.toLowerCase().includes(searchLower) ||
      action.description?.toLowerCase().includes(searchLower) ||
      action.keywords.some(keyword => keyword.toLowerCase().includes(searchLower))
    );
  });

  // Group actions by category
  const groupedActions = filteredActions.reduce((groups, action) => {
    const category = action.category;
    if (!groups[category]) {
      groups[category] = [];
    }
    groups[category].push(action);
    return groups;
  }, {} as Record<string, CommandAction[]>);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K to open command palette
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        onOpenChange(!open);
      }
      
      // Escape to close
      if (e.key === 'Escape' && open) {
        onOpenChange(false);
      }

      // Handle other shortcuts when palette is closed
      if (!open) {
        const action = allActions.find(a => {
          if (!a.shortcut) return false;
          const keys = a.shortcut.toLowerCase().split('+');
          const ctrlOrCmd = keys.includes('ctrl') || keys.includes('cmd');
          const key = keys[keys.length - 1];
          
          return (e.ctrlKey || e.metaKey) === ctrlOrCmd && e.key.toLowerCase() === key;
        });
        
        if (action) {
          e.preventDefault();
          action.action();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, onOpenChange, allActions]);

  const runCommand = useCallback((command: CommandAction) => {
    onOpenChange(false);
    command.action();
  }, [onOpenChange]);

  const runPlatformResult = useCallback(
    (result: PlatformSearchResult) => {
      onOpenChange(false);
      router.push(result.path);
    },
    [onOpenChange, router]
  );

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput 
        placeholder={t('common.searchPlaceholder')}
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>نتیجه‌ای یافت نشد.</CommandEmpty>

        {platformResults.length > 0 && (
          <>
            <CommandGroup heading="پلتفرم">
              {platformResults.map((result) => (
                <CommandItem
                  key={result.id}
                  value={`${result.title} ${result.path}`}
                  onSelect={() => runPlatformResult(result)}
                  className="flex items-center gap-3 py-3"
                >
                  <Search className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <div className="font-medium">{result.title}</div>
                    <div className="text-xs text-muted-foreground">{result.path}</div>
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
            <CommandSeparator />
          </>
        )}
        
        {Object.entries(groupedActions).map(([category, actions]) => (
          <CommandGroup key={category} heading={category}>
            {actions.map((action) => {
              const Icon = action.icon;
              return (
                <CommandItem
                  key={action.id}
                  value={`${action.title} ${action.description} ${action.keywords.join(' ')}`}
                  onSelect={() => runCommand(action)}
                  className="flex items-center justify-between py-3"
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="h-4 w-4" />
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{action.title}</span>
                        {action.badge && (
                          <Badge variant="secondary" className="text-xs">
                            {action.badge}
                          </Badge>
                        )}
                      </div>
                      {action.description && (
                        <div className="text-sm text-muted-foreground">
                          {action.description}
                        </div>
                      )}
                    </div>
                  </div>
                  {action.shortcut && (
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      {action.shortcut.split('+').map((key, index) => (
                        <span key={index}>
                          {index > 0 && <span className="mx-1">+</span>}
                          <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs">
                            {key}
                          </kbd>
                        </span>
                      ))}
                    </div>
                  )}
                </CommandItem>
              );
            })}
          </CommandGroup>
        ))}
        
        {search && filteredActions.length === 0 && (
          <CommandGroup>
            <CommandItem disabled>
              <Search className="h-4 w-4 mr-3" />
              <span>هیچ دستوری برای «{search}» یافت نشد</span>
            </CommandItem>
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}

// Helper component for showing command palette trigger
export function CommandPaletteTrigger({
  onOpen,
  iconOnly = false,
}: {
  onOpen: () => void;
  iconOnly?: boolean;
}) {
  const t = useTranslations();
  if (iconOnly) {
    return (
      <Button
        variant="outline"
        size="icon"
        className="h-9 w-9 shrink-0 bg-background shadow-none"
        onClick={onOpen}
        aria-label={t('common.commandPalette')}
      >
        <CommandIcon className="h-4 w-4" />
      </Button>
    );
  }
  return (
    <Button
      variant="outline"
      className="relative h-8 w-full justify-start bg-background text-sm font-normal text-muted-foreground shadow-none sm:pr-12 md:w-40 lg:w-64"
      onClick={onOpen}
      aria-label={t('common.commandPalette')}
    >
      <span className="hidden lg:inline-flex">{t('common.commandPalette')}</span>
      <span className="inline-flex lg:hidden">{t('common.commandPaletteShort')}</span>
      <div className="pointer-events-none absolute right-1.5 top-1.5 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
        <span className="text-xs">⌘</span>K
      </div>
    </Button>
  );
} 