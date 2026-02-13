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

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const t = useTranslations();

  // Navigation actions
  const navigationActions: CommandAction[] = [
    {
      id: 'nav-dashboard',
      title: 'Dashboard',
      description: 'Go to main dashboard',
      action: () => router.push('/'),
      icon: BarChart3,
      category: 'Navigation',
      keywords: ['dashboard', 'home', 'overview'],
      shortcut: 'Ctrl+D'
    },
    {
      id: 'nav-portfolio',
      title: 'Portfolio',
      description: 'View your investment portfolio',
      action: () => router.push('/portfolio'),
      icon: Briefcase,
      category: 'Navigation',
      keywords: ['portfolio', 'investments', 'assets']
    },
    {
      id: 'nav-options',
      title: 'Options Trading',
      description: 'Access options trading platform',
      action: () => router.push('/options'),
      icon: DollarSign,
      category: 'Navigation',
      keywords: ['options', 'trading', 'derivatives']
    },
    {
      id: 'nav-strategies',
      title: 'Trading Strategies',
      description: 'Manage your trading strategies',
      action: () => router.push('/strategies'),
      icon: Target,
      category: 'Navigation',
      keywords: ['strategies', 'algorithms', 'trading']
    },
    {
      id: 'nav-trades',
      title: 'Trade History',
      description: 'View past and current trades',
      action: () => router.push('/trades'),
      icon: TrendingUp,
      category: 'Navigation',
      keywords: ['trades', 'history', 'orders']
    },
    {
      id: 'nav-realtime',
      title: 'Real-time Data',
      description: 'Live market data and analytics',
      action: () => router.push('/realtime'),
      icon: Activity,
      category: 'Navigation',
      keywords: ['realtime', 'live', 'market', 'data']
    },
    {
      id: 'nav-ai-models',
      title: 'AI Models',
      description: 'Machine learning models and training',
      action: () => router.push('/ai-models'),
      icon: Brain,
      category: 'Navigation',
      keywords: ['ai', 'ml', 'models', 'training']
    },
    {
      id: 'nav-risk',
      title: 'Risk Management',
      description: 'Risk analysis and stress testing',
      action: () => router.push('/risk'),
      icon: Shield,
      category: 'Navigation',
      keywords: ['risk', 'analysis', 'var', 'stress']
    },
    {
      id: 'nav-visualization',
      title: 'Charts & Visualization',
      description: 'Advanced charting and data visualization',
      action: () => router.push('/visualization'),
      icon: PieChart,
      category: 'Navigation',
      keywords: ['charts', 'visualization', 'graphs']
    },
    {
      id: 'nav-reports',
      title: 'Reports',
      description: 'Generate and view reports',
      action: () => router.push('/reports'),
      icon: FileText,
      category: 'Navigation',
      keywords: ['reports', 'analytics', 'pdf']
    }
  ];

  // Trading actions
  const tradingActions: CommandAction[] = [
    {
      id: 'trade-buy',
      title: 'Buy Stock',
      description: 'Place a buy order',
      action: () => {
        onOpenChange(false);
        // Mock action - would open buy dialog
        console.log('Opening buy order dialog');
      },
      icon: TrendingUp,
      category: 'Trading',
      keywords: ['buy', 'purchase', 'long', 'order'],
      shortcut: 'Ctrl+B',
      badge: 'Quick'
    },
    {
      id: 'trade-sell',
      title: 'Sell Stock',
      description: 'Place a sell order',
      action: () => {
        onOpenChange(false);
        console.log('Opening sell order dialog');
      },
      icon: TrendingUp,
      category: 'Trading',
      keywords: ['sell', 'short', 'order'],
      shortcut: 'Ctrl+S'
    },
    {
      id: 'trade-options',
      title: 'Trade Options',
      description: 'Open options trading interface',
      action: () => router.push('/options'),
      icon: Target,
      category: 'Trading',
      keywords: ['options', 'calls', 'puts', 'derivatives']
    },
    {
      id: 'trade-portfolio-rebalance',
      title: 'Rebalance Portfolio',
      description: 'Automatically rebalance your portfolio',
      action: () => {
        onOpenChange(false);
        console.log('Starting portfolio rebalancing');
      },
      icon: Briefcase,
      category: 'Trading',
      keywords: ['rebalance', 'portfolio', 'allocation']
    }
  ];

  // Analysis actions
  const analysisActions: CommandAction[] = [
    {
      id: 'analysis-stock',
      title: 'Analyze Stock',
      description: 'Run fundamental and technical analysis',
      action: () => {
        onOpenChange(false);
        console.log('Opening stock analysis');
      },
      icon: BarChart3,
      category: 'Analysis',
      keywords: ['analyze', 'stock', 'fundamental', 'technical']
    },
    {
      id: 'analysis-backtest',
      title: 'Backtest Strategy',
      description: 'Test strategy against historical data',
      action: () => router.push('/backtesting'),
      icon: FlaskConical,
      category: 'Analysis',
      keywords: ['backtest', 'strategy', 'historical', 'simulation']
    },
    {
      id: 'analysis-risk',
      title: 'Risk Assessment',
      description: 'Analyze portfolio risk metrics',
      action: () => router.push('/risk'),
      icon: Shield,
      category: 'Analysis',
      keywords: ['risk', 'var', 'stress', 'assessment']
    },
    {
      id: 'analysis-generate-report',
      title: 'Generate Report',
      description: 'Create comprehensive trading report',
      action: () => {
        onOpenChange(false);
        console.log('Generating report');
      },
      icon: FileText,
      category: 'Analysis',
      keywords: ['report', 'generate', 'pdf', 'analytics']
    }
  ];

  // System actions
  const systemActions: CommandAction[] = [
    {
      id: 'system-settings',
      title: 'Settings',
      description: 'Open application settings',
      action: () => router.push('/settings'),
      icon: SettingsIcon,
      category: 'System',
      keywords: ['settings', 'preferences', 'config']
    },
    {
      id: 'system-profile',
      title: 'Profile',
      description: 'View and edit your profile',
      action: () => router.push('/profile'),
      icon: User,
      category: 'System',
      keywords: ['profile', 'account', 'user']
    },
    {
      id: 'system-notifications',
      title: 'Notifications',
      description: 'View system notifications',
      action: () => router.push('/notifications'),
      icon: Bell,
      category: 'System',
      keywords: ['notifications', 'alerts', 'messages']
    },
    {
      id: 'system-help',
      title: 'Help & Documentation',
      description: 'Access help and documentation',
      action: () => router.push('/help'),
      icon: BookOpen,
      category: 'System',
      keywords: ['help', 'docs', 'documentation', 'support']
    },
    {
      id: 'system-logout',
      title: 'Logout',
      description: 'Sign out of your account',
      action: () => {
        onOpenChange(false);
        console.log('Logging out');
      },
      icon: LogOut,
      category: 'System',
      keywords: ['logout', 'signout', 'exit']
    }
  ];

  // Quick actions
  const quickActions: CommandAction[] = [
    {
      id: 'quick-refresh',
      title: 'Refresh Data',
      description: 'Refresh all market data',
      action: () => {
        onOpenChange(false);
        console.log('Refreshing data');
      },
      icon: RefreshCw,
      category: 'Quick Actions',
      keywords: ['refresh', 'reload', 'update'],
      shortcut: 'Ctrl+R'
    },
    {
      id: 'quick-export',
      title: 'Export Data',
      description: 'Export portfolio data',
      action: () => {
        onOpenChange(false);
        console.log('Exporting data');
      },
      icon: Download,
      category: 'Quick Actions',
      keywords: ['export', 'download', 'csv', 'pdf']
    },
    {
      id: 'quick-calculator',
      title: 'Options Calculator',
      description: 'Open options pricing calculator',
      action: () => {
        onOpenChange(false);
        console.log('Opening calculator');
      },
      icon: Calculator,
      category: 'Quick Actions',
      keywords: ['calculator', 'options', 'pricing', 'greeks']
    },
    {
      id: 'quick-watchlist',
      title: 'Add to Watchlist',
      description: 'Add current symbol to watchlist',
      action: () => {
        onOpenChange(false);
        console.log('Adding to watchlist');
      },
      icon: Star,
      category: 'Quick Actions',
      keywords: ['watchlist', 'favorite', 'star', 'add']
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

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput 
        placeholder={t('common.searchPlaceholder')}
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        
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
              <span>No commands found for "{search}"</span>
            </CommandItem>
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}

// Helper component for showing command palette trigger
export function CommandPaletteTrigger({ onOpen }: { onOpen: () => void }) {
  const t = useTranslations();
  return (
    <Button
      variant="outline"
      className="relative h-8 w-full justify-start bg-background text-sm font-normal text-muted-foreground shadow-none sm:pr-12 md:w-40 lg:w-64"
      onClick={onOpen}
    >
      <span className="hidden lg:inline-flex">{t('common.search')}</span>
      <span className="inline-flex lg:hidden">{t('common.searchShort')}</span>
      <div className="pointer-events-none absolute right-1.5 top-1.5 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
        <span className="text-xs">⌘</span>K
      </div>
    </Button>
  );
} 