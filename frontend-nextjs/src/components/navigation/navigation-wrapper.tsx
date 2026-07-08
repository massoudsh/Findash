'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { OctopusLogo } from '@/components/ui/octopus-logo';
import { 
  BarChart3,
  Menu,
  Target,
  TrendingUp,
  Activity,
  Brain,
  MessageSquare,
  PieChart,
  FileText,
  User,
  Settings as SettingsIcon,
  Bell,
  ServerCog,
  BookOpen,
  ListChecks,
  Database,
  Percent,
  ChevronLeft,
  ChevronRight,
  LineChart,
  Cpu,
  GitBranch,
  Newspaper,
  BellRing,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { UserMenu } from "@/components/navigation/user-menu";
import { ThemeSwitcher } from "@/components/ui/theme-switcher";
import { LanguageSwitcher } from "@/components/ui/language-switcher";
import { NotificationCenter } from "@/components/ui/notification-center";
import { CommandPalette, CommandPaletteTrigger } from "@/components/ui/command-palette";
import { useTranslations } from '@/lib/i18n/locale-context';

interface NavigationWrapperProps {
  children: React.ReactNode;
}

// Left Sidebar - Trading + Analysis & Research (unique icons when wrapped)
const leftSidebarItems = {
  'Trading': [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'Command Center', href: '/trading', icon: TrendingUp },
    { name: 'Iran Market News', href: '/news', icon: Newspaper },
    { name: 'Price Alerts', href: '/alerts', icon: BellRing },
  ],
  'Analysis & Research': [
    { name: 'Technical', href: '/technical', icon: Target },
    { name: 'Fundamental Research', href: '/fundamental-data', icon: Brain },
    { name: 'Macro', href: '/macro', icon: LineChart },
    { name: 'On-chain', href: '/on-chain', icon: Database },
    { name: 'Social Signals', href: '/social', icon: MessageSquare },
    { name: 'AI Models', href: '/ai-models', icon: Cpu },
  ],
};

// Right Sidebar - Tools & System (unique icons when wrapped)
const rightSidebarItems = {
  'Tools & System': [
    { name: 'Data & Charts', href: '/data', icon: PieChart },
    { name: 'Reports', href: '/reports', icon: FileText },
    { name: 'API Playground', href: '/api-playground', icon: Activity },
    { name: 'Notifications', href: '/notifications', icon: Bell },
    { name: 'Admin', href: '/admin', icon: ServerCog },
    { name: 'Account', href: '/account', icon: User },
    { name: 'Workflow', href: '/workflow', icon: GitBranch },
    { name: 'Help', href: '/help', icon: BookOpen },
  ],
};

const SIDEBAR_EXPANDED = 256; // w-64 = 16rem = 256px
const SIDEBAR_COLLAPSED = 64;  // w-16 = 4rem = 64px

const SIDEBAR_HOVER_LEAVE_DELAY_MS = 200;

export function NavigationWrapper({ children }: NavigationWrapperProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(true);
  const [rightCollapsed, setRightCollapsed] = useState(true);
  const leftLeaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rightLeaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const t = useTranslations();

  type NavItem = { name: string; href: string; icon: React.ComponentType<{ className?: string }> };

  const isActive = (item: NavItem) => {
    if (pathname === item.href) return true;
    if (item.href === '/trading') return pathname === '/trading';
    if (item.href === '/dashboard') return pathname === '/dashboard';
    return false;
  };

  const NavigationGroup = ({
    title,
    items,
    collapsed,
  }: {
    title: string;
    items: NavItem[];
    collapsed: boolean;
  }) => (
    <div className={cn('mb-6', collapsed && 'mb-4')}>
      {!collapsed && (
        <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {t(`nav.group.${title}`)}
        </h3>
      )}
      <nav className={cn('space-y-1', collapsed && 'space-y-0.5')}>
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={item.href}
              title={collapsed ? t(`nav.item.${item.name}`) : undefined}
              className={cn(
                'flex items-center rounded-lg text-sm font-medium transition-colors',
                collapsed
                  ? 'justify-center px-2 py-2.5'
                  : 'gap-3 px-3 py-2',
                isActive(item)
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
              onClick={() => setIsOpen(false)}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{t(`nav.item.${item.name}`)}</span>}
            </Link>
          );
        })}
      </nav>
    </div>
  );

  const LeftNavigationContent = ({ collapsed }: { collapsed: boolean }) => (
    <div>
      {Object.entries(leftSidebarItems).map(([title, items]) => (
        <NavigationGroup key={title} title={title} items={items} collapsed={collapsed} />
      ))}
    </div>
  );

  const RightNavigationContent = ({ collapsed }: { collapsed: boolean }) => (
    <div>
      {Object.entries(rightSidebarItems).map(([title, items]) => (
        <NavigationGroup key={title} title={title} items={items} collapsed={collapsed} />
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Navigation */}
      <div className="lg:hidden">
        <div className="flex items-center justify-between border-b px-4 py-4">
          <Link href="/" aria-label="Octopus home" className="flex items-center">
            <OctopusLogo size={48} showText={false} />
          </Link>
          <div className="flex items-center gap-4">
            <CommandPaletteTrigger onOpen={() => setCommandOpen(true)} />
            <NotificationCenter />
            <LanguageSwitcher />
            <ThemeSwitcher />
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Menu className="h-6 w-6" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64">
                <div className="mt-8">
                  <LeftNavigationContent collapsed={false} />
                </div>
              </SheetContent>
            </Sheet>
            <UserMenu />
          </div>
        </div>
      </div>

      <div className="lg:flex">
        {/* Left Desktop Sidebar - hover to expand, leave to collapse */}
        <div
          className={cn(
            'hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-40 lg:flex lg:flex-col transition-[width] duration-200 ease-out',
            leftCollapsed ? 'lg:w-16' : 'lg:w-64'
          )}
          onMouseEnter={() => {
            if (leftLeaveTimeoutRef.current) {
              clearTimeout(leftLeaveTimeoutRef.current);
              leftLeaveTimeoutRef.current = null;
            }
            setLeftCollapsed(false);
          }}
          onMouseLeave={() => {
            leftLeaveTimeoutRef.current = setTimeout(() => {
              setLeftCollapsed(true);
              leftLeaveTimeoutRef.current = null;
            }, SIDEBAR_HOVER_LEAVE_DELAY_MS);
          }}
        >
          <div className="flex grow flex-col gap-y-5 overflow-y-auto overflow-x-hidden border-r border-white/30 dark:border-white/20 bg-card pb-4 transition-[padding] duration-200 ease-out border border-green-500/20 shadow-[0_0_14px_rgba(34,197,94,0.12)] rounded-r-lg">
            <div className={cn('px-2 pt-4', leftCollapsed && 'flex justify-center px-0')}>
              <CommandPaletteTrigger
                onOpen={() => setCommandOpen(true)}
                iconOnly={leftCollapsed}
              />
            </div>
            <div className={leftCollapsed ? 'px-0' : 'px-2'}>
              <LeftNavigationContent collapsed={leftCollapsed} />
            </div>
            <div className="mt-auto">
              <div className={cn('flex mb-4 gap-2', leftCollapsed ? 'flex-col items-center px-0' : 'flex-row justify-between px-2')}>
                <ThemeSwitcher showLabel={!leftCollapsed} />
                <LanguageSwitcher />
              </div>
              <div className={leftCollapsed ? 'flex justify-center' : 'px-2'}>
                <UserMenu />
              </div>
              <Button
                variant="ghost"
                size="icon"
                className={cn('w-full rounded-t-lg mt-2', leftCollapsed && 'w-10 mx-auto')}
                onClick={() => setLeftCollapsed((c) => !c)}
                aria-label={leftCollapsed ? 'Expand left sidebar' : 'Collapse left sidebar'}
              >
                {leftCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Main Content - padding responds to sidebar width */}
        <div
          className={cn(
            'min-w-0 transition-[padding] duration-200 ease-out',
            leftCollapsed ? 'lg:pl-16' : 'lg:pl-64',
            rightCollapsed ? 'lg:pr-16' : 'lg:pr-64'
          )}
        >
          <main className="py-6 pb-24 lg:pb-6">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              {children}
            </div>
          </main>
        </div>

        {/* Right Desktop Sidebar - hover to expand, leave to collapse */}
        <div
          className={cn(
            'hidden lg:fixed lg:inset-y-0 lg:right-0 lg:z-40 lg:flex lg:flex-col transition-[width] duration-200 ease-out',
            rightCollapsed ? 'lg:w-16' : 'lg:w-64'
          )}
          onMouseEnter={() => {
            if (rightLeaveTimeoutRef.current) {
              clearTimeout(rightLeaveTimeoutRef.current);
              rightLeaveTimeoutRef.current = null;
            }
            setRightCollapsed(false);
          }}
          onMouseLeave={() => {
            rightLeaveTimeoutRef.current = setTimeout(() => {
              setRightCollapsed(true);
              rightLeaveTimeoutRef.current = null;
            }, SIDEBAR_HOVER_LEAVE_DELAY_MS);
          }}
        >
          <div className="flex grow flex-col gap-y-5 overflow-y-auto overflow-x-hidden border-l border-white/30 dark:border-white/20 bg-card pb-4 transition-[padding] duration-200 ease-out border border-green-500/20 shadow-[0_0_14px_rgba(34,197,94,0.12)] rounded-l-lg">
            <div className={cn('flex h-16 shrink-0 items-center justify-center', !rightCollapsed && 'px-2')}>
              {!rightCollapsed ? (
                <h2 className="text-lg font-semibold text-muted-foreground">{t('nav.title.Tools & System')}</h2>
              ) : null}
            </div>
            <div className={cn('grow overflow-y-auto', rightCollapsed ? 'px-0' : 'px-2')}>
              <RightNavigationContent collapsed={rightCollapsed} />
            </div>
            <Button
              variant="ghost"
              size="icon"
              className={cn('shrink-0 rounded-t-lg mt-auto', rightCollapsed ? 'w-10 mx-auto' : 'w-full')}
              onClick={() => setRightCollapsed((c) => !c)}
              aria-label={rightCollapsed ? 'Expand right sidebar' : 'Collapse right sidebar'}
            >
              {rightCollapsed ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
      
      {/* Command Palette */}
      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} />

      {/* Mobile Bottom Navigation — ناوبری پایین موبایل */}
      <nav className="lg:hidden fixed bottom-0 inset-x-0 z-50 bg-card/95 backdrop-blur-xl border-t border-white/10 safe-area-inset-bottom"
           style={{ boxShadow: '0 -4px 20px rgba(0,0,0,0.3), 0 -1px 0 rgba(34,197,94,0.1)' }}>
        {/* نوار تزئینی سبز */}
        <div className="h-px w-full bg-gradient-to-r from-transparent via-green-500/40 to-transparent" />
        <div className="flex items-stretch h-16">
          {[
            { href: '/dashboard', icon: BarChart3, label: 'داشبورد' },
            { href: '/trading', icon: TrendingUp, label: 'معاملات' },
            { href: '/news', icon: Newspaper, label: 'اخبار' },
            { href: '/portfolio', icon: PieChart, label: 'پرتفولیو' },
            { href: '/technical', icon: Target, label: 'تحلیل' },
          ].map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'flex flex-1 flex-col items-center justify-center gap-1 text-[10px] font-medium transition-all duration-200 active:scale-95',
                  active
                    ? 'text-green-400'
                    : 'text-muted-foreground'
                )}
              >
                <div className={cn(
                  'rounded-xl p-1.5 transition-all duration-200',
                  active && 'bg-green-500/15 shadow-[0_0_10px_rgba(34,197,94,0.2)]'
                )}>
                  <Icon className={cn('h-5 w-5', active && 'drop-shadow-[0_0_4px_rgba(34,197,94,0.8)]')} />
                </div>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
        <div className="h-[env(safe-area-inset-bottom)]" />
      </nav>
    </div>
  );
} 