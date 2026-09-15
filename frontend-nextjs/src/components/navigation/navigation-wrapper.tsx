'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger, SheetTitle, SheetDescription } from '@/components/ui/sheet';
import { OctopusLogo } from '@/components/ui/octopus-logo';
import {
  BarChart3, Menu, Target, TrendingUp, Activity, Brain, MessageSquare,
  PieChart, FileText, User, Bell, ServerCog, BookOpen, Database,
  ChevronLeft, ChevronRight, LineChart, Cpu, GitBranch, BellRing,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { UserMenu } from '@/components/navigation/user-menu';
import { ThemeSwitcher } from '@/components/ui/theme-switcher';
import { LanguageSwitcher } from '@/components/ui/language-switcher';
import { NotificationCenter } from '@/components/ui/notification-center';
import { CommandPalette, CommandPaletteTrigger } from '@/components/ui/command-palette';
import { NewsTicker } from '@/components/navigation/news-ticker';
import { useLocale } from '@/lib/i18n/locale-context';

const leftSidebarItems = {
  'Trading': [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'Command Center', href: '/trading', icon: TrendingUp },
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

type NavItem = { name: string; href: string; icon: React.ComponentType<{ className?: string }> };

export function NavigationWrapper({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [leftCollapsed, setLeftCollapsed] = useState(true);
  const [rightCollapsed, setRightCollapsed] = useState(true);
  const pathname = usePathname();
  const { t, isRtl } = useLocale();

  const renderGroups = (groups: Record<string, NavItem[]>, collapsed: boolean) =>
    Object.entries(groups).map(([title, items]) => (
      <div key={title} className="mb-6">
        {!collapsed && <h3 className="mb-3 px-3 text-xs font-semibold text-muted-foreground">{t(`nav.group.${title}`)}</h3>}
        <nav aria-label={t(`nav.group.${title}`)} className="space-y-1">
          {items.map(({ name, href, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              aria-label={t(`nav.item.${name}`)}
              aria-current={pathname === href ? 'page' : undefined}
              title={collapsed ? t(`nav.item.${name}`) : undefined}
              className={cn(
                'flex min-h-11 items-center rounded-xl text-sm font-medium transition-colors',
                collapsed ? 'justify-center px-2' : 'gap-3 px-3',
                pathname === href
                  ? 'bg-primary/10 text-primary ring-1 ring-inset ring-primary/25'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              )}
              onClick={() => setIsOpen(false)}
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{t(`nav.item.${name}`)}</span>}
            </Link>
          ))}
        </nav>
      </div>
    ));

  return (
    <div className="min-h-screen bg-background">
      <a href="#main-content" className="skip-link">{isRtl ? 'رفتن به محتوای اصلی' : 'Skip to content'}</a>
      <header className="border-b border-border bg-card lg:hidden">
        <div className="flex min-h-16 items-center justify-between gap-2 px-3">
          <Link href="/" aria-label={isRtl ? 'خانه اختاپوس' : 'Octopus home'} className="shrink-0">
            <OctopusLogo size={40} showText={false} />
          </Link>
          <div className="flex items-center gap-1">
            <CommandPaletteTrigger onOpen={() => setCommandOpen(true)} iconOnly />
            <NotificationCenter />
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" aria-label={isRtl ? 'باز کردن منو' : 'Open navigation'}>
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side={isRtl ? 'right' : 'left'} className="w-[min(22rem,90vw)] overflow-y-auto pb-[calc(1.5rem+env(safe-area-inset-bottom))]">
                <SheetTitle className="mt-10">{isRtl ? 'منوی اختاپوس' : 'Octopus navigation'}</SheetTitle>
                <SheetDescription className="mb-6 mt-2">{isRtl ? 'بازار، تحلیل و مدیریت حساب' : 'Markets, research and account tools'}</SheetDescription>
                {renderGroups(leftSidebarItems, false)}
                {renderGroups(rightSidebarItems, false)}
                <div className="flex items-center justify-between gap-2 border-t pt-4">
                  <ThemeSwitcher />
                  <LanguageSwitcher />
                  <UserMenu />
                </div>
              </SheetContent>
            </Sheet>
            <UserMenu />
          </div>
        </div>
        <NewsTicker />
      </header>

      <aside className={cn('fixed inset-y-0 left-0 z-40 hidden flex-col border-r border-border bg-card lg:flex', leftCollapsed ? 'w-16' : 'w-64')}>
        <div className="flex h-20 shrink-0 items-center justify-center border-b border-border">
          <CommandPaletteTrigger onOpen={() => setCommandOpen(true)} iconOnly={leftCollapsed} />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2 pt-5">{renderGroups(leftSidebarItems, leftCollapsed)}</div>
        <div className="space-y-3 border-t border-border p-2">
          <div className={cn('flex items-center gap-2', leftCollapsed ? 'flex-col' : 'justify-between')}>
            <ThemeSwitcher showLabel={!leftCollapsed} />
            <LanguageSwitcher />
            <UserMenu />
          </div>
          <Button variant="ghost" size="icon" className="w-full" onClick={() => setLeftCollapsed(!leftCollapsed)} aria-expanded={!leftCollapsed} aria-label={isRtl ? 'باز و بسته کردن منوی معاملات' : 'Toggle trading sidebar'}>
            {leftCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>
      </aside>

      <div className={cn('min-w-0 w-full', leftCollapsed ? 'lg:pl-16' : 'lg:pl-64', rightCollapsed ? 'lg:pr-16' : 'lg:pr-64')}>
        <div className="sticky top-0 z-30 hidden lg:block"><NewsTicker /></div>
        <main id="main-content" tabIndex={-1} className="py-6 pb-[calc(6rem+env(safe-area-inset-bottom))] focus:outline-none lg:pb-8">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">{children}</div>
        </main>
      </div>

      <aside className={cn('fixed inset-y-0 right-0 z-40 hidden flex-col border-l border-border bg-card lg:flex', rightCollapsed ? 'w-16' : 'w-64')}>
        <div className="flex h-20 shrink-0 items-center justify-center border-b border-border px-2">
          {rightCollapsed ? <OctopusLogo size={36} showText={false} /> : <h2 className="text-sm font-semibold">{t('nav.title.Tools & System')}</h2>}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-2 pt-5">{renderGroups(rightSidebarItems, rightCollapsed)}</div>
        <div className="border-t border-border p-2">
          <Button variant="ghost" size="icon" className="w-full" onClick={() => setRightCollapsed(!rightCollapsed)} aria-expanded={!rightCollapsed} aria-label={isRtl ? 'باز و بسته کردن منوی ابزارها' : 'Toggle tools sidebar'}>
            {rightCollapsed ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </Button>
        </div>
      </aside>

      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} />
      <nav aria-label={isRtl ? 'دسترسی سریع' : 'Quick navigation'} className="fixed inset-x-0 bottom-0 z-40 border-t border-border bg-card/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl lg:hidden">
        <div className="mx-auto flex h-16 max-w-lg items-stretch px-2">
          {[
            { href: '/dashboard', icon: BarChart3, label: t('nav.item.Dashboard') },
            { href: '/trading', icon: TrendingUp, label: t('nav.item.Command Center') },
            { href: '/portfolio', icon: PieChart, label: isRtl ? 'پرتفولیو' : 'Portfolio' },
            { href: '/technical', icon: Target, label: t('nav.item.Technical') },
          ].map(({ href, icon: Icon, label }) => (
            <Link key={href} href={href} aria-current={pathname === href ? 'page' : undefined} className={cn('flex min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-xl px-1 text-xs font-medium transition-colors', pathname === href ? 'text-primary' : 'text-muted-foreground hover:text-foreground')}>
              <span className={cn('rounded-xl px-4 py-1', pathname === href && 'bg-primary/10')}><Icon className="h-5 w-5" /></span>
              <span className="max-w-full truncate">{label}</span>
            </Link>
          ))}
        </div>
      </nav>
    </div>
  );
}
