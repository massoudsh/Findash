'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { OctopusLogo } from '@/components/ui/octopus-logo';
import { 
  BarChart3, 
  Briefcase, 
  Menu, 
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
  Percent
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { UserMenu } from "@/components/navigation/user-menu";
import { GlobalSearch } from "@/components/search/global-search";
import { ThemeSwitcher } from "@/components/ui/theme-switcher";
import { NotificationCenter } from "@/components/ui/notification-center";
import { CommandPalette, CommandPaletteTrigger } from "@/components/ui/command-palette";

interface NavigationWrapperProps {
  children: React.ReactNode;
}

// Left Sidebar - Trading & Portfolio Management
const leftSidebarItems = {
  'Trading': [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'Market', href: '/realtime', icon: Activity },
    { name: 'Options', href: '/options', icon: DollarSign },
    { name: 'Trading Center', href: '/trades', icon: TrendingUp },
    { name: 'Trading Bots', href: '/trading-bots', icon: Brain },
    { name: 'Paper Trading & Backtesting', href: '/backtesting', icon: FlaskConical },
  ],
  'Portfolio': [
    { name: 'Portfolio', href: '/portfolio', icon: Briefcase },
    { name: 'Strategies', href: '/strategies', icon: Target },
    { name: 'Risk Assessment', href: '/risk', icon: Shield },
  ],
};

// Right Sidebar - Analysis & Tools
const rightSidebarItems = {
  'Analysis': [
    { name: 'Technical', href: '/technical', icon: Target },
    { name: 'Fundamental Research', href: '/fundamental-data', icon: Brain },
    { name: 'Macro', href: '/macro', icon: TrendingUp },
    { name: 'On-chain', href: '/on-chain', icon: Database },
    { name: 'Social Signals', href: '/social', icon: MessageSquare },
    { name: 'AI Models', href: '/ai-models', icon: Brain },
  ],
  'Tools & Data': [
    { name: 'Data Explorer', href: '/data-explorer', icon: Database },
    { name: 'Visualization', href: '/visualization', icon: PieChart },
    { name: 'Reports', href: '/reports', icon: FileText },
    { name: 'API Playground', href: '/api-playground', icon: Activity },
  ],
  'System': [
    { name: 'Notifications', href: '/notifications', icon: Bell },
    { name: 'Admin Panel', href: '/admin', icon: ServerCog },
    { name: 'Audit Log', href: '/audit-log', icon: ListChecks },
    { name: 'Profile', href: '/profile', icon: User },
    { name: 'Settings', href: '/settings', icon: SettingsIcon },
    { name: 'Help', href: '/help', icon: BookOpen },
  ],
};

export function NavigationWrapper({ children }: NavigationWrapperProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const pathname = usePathname();

  type NavItem = { name: string; href: string; icon: React.ComponentType<{ className?: string }> };
  
  const NavigationGroup = ({ title, items }: { title: string; items: NavItem[] }) => (
    <div className="mb-6">
      <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      <nav className="space-y-1">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center space-x-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                pathname === item.href
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )}
              onClick={() => setIsOpen(false)}
            >
              <Icon className="h-4 w-4" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );

  const LeftNavigationContent = () => (
    <div>
      {Object.entries(leftSidebarItems).map(([title, items]) => (
        <NavigationGroup key={title} title={title} items={items} />
      ))}
    </div>
  );

  const RightNavigationContent = () => (
    <div>
      {Object.entries(rightSidebarItems).map(([title, items]) => (
        <NavigationGroup key={title} title={title} items={items} />
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Navigation */}
      <div className="lg:hidden">
        <div className="flex items-center justify-between border-b px-4 py-4">
          <OctopusLogo size={48} showText={false} />
          <div className="flex items-center gap-4">
            <CommandPaletteTrigger onOpen={() => setCommandOpen(true)} />
            <NotificationCenter />
            <ThemeSwitcher />
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Menu className="h-6 w-6" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64">
                <div className="mt-8">
                  <LeftNavigationContent />
                </div>
              </SheetContent>
            </Sheet>
            <UserMenu />
          </div>
        </div>
      </div>

      <div className="lg:flex">
        {/* Left Desktop Sidebar - Trading & Portfolio */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-40 lg:w-64 lg:flex lg:flex-col">
          <div className="flex grow flex-col gap-y-5 overflow-y-auto border-r bg-card px-6 pb-4">
            <div className="flex h-16 shrink-0 items-center">
              <OctopusLogo size={56} showText={true} textSize="xl" />
            </div>
            <div className="px-2">
              <CommandPaletteTrigger onOpen={() => setCommandOpen(true)} />
            </div>
            <LeftNavigationContent />
            <div className="mt-auto">
              <div className="flex items-center justify-between mb-4">
                <ThemeSwitcher showLabel />
              </div>
              <UserMenu />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:pl-64 lg:pr-64">
          <main className="py-6">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              {children}
            </div>
          </main>
        </div>

        {/* Right Desktop Sidebar - Analysis & Tools */}
        <div className="hidden lg:fixed lg:inset-y-0 lg:right-0 lg:z-40 lg:w-64 lg:flex lg:flex-col">
          <div className="flex grow flex-col gap-y-5 overflow-y-auto border-l bg-card px-6 pb-4">
            <div className="flex h-16 shrink-0 items-center justify-center">
              <h2 className="text-lg font-semibold text-muted-foreground">Analysis & Tools</h2>
            </div>
            <RightNavigationContent />
          </div>
        </div>
      </div>
      
      {/* Command Palette */}
      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} />
    </div>
  );
} 