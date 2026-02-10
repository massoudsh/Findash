"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Search, Calculator, BarChart3, TrendingUp, DollarSign, Users, Settings, Bell } from "lucide-react"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { Button } from "@/components/ui/button"

interface SearchItem {
  id: string
  title: string
  description?: string
  icon: React.ReactNode
  url: string
  category: string
}

const searchItems: SearchItem[] = [
  // Navigation
  { id: 'dashboard', title: 'Dashboard', description: 'Trading overview and analytics', icon: <BarChart3 className="w-4 h-4" />, url: '/dashboard', category: 'Navigation' },
  { id: 'portfolio', title: 'Portfolio', description: 'View and manage your portfolio', icon: <DollarSign className="w-4 h-4" />, url: '/portfolio', category: 'Navigation' },
  { id: 'trades', title: 'Trades', description: 'Trading history and orders', icon: <TrendingUp className="w-4 h-4" />, url: '/trades', category: 'Navigation' },
  { id: 'strategies', title: 'Strategies', description: 'Trading strategies and backtesting', icon: <Calculator className="w-4 h-4" />, url: '/strategies', category: 'Navigation' },
  { id: 'risk', title: 'Risk Management', description: 'Portfolio risk analysis', icon: <BarChart3 className="w-4 h-4" />, url: '/risk', category: 'Navigation' },
  { id: 'backtesting', title: 'Backtesting', description: 'Strategy backtesting tools', icon: <TrendingUp className="w-4 h-4" />, url: '/backtesting', category: 'Navigation' },
  { id: 'models', title: 'AI Models', description: 'Machine learning models', icon: <Calculator className="w-4 h-4" />, url: '/models', category: 'Navigation' },
  { id: 'visualization', title: 'Visualization', description: 'Charts and data visualization', icon: <BarChart3 className="w-4 h-4" />, url: '/visualization', category: 'Navigation' },
  { id: 'realtime', title: 'Real-time Data', description: 'Live market data', icon: <TrendingUp className="w-4 h-4" />, url: '/realtime', category: 'Navigation' },
  { id: 'social', title: 'Social Trading', description: 'Social sentiment and analysis', icon: <Users className="w-4 h-4" />, url: '/social', category: 'Navigation' },
  
  // Quick Actions
  { id: 'new-trade', title: 'New Trade', description: 'Place a new trade order', icon: <DollarSign className="w-4 h-4" />, url: '/trades?action=new', category: 'Quick Actions' },
  { id: 'create-strategy', title: 'Create Strategy', description: 'Build a new trading strategy', icon: <Calculator className="w-4 h-4" />, url: '/strategies?action=create', category: 'Quick Actions' },
  { id: 'run-backtest', title: 'Run Backtest', description: 'Test strategy performance', icon: <TrendingUp className="w-4 h-4" />, url: '/backtesting?action=new', category: 'Quick Actions' },
  { id: 'portfolio-analysis', title: 'Portfolio Analysis', description: 'Analyze portfolio performance', icon: <BarChart3 className="w-4 h-4" />, url: '/portfolio?tab=analysis', category: 'Quick Actions' },
  
  // Settings
  { id: 'settings', title: 'Settings', description: 'Application settings', icon: <Settings className="w-4 h-4" />, url: '/settings', category: 'Settings' },
  { id: 'notifications', title: 'Notifications', description: 'Manage notifications', icon: <Bell className="w-4 h-4" />, url: '/notifications', category: 'Settings' },
  { id: 'profile', title: 'Profile', description: 'User profile settings', icon: <Users className="w-4 h-4" />, url: '/profile', category: 'Settings' },
]

export function GlobalSearch() {
  const [open, setOpen] = React.useState(false)
  const router = useRouter()

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((open) => !open)
      }
      if (e.key === "Escape") {
        setOpen(false)
      }
    }
    document.addEventListener("keydown", down)
    return () => document.removeEventListener("keydown", down)
  }, [])

  const runCommand = React.useCallback((command: () => unknown) => {
    setOpen(false)
    command()
  }, [])

  return (
    <>
      <Button
        variant="outline"
        className="relative w-full justify-start text-sm text-muted-foreground sm:pr-12 md:w-40 lg:w-64"
        onClick={() => setOpen(true)}
      >
        <Search className="mr-2 h-4 w-4" />
        Search...
        <kbd className="pointer-events-none absolute right-1.5 top-1.5 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>
      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Type a command or search..." />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>
          {Object.entries(
            searchItems.reduce((acc, item) => {
              if (!acc[item.category]) {
                acc[item.category] = []
              }
              acc[item.category].push(item)
              return acc
            }, {} as Record<string, SearchItem[]>)
          ).map(([category, items]) => (
            <CommandGroup key={category} heading={category}>
              {items.map((item) => (
                <CommandItem
                  key={item.id}
                  onSelect={() => {
                    runCommand(() => router.push(item.url))
                  }}
                >
                  {item.icon}
                  <div className="ml-2 flex flex-col">
                    <span>{item.title}</span>
                    {item.description && (
                      <span className="text-xs text-muted-foreground">
                        {item.description}
                      </span>
                    )}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          ))}
        </CommandList>
      </CommandDialog>
    </>
  )
} 