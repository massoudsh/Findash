"use client"

import * as React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  TrendingUp, 
  DollarSign, 
  Calculator, 
  BarChart3, 
  Users, 
  Bell, 
  Download,
  Plus,
  RefreshCw,
  Activity,
  Target,
  Zap
} from "lucide-react"
import { useRouter } from "next/navigation"

interface QuickAction {
  id: string
  title: string
  description: string
  icon: React.ReactNode
  action: () => void
  category: 'trading' | 'analysis' | 'data' | 'tools'
  badge?: string
  disabled?: boolean
}

export function QuickActions() {
  const router = useRouter()

  const quickActions: QuickAction[] = [
    // Trading Actions
    {
      id: 'new-trade',
      title: 'New Trade',
      description: 'Place a buy/sell order',
      icon: <DollarSign className="w-5 h-5" />,
      action: () => router.push('/trades?action=new'),
      category: 'trading',
      badge: 'Hot'
    },
    {
      id: 'portfolio-overview',
      title: 'Portfolio Overview',
      description: 'View current positions',
      icon: <BarChart3 className="w-5 h-5" />,
      action: () => router.push('/portfolio'),
      category: 'trading'
    },
    {
      id: 'market-watch',
      title: 'Market Watch',
      description: 'Live market data',
      icon: <Activity className="w-5 h-5" />,
      action: () => router.push('/realtime'),
      category: 'trading'
    },

    // Analysis Actions
    {
      id: 'run-backtest',
      title: 'Run Backtest',
      description: 'Test strategy performance',
      icon: <Calculator className="w-5 h-5" />,
      action: () => router.push('/strategies?tab=backtesting&action=new'),
      category: 'analysis'
    },
    {
      id: 'risk-analysis',
      title: 'Risk Analysis',
      description: 'Portfolio risk assessment',
      icon: <Target className="w-5 h-5" />,
      action: () => router.push('/risk'),
      category: 'analysis'
    },
    {
      id: 'performance-report',
      title: 'Performance Report',
      description: 'Generate detailed report',
      icon: <TrendingUp className="w-5 h-5" />,
      action: () => router.push('/reports?type=performance'),
      category: 'analysis'
    },

    // Data Actions
    {
      id: 'data-explorer',
      title: 'Data Explorer',
      description: 'Browse all data sources',
      icon: <BarChart3 className="w-5 h-5" />,
      action: () => router.push('/data-explorer'),
      category: 'data',
      badge: 'New'
    },
    {
      id: 'export-data',
      title: 'Export Data',
      description: 'Download portfolio data',
      icon: <Download className="w-5 h-5" />,
      action: () => {
        // Mock export functionality
        console.log('Starting data export...')
      },
      category: 'data'
    },
    {
      id: 'refresh-data',
      title: 'Refresh Data',
      description: 'Update market data',
      icon: <RefreshCw className="w-5 h-5" />,
      action: () => {
        // Mock refresh functionality
        console.log('Refreshing market data...')
      },
      category: 'data'
    },

    // Tools Actions
    {
      id: 'create-strategy',
      title: 'Create Strategy',
      description: 'Build new trading strategy',
      icon: <Zap className="w-5 h-5" />,
      action: () => router.push('/strategies?action=create'),
      category: 'tools'
    },
    {
      id: 'social-sentiment',
      title: 'Social Sentiment',
      description: 'Check market sentiment',
      icon: <Users className="w-5 h-5" />,
      action: () => router.push('/social'),
      category: 'tools'
    },
    {
      id: 'alerts-setup',
      title: 'Setup Alerts',
      description: 'Configure notifications',
      icon: <Bell className="w-5 h-5" />,
      action: () => router.push('/notifications?action=create'),
      category: 'tools'
    }
  ]

  const groupedActions = quickActions.reduce((acc, action) => {
    if (!acc[action.category]) {
      acc[action.category] = []
    }
    acc[action.category].push(action)
    return acc
  }, {} as Record<string, QuickAction[]>)

  const categoryTitles = {
    trading: 'Trading',
    analysis: 'Analysis',
    data: 'Data Management',
    tools: 'Tools & Utilities'
  }

  const categoryIcons = {
    trading: <DollarSign className="w-4 h-4" />,
    analysis: <BarChart3 className="w-4 h-4" />,
    data: <Activity className="w-4 h-4" />,
    tools: <Zap className="w-4 h-4" />
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Quick Actions</h2>
          <p className="text-muted-foreground">
            Fast access to commonly used features and tools
          </p>
        </div>
      </div>

      {Object.entries(groupedActions).map(([category, actions]) => (
        <Card key={category}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {categoryIcons[category as keyof typeof categoryIcons]}
              {categoryTitles[category as keyof typeof categoryTitles]}
            </CardTitle>
            <CardDescription>
              {actions.length} actions available
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {actions.map((action) => (
                <div
                  key={action.id}
                  className="group relative rounded-lg border p-4 hover:shadow-md transition-all cursor-pointer"
                  onClick={action.action}
                >
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 p-2 rounded-md bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                      {action.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-sm">{action.title}</h3>
                        {action.badge && (
                          <Badge variant="secondary" className="text-xs">
                            {action.badge}
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {action.description}
                      </p>
                    </div>
                  </div>
                  {action.disabled && (
                    <div className="absolute inset-0 bg-background/50 rounded-lg flex items-center justify-center">
                      <Badge variant="outline">Coming Soon</Badge>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
} 