'use client';

import { DashboardContent } from '@/components/dashboard/dashboard-content';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, GlassCard } from '@/components/ui/card';
import { 
  RefreshCw as Refresh, 
  Download, 
  Settings, 
  Bell,
  TrendingUp,
  Users,
  Activity
} from 'lucide-react';

export default function DashboardPage() {
  const currentTime = new Date().toLocaleString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'America/New_York'
  });

  return (
    <div className="min-h-screen bg-background">
      {/* Header Section */}
      <div className="border-b bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between mb-4">
            <div className="min-w-0">
              <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-foreground antialiased">
                Command center
              </h1>
              <p className="text-sm text-muted-foreground mt-1.5 tracking-normal">
                Portfolio at a glance · live data
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="text-xs">
                <Activity className="h-3 w-3 mr-1" />
                Live Market Data
              </Badge>
              <Button variant="outline" size="sm">
                <Refresh className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
          
          {/* Market Status Bar */}
          <GlassCard className="bg-gradient-to-r from-green-500/10 via-blue-500/10 to-green-500/10 border-green-500/20 dark:border-green-500/10">
            <CardContent className="p-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full animate-pulse shadow-lg shadow-green-500/50" />
                    <span className="text-sm font-semibold bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400 bg-clip-text text-transparent">Market Open</span>
                  </div>
                  <div className="text-xs text-muted-foreground font-medium">
                    {currentTime} EST
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-green-500/10 border border-green-500/20">
                      <TrendingUp className="h-3 w-3 text-green-600" />
                      <span className="font-semibold">S&P 500: +0.8%</span>
                    </div>
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-green-500/10 border border-green-500/20">
                      <TrendingUp className="h-3 w-3 text-green-600" />
                      <span className="font-semibold">NASDAQ: +1.2%</span>
                    </div>
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <Users className="h-3 w-3 text-blue-600" />
                      <span className="font-semibold">247 Active Traders</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" className="hover:bg-primary/10">
                    <Bell className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </GlassCard>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <DashboardContent />
      </div>
    </div>
  );
} 