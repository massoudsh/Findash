'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, GlassCard, ElevatedCard } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { formatCurrency, formatPercentage } from '@/lib/utils';
import {
  DashboardBarChart,
  DashboardWaterfallChart,
  DashboardPieChart,
  DashboardLineChart,
} from '@/components/dashboard/dashboard-charts';

// Lazy load API to avoid webpack issues
let apiLoaded = false;
let getPortfolios: () => Promise<any>;
let getTrades: (id: number) => Promise<any>;

async function loadApi() {
  if (!apiLoaded) {
    const api = await import('@/lib/services/api');
    getPortfolios = api.getPortfolios;
    getTrades = api.getTrades;
    apiLoaded = true;
  }
}
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  DollarSign, 
  Target, 
  PieChart,
  BarChart3,
  LineChart,
  AlertTriangle,
  Shield,
  Zap,
  Globe,
  Clock,
  Users,
  ArrowUpRight,
  ArrowDownRight,
  Dot,
  CreditCard,
  Star,
  Calendar,
  Building,
  Banknote
} from 'lucide-react';
import { AccountCard } from '@/components/dashboard/account-card';

interface DashboardData {
  totalPortfolioValue: number;
  activeTrades: number;
  winRate: number;
  profitLoss: number;
  portfolioChange: number;
  tradesChange: number;
  sharpeRatio: number;
  volatility: number;
  maxDrawdown: number;
  beta: number;
  totalEarnings: number;
  expenses: number;
  weeklyStats: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  icon?: string;
}

interface Position {
  symbol: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  weight: number;
  sector: string;
  icon?: string;
}

interface WalletCard {
  id: string;
  name: string;
  number: string;
  balance: number;
  type: 'visa' | 'crypto' | 'savings';
  gradient: string;
}

export function DashboardContent() {
  const [data, setData] = useState<DashboardData>({
    totalPortfolioValue: 2847231.89,
    activeTrades: 23,
    winRate: 78.4,
    profitLoss: 87458,
    portfolioChange: 4.2,
    tradesChange: 5,
    sharpeRatio: 1.67,
    volatility: 14.2,
    maxDrawdown: -8.3,
    beta: 1.12,
    totalEarnings: 510596.57,
    expenses: 4166.80,
    weeklyStats: 1257.12
  });

  const [walletCards] = useState<WalletCard[]>([
    {
      id: '1',
      name: 'Trading Account',
      number: '•••• •••• •••• 1234',
      balance: 510596.57,
      type: 'visa',
      gradient: 'from-slate-800 to-slate-900'
    },
    {
      id: '2', 
      name: 'Crypto Wallet',
      number: '•••• •••• •••• 5678',
      balance: 234567.89,
      type: 'crypto',
      gradient: 'from-blue-600 to-purple-700'
    },
    {
      id: '3',
      name: 'Savings Account', 
      number: '•••• •••• •••• 9012',
      balance: 125000.00,
      type: 'savings',
      gradient: 'from-green-600 to-emerald-700'
    }
  ]);
  
  const [marketData] = useState<MarketData[]>([
    { symbol: 'BTC', price: 67342.45, change: 1543.21, changePercent: 2.34, volume: '28.4B', icon: '₿' },
    { symbol: 'ETH', price: 3789.12, change: -124.56, changePercent: -3.18, volume: '15.2B', icon: 'Ξ' },
    { symbol: 'BNB', price: 587.89, change: 23.45, changePercent: 4.15, volume: '2.1B', icon: 'BNB' },
    { symbol: 'SPY', price: 445.67, change: 2.34, changePercent: 0.53, volume: '89.2M', icon: '📈' }
  ]);

  const [positions] = useState<Position[]>([
    { symbol: 'AAPL', shares: 500, avgPrice: 175.50, currentPrice: 189.25, marketValue: 94625, unrealizedPL: 6875, unrealizedPLPercent: 7.8, weight: 3.3, sector: 'Technology', icon: '🍎' },
    { symbol: 'MSFT', shares: 300, avgPrice: 342.10, currentPrice: 378.45, marketValue: 113535, unrealizedPL: 10905, unrealizedPLPercent: 10.6, weight: 4.0, sector: 'Technology', icon: '🏢' },
    { symbol: 'GOOGL', shares: 150, avgPrice: 2687.50, currentPrice: 2845.30, marketValue: 426795, unrealizedPL: 23670, unrealizedPLPercent: 5.9, weight: 15.0, sector: 'Technology', icon: '🔍' },
    { symbol: 'TSLA', shares: 200, avgPrice: 245.75, currentPrice: 234.50, marketValue: 46900, unrealizedPL: -2250, unrealizedPLPercent: -4.6, weight: 1.6, sector: 'Automotive', icon: '🚗' },
    { symbol: 'NVDA', shares: 100, avgPrice: 892.50, currentPrice: 1167.80, marketValue: 116780, unrealizedPL: 27530, unrealizedPLPercent: 30.8, weight: 4.1, sector: 'Technology', icon: '🎮' }
  ]);

  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    async function fetchDashboardData() {
      try {
        // Lazy load the API module
        await loadApi();
        
        const portfoliosResponse = await getPortfolios();
        const portfolios = portfoliosResponse.data;
        
        if (portfolios && portfolios.length > 0) {
          const totalValue = portfolios.reduce((sum: number, portfolio: any) => 
            sum + (portfolio.current_value || 0), 0);

          let totalActiveTrades = 0;
          for (const portfolio of portfolios) {
            try {
              const tradesResponse = await getTrades(portfolio.id);
              totalActiveTrades += tradesResponse.data.filter((trade: any) => 
                trade.status === 'open').length;
            } catch (error) {
              console.error(`Error fetching trades for portfolio ${portfolio.id}:`, error);
            }
          }

          setData(prev => ({
            ...prev,
            totalPortfolioValue: totalValue || prev.totalPortfolioValue,
            activeTrades: totalActiveTrades || prev.activeTrades,
          }));
          
          setHasError(false);
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setHasError(true);
        setErrorMessage('Live data unavailable. Displaying simulated market data.');
      } finally {
        setIsLoading(false);
      }
    }

    fetchDashboardData();
  }, []);

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  return (
    <>
    <div className="relative min-h-screen rounded-2xl overflow-hidden">
      {/* Greeny fintech modern background */}
      <div
        className="absolute inset-0 -z-10 bg-gradient-to-br from-emerald-50/90 via-green-50/70 to-teal-50/90 dark:from-emerald-950/40 dark:via-green-950/30 dark:to-teal-950/40"
        aria-hidden
      />
      <div
        className="absolute inset-0 -z-10 opacity-[0.4] dark:opacity-[0.25]"
        style={{
          backgroundImage: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(16, 185, 129, 0.25), transparent), radial-gradient(ellipse 60% 40% at 100% 50%, rgba(20, 184, 166, 0.15), transparent), radial-gradient(ellipse 50% 30% at 0% 80%, rgba(5, 150, 105, 0.15), transparent)',
        }}
        aria-hidden
      />

      <div className="space-y-6 relative">
        {hasError && (
          <Alert className="border-amber-200 bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="flex items-center gap-2">
              {errorMessage}
            </AlertDescription>
          </Alert>
        )}

        {/* Account / Wallet Cards — glassy, greeny fintech */}
        <div className="grid gap-5 grid-cols-1 sm:grid-cols-2 xl:grid-cols-3">
          {walletCards.map((wallet) => (
            <AccountCard key={wallet.id} account={wallet} className="h-full" />
          ))}
        </div>

        {/* Financial Summary Cards - Modern Style */}
        <div className="grid gap-4 md:grid-cols-4">
        <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-blue-500/10 via-blue-500/5 to-transparent border-blue-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <CardContent className="p-6 relative">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <Banknote className="h-6 w-6 text-white" />
              </div>
              <Badge variant="outline" className="text-xs text-green-600 border-green-600 bg-green-50 dark:bg-green-950/30">
                +{formatPercentage(data.portfolioChange)}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1 font-medium">Total Earnings</p>
              <p className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 dark:from-blue-400 dark:to-blue-300 bg-clip-text text-transparent">
                {formatCurrency(data.totalEarnings)}
              </p>
            </div>
          </CardContent>
        </ElevatedCard>

        <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-red-500/10 via-red-500/5 to-transparent border-red-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <CardContent className="p-6 relative">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-red-600 rounded-xl flex items-center justify-center shadow-lg shadow-red-500/20">
                <CreditCard className="h-6 w-6 text-white" />
              </div>
              <Badge variant="outline" className="text-xs text-red-600 border-red-600 bg-red-50 dark:bg-red-950/30">
                Monthly
              </Badge>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1 font-medium">Expenses</p>
              <p className="text-2xl font-bold bg-gradient-to-r from-red-600 to-red-400 dark:from-red-400 dark:to-red-300 bg-clip-text text-transparent">
                {formatCurrency(data.expenses)}
              </p>
            </div>
          </CardContent>
        </ElevatedCard>

        <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-purple-500/10 via-purple-500/5 to-transparent border-purple-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <CardContent className="p-6 relative">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <Badge variant="outline" className="text-xs text-blue-600 border-blue-600 bg-blue-50 dark:bg-blue-950/30">
                This Week
              </Badge>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1 font-medium">Weekly Stats</p>
              <p className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-purple-400 dark:from-purple-400 dark:to-purple-300 bg-clip-text text-transparent">
                {formatCurrency(data.weeklyStats)}
              </p>
            </div>
          </CardContent>
        </ElevatedCard>

        <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-green-500/10 via-green-500/5 to-transparent border-green-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <CardContent className="p-6 relative">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center shadow-lg shadow-green-500/20">
                <Target className="h-6 w-6 text-white" />
              </div>
              <Badge variant="outline" className="text-xs text-green-600 border-green-600 bg-green-50 dark:bg-green-950/30">
                {formatPercentage(data.winRate)}
              </Badge>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1 font-medium">Success Rate</p>
              <Progress value={data.winRate} className="mt-2 h-2 bg-muted" />
            </div>
          </CardContent>
        </ElevatedCard>
        </div>

        {/* Enhanced Tabbed Interface */}
        <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5 bg-muted/50">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="positions" className="flex items-center gap-2">
            <PieChart className="h-4 w-4" />
            Holdings
          </TabsTrigger>
          <TabsTrigger value="market" className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            Markets
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Activity
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* One of each: Bar, Waterfall, Pie, Line */}
          <div className="grid gap-6 md:grid-cols-2">
            <DashboardBarChart
              data={[
                { name: 'Jan', value: 125000 },
                { name: 'Feb', value: 142000 },
                { name: 'Mar', value: 138500 },
                { name: 'Apr', value: 165200 },
                { name: 'May', value: 158900 },
                { name: 'Jun', value: 184200 },
              ]}
              title="Monthly P&L"
              subtitle="Realized profit by month"
              height={260}
              barColor="#3b82f6"
            />
            <DashboardWaterfallChart
              data={[
                { name: 'Opening', start: 0, delta: 2700000 },
                { name: 'Trades', start: 2700000, delta: 87458 },
                { name: 'Dividends', start: 2787458, delta: 12450 },
                { name: 'Fees', start: 2799908, delta: -1200 },
                { name: 'Closing', start: 2798708, delta: 0 },
              ]}
              title="Cash flow waterfall"
              subtitle="Portfolio value build-up"
              height={260}
            />
            <DashboardPieChart
              data={[
                { name: 'Technology', value: 68.4, amount: 1948231 },
                { name: 'Healthcare', value: 12.8, amount: 364446 },
                { name: 'Financial', value: 9.2, amount: 261945 },
                { name: 'Consumer', value: 6.1, amount: 173681 },
                { name: 'Cash', value: 3.5, amount: 99653 },
              ]}
              title="Portfolio by sector"
              subtitle="Allocation"
              height={260}
            />
            <DashboardLineChart
              data={[
                { name: 'W1', value: 2680000 },
                { name: 'W2', value: 2715000 },
                { name: 'W3', value: 2692000 },
                { name: 'W4', value: 2758000 },
                { name: 'W5', value: 2784000 },
                { name: 'W6', value: 2847232 },
              ]}
              title="Portfolio value over time"
              subtitle="Last 6 weeks"
              height={260}
              strokeColor="#8b5cf6"
            />
          </div>
        </TabsContent>

        <TabsContent value="positions" className="space-y-4">
          <ElevatedCard>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <PieChart className="h-4 w-4 text-white" />
                </div>
                Top Holdings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {positions.map((position) => (
                  <div key={position.symbol} className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-muted/50 to-muted/30 hover:from-muted/70 hover:to-muted/50 transition-all duration-300 border border-border/50 hover:border-primary/20 hover:shadow-md">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center text-xl shadow-sm">
                        {position.icon}
                      </div>
                      <div>
                        <div className="font-semibold text-base">{position.symbol}</div>
                        <div className="text-sm text-muted-foreground font-medium">{position.sector}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-base">{formatCurrency(position.marketValue)}</div>
                      <div className={`text-sm flex items-center gap-1 font-semibold ${position.unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {position.unrealizedPL >= 0 ? (
                          <ArrowUpRight className="h-3 w-3" />
                        ) : (
                          <ArrowDownRight className="h-3 w-3" />
                        )}
                        {position.unrealizedPL >= 0 ? '+' : ''}{formatPercentage(position.unrealizedPLPercent)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </ElevatedCard>
        </TabsContent>

        <TabsContent value="market" className="space-y-4">
          <GlassCard>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                    <Globe className="h-4 w-4 text-white" />
                  </div>
                  Live Prices
                </CardTitle>
                <Button variant="outline" size="sm" className="hover:bg-primary/10">
                  See All
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {marketData.map((market) => (
                  <div key={market.symbol} className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-muted/50 to-muted/30 hover:from-muted/70 hover:to-muted/50 transition-all duration-300 border border-border/50 hover:border-primary/20 hover:shadow-md">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl flex items-center justify-center text-xl shadow-sm">
                        {market.icon}
                      </div>
                      <div>
                        <div className="font-semibold text-base">{market.symbol}</div>
                        <div className="text-xs text-muted-foreground font-medium">
                          {market.symbol === 'BTC' ? 'Bitcoin' : 
                           market.symbol === 'ETH' ? 'Ethereum' : 
                           market.symbol === 'BNB' ? 'BNB' : 'S&P 500 ETF'}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-lg bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">${market.price.toLocaleString()}</div>
                      <div className={`text-sm flex items-center gap-1 font-semibold ${market.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {market.change >= 0 ? (
                          <ArrowUpRight className="h-3 w-3" />
                        ) : (
                          <ArrowDownRight className="h-3 w-3" />
                        )}
                        {market.change >= 0 ? '+' : ''}{market.changePercent.toFixed(2)}%
                      </div>
                    </div>
                    <div className="w-20 h-10 flex items-end gap-1">
                      {[...Array(8)].map((_, i) => (
                        <div 
                          key={i}
                          className={`flex-1 rounded-sm ${market.change >= 0 ? 'bg-green-500' : 'bg-red-500'} opacity-60`}
                          style={{ height: `${Math.random() * 100}%` }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </GlassCard>
        </TabsContent>

        <TabsContent value="activity" className="space-y-4">
          <GlassCard>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                  <Clock className="h-4 w-4 text-white" />
                </div>
                Trading Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { action: 'BOUGHT', symbol: 'NVDA', shares: '50', price: '$1,167.80', time: '2 minutes ago', type: 'buy' },
                  { action: 'SOLD', symbol: 'TSLA', shares: '25', price: '$234.50', time: '15 minutes ago', type: 'sell' },
                  { action: 'ALERT', symbol: 'AAPL', shares: '', price: 'Stop loss triggered', time: '1 hour ago', type: 'alert' },
                  { action: 'BOUGHT', symbol: 'MSFT', shares: '100', price: '$378.45', time: '2 hours ago', type: 'buy' },
                  { action: 'DIVIDEND', symbol: 'JNJ', shares: '', price: '$124.50 received', time: '1 day ago', type: 'dividend' }
                ].map((activity, index) => (
                  <div key={index} className="flex items-center gap-4 p-4 rounded-xl bg-gradient-to-r from-muted/50 to-muted/30 hover:from-muted/70 hover:to-muted/50 transition-all duration-300 border border-border/50 hover:border-primary/20 hover:shadow-md">
                    <div className={`w-2 h-2 rounded-full ${
                      activity.type === 'buy' ? 'bg-green-500' : 
                      activity.type === 'sell' ? 'bg-red-500' : 
                      activity.type === 'alert' ? 'bg-yellow-500' : 'bg-blue-500'
                    }`} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant={activity.type === 'buy' ? 'default' : activity.type === 'sell' ? 'destructive' : 'secondary'}>
                          {activity.action}
                        </Badge>
                        <span className="font-medium">{activity.symbol}</span>
                        {activity.shares && <span className="text-muted-foreground">{activity.shares} shares</span>}
                      </div>
                      <div className="text-sm text-muted-foreground">{activity.price}</div>
                    </div>
                    <div className="text-xs text-muted-foreground">{activity.time}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </GlassCard>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-4">
            <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-blue-500/10 via-blue-500/5 to-transparent border-blue-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                    <Shield className="h-3.5 w-3.5 text-white" />
                  </div>
                  Sharpe Ratio
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 dark:from-blue-400 dark:to-blue-300 bg-clip-text text-transparent">{data.sharpeRatio}</div>
                <Badge variant={data.sharpeRatio > 1 ? "default" : "secondary"} className="mt-2 bg-primary/10 border-primary/20">
                  {data.sharpeRatio > 1 ? "Excellent" : "Good"}
                </Badge>
              </CardContent>
            </ElevatedCard>
            
            <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-purple-500/10 via-purple-500/5 to-transparent border-purple-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <BarChart3 className="h-3.5 w-3.5 text-white" />
                  </div>
                  Volatility
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-purple-400 dark:from-purple-400 dark:to-purple-300 bg-clip-text text-transparent">{data.volatility}%</div>
                <Progress value={data.volatility} max={30} className="mt-2 h-2 bg-muted" />
              </CardContent>
            </ElevatedCard>
            
            <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-red-500/10 via-red-500/5 to-transparent border-red-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-red-500 to-red-600 rounded-lg flex items-center justify-center">
                    <ArrowDownRight className="h-3.5 w-3.5 text-white" />
                  </div>
                  Max Drawdown
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold bg-gradient-to-r from-red-600 to-red-400 dark:from-red-400 dark:to-red-300 bg-clip-text text-transparent">{data.maxDrawdown}%</div>
                <p className="text-xs text-muted-foreground font-medium mt-1">Last 12 months</p>
              </CardContent>
            </ElevatedCard>
            
            <ElevatedCard className="relative overflow-hidden group bg-gradient-to-br from-green-500/10 via-green-500/5 to-transparent border-green-500/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <div className="w-7 h-7 bg-gradient-to-br from-green-500 to-green-600 rounded-lg flex items-center justify-center">
                    <LineChart className="h-3.5 w-3.5 text-white" />
                  </div>
                  Beta
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold bg-gradient-to-r from-green-600 to-green-400 dark:from-green-400 dark:to-green-300 bg-clip-text text-transparent">{data.beta}</div>
                <Badge variant={data.beta < 1 ? "outline" : "default"} className="mt-2 bg-primary/10 border-primary/20">
                  {data.beta < 1 ? "Defensive" : "Aggressive"}
                </Badge>
              </CardContent>
            </ElevatedCard>
          </div>
        </TabsContent>
      </Tabs>
      </div>
    </div>
    </>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-3">
        {[...Array(3)].map((_, i) => (
          <Card key={i} className="h-40">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex justify-between">
                  <div className="h-4 w-24 bg-muted animate-pulse rounded" />
                  <div className="h-6 w-6 bg-muted animate-pulse rounded" />
                </div>
                <div className="h-8 w-32 bg-muted animate-pulse rounded" />
                <div className="flex gap-2">
                  <div className="h-6 w-16 bg-muted animate-pulse rounded" />
                  <div className="h-6 w-16 bg-muted animate-pulse rounded" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      <div className="grid gap-4 md:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <div className="h-4 w-24 bg-muted animate-pulse rounded" />
            </CardHeader>
            <CardContent>
              <div className="h-6 w-16 bg-muted animate-pulse rounded mb-2" />
              <div className="h-4 w-12 bg-muted animate-pulse rounded" />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="space-y-4">
        <div className="h-10 w-full bg-muted animate-pulse rounded" />
        <div className="grid gap-4 md:grid-cols-2">
          {[...Array(2)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <div className="h-5 w-32 bg-muted animate-pulse rounded" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[...Array(4)].map((_, j) => (
                    <div key={j} className="flex items-center justify-between">
                      <div className="h-4 w-24 bg-muted animate-pulse rounded" />
                      <div className="h-4 w-16 bg-muted animate-pulse rounded" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
} 