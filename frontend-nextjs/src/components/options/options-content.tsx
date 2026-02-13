'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { toast } from '@/components/ui/toast';
import { 
  Plus,
  Target, 
  Calculator,
  PlayCircle,
  StopCircle,
  Archive,
  BarChart3,
  Trash2,
  Eye,
  TrendingUp,
  Filter,
  Settings,
  RefreshCw,
  ChevronRight,
  Zap,
  Shield,
  Sparkles,
  Bell,
  Search,
  MoreHorizontal,
  ArrowUpRight,
  ArrowDownLeft,
  Calendar,
  Bookmark,
  Activity,
  Layers,
  ExternalLink,
  BookOpen,
  DollarSign
} from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';

interface OptionPosition {
  id: string;
  symbol: string;
  optionType: 'call' | 'put';
  action: 'buy' | 'sell';
  strike: number;
  expiration: string;
  premium: number;
  quantity: number;
  currentPrice: number;
  openDate: string;
  closeDate?: string;
  status: 'open' | 'closed' | 'expired';
  underlyingPrice: number;
  currentUnderlyingPrice: number;
  unrealizedPnL?: number;
  realizedPnL?: number;
  theta: number;
  delta: number;
  gamma: number;
  vega: number;
  impliedVolatility: number;
  daysToExpiration: number;
}

interface PositionFormData {
  symbol: string;
  optionType: 'call' | 'put';
  action: 'buy' | 'sell';
  strike: string;
  expiration: string;
  premium: string;
  quantity: string;
}

// Mock data with enhanced fields
const portfolioMetrics = {
  totalValue: 125420.50,
  dailyPnL: 2340.25,
  dailyPnLPercent: 1.9,
  weeklyPnL: -1250.75,
  weeklyPnLPercent: -0.99,
  monthlyPnL: 8750.30,
  monthlyPnLPercent: 7.5,
  positions: 12,
  contracts: 48
}



export const strategyTemplates = [
  // Basic Strategies
  {
    name: 'Long Call',
    description: 'Bullish strategy with unlimited upside potential',
    complexity: 'Beginner',
    profitPotential: 'Unlimited',
    riskLevel: 'Medium',
    icon: '📈',
    popularity: 95,
    category: 'Directional',
    maxRisk: 'Premium Paid',
    maxProfit: 'Unlimited',
    breakeven: 'Strike + Premium',
    legs: 1,
    marketOutlook: 'Bullish',
    volatilityOutlook: 'High',
    timeDecay: 'Negative',
    marginRequired: false
  },
  {
    name: 'Long Put',
    description: 'Bearish strategy with high profit potential',
    complexity: 'Beginner',
    profitPotential: 'High',
    riskLevel: 'Medium',
    icon: '📉',
    popularity: 90,
    category: 'Directional',
    maxRisk: 'Premium Paid',
    maxProfit: 'Strike - Premium',
    breakeven: 'Strike - Premium',
    legs: 1,
    marketOutlook: 'Bearish',
    volatilityOutlook: 'High',
    timeDecay: 'Negative',
    marginRequired: false
  },
  {
    name: 'Covered Call',
    description: 'Generate income from stock holdings',
    complexity: 'Beginner',
    profitPotential: 'Limited',
    riskLevel: 'Low',
    icon: '☂️',
    popularity: 92,
    category: 'Income',
    maxRisk: 'Stock Price - Premium',
    maxProfit: 'Premium + (Strike - Stock Price)',
    breakeven: 'Stock Price - Premium',
    legs: 2,
    marketOutlook: 'Neutral to Bullish',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: false
  },
  {
    name: 'Cash-Secured Put',
    description: 'Generate income while prepared to buy stock',
    complexity: 'Beginner',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '💰',
    popularity: 88,
    category: 'Income',
    maxRisk: 'Strike - Premium',
    maxProfit: 'Premium',
    breakeven: 'Strike - Premium',
    legs: 1,
    marketOutlook: 'Neutral to Bullish',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },

  // Spread Strategies
  {
    name: 'Bull Call Spread',
    description: 'Limited risk bullish strategy',
    complexity: 'Intermediate',
    profitPotential: 'Limited',
    riskLevel: 'Low',
    icon: '🐂',
    popularity: 85,
    category: 'Spread',
    maxRisk: 'Net Premium Paid',
    maxProfit: 'Strike Difference - Net Premium',
    breakeven: 'Lower Strike + Net Premium',
    legs: 2,
    marketOutlook: 'Moderately Bullish',
    volatilityOutlook: 'Low to Medium',
    timeDecay: 'Neutral',
    marginRequired: false
  },
  {
    name: 'Bear Put Spread',
    description: 'Limited risk bearish strategy',
    complexity: 'Intermediate',
    profitPotential: 'Limited',
    riskLevel: 'Low',
    icon: '🐻',
    popularity: 82,
    category: 'Spread',
    maxRisk: 'Net Premium Paid',
    maxProfit: 'Strike Difference - Net Premium',
    breakeven: 'Higher Strike - Net Premium',
    legs: 2,
    marketOutlook: 'Moderately Bearish',
    volatilityOutlook: 'Low to Medium',
    timeDecay: 'Neutral',
    marginRequired: false
  },
  {
    name: 'Iron Condor',
    description: 'Neutral strategy for range-bound markets',
    complexity: 'Advanced',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🦅',
    popularity: 85,
    category: 'Neutral',
    maxRisk: 'Strike Width - Net Credit',
    maxProfit: 'Net Credit Received',
    breakeven: 'Two breakeven points',
    legs: 4,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },
  {
    name: 'Iron Butterfly',
    description: 'High probability neutral strategy',
    complexity: 'Advanced',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🦋',
    popularity: 78,
    category: 'Neutral',
    maxRisk: 'Strike Width - Net Credit',
    maxProfit: 'Net Credit Received',
    breakeven: 'Two breakeven points',
    legs: 3,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },

  // Volatility Strategies
  {
    name: 'Long Straddle',
    description: 'Profit from high volatility moves',
    complexity: 'Intermediate',
    profitPotential: 'Unlimited',
    riskLevel: 'High',
    icon: '⚡',
    popularity: 73,
    category: 'Volatility',
    maxRisk: 'Total Premium Paid',
    maxProfit: 'Unlimited',
    breakeven: 'Strike ± Total Premium',
    legs: 2,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'High',
    timeDecay: 'Negative',
    marginRequired: false
  },
  {
    name: 'Long Strangle',
    description: 'Lower cost volatility play',
    complexity: 'Intermediate',
    profitPotential: 'Unlimited',
    riskLevel: 'High',
    icon: '🎯',
    popularity: 70,
    category: 'Volatility',
    maxRisk: 'Total Premium Paid',
    maxProfit: 'Unlimited',
    breakeven: 'Two breakeven points',
    legs: 2,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'High',
    timeDecay: 'Negative',
    marginRequired: false
  },
  {
    name: 'Short Straddle',
    description: 'Profit from low volatility',
    complexity: 'Expert',
    profitPotential: 'Limited',
    riskLevel: 'Very High',
    icon: '💥',
    popularity: 45,
    category: 'Volatility',
    maxRisk: 'Unlimited',
    maxProfit: 'Net Credit Received',
    breakeven: 'Strike ± Net Credit',
    legs: 2,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },

  // Advanced Strategies
  {
    name: 'Collar',
    description: 'Protective strategy for stock positions',
    complexity: 'Intermediate',
    profitPotential: 'Limited',
    riskLevel: 'Low',
    icon: '🛡️',
    popularity: 65,
    category: 'Protective',
    maxRisk: 'Stock Price - Put Strike + Net Premium',
    maxProfit: 'Call Strike - Stock Price - Net Premium',
    breakeven: 'Stock Price + Net Premium',
    legs: 3,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Low',
    timeDecay: 'Neutral',
    marginRequired: false
  },
  {
    name: 'Protective Put',
    description: 'Insurance for stock positions',
    complexity: 'Beginner',
    profitPotential: 'Unlimited',
    riskLevel: 'Low',
    icon: '🔒',
    popularity: 80,
    category: 'Protective',
    maxRisk: 'Stock Price - Put Strike + Premium',
    maxProfit: 'Unlimited',
    breakeven: 'Stock Price + Premium',
    legs: 2,
    marketOutlook: 'Bullish',
    volatilityOutlook: 'Any',
    timeDecay: 'Negative',
    marginRequired: false
  },

  // Algorithmic & Systematic Strategies
  {
    name: 'Martingale Options',
    description: 'Double down strategy for options trading',
    complexity: 'Expert',
    profitPotential: 'High',
    riskLevel: 'Very High',
    icon: '🎲',
    popularity: 35,
    category: 'Algorithmic',
    maxRisk: 'Exponentially Increasing',
    maxProfit: 'Initial Target',
    breakeven: 'Variable',
    legs: 'Variable',
    marketOutlook: 'Directional',
    volatilityOutlook: 'Medium',
    timeDecay: 'Negative',
    marginRequired: true,
    warning: 'High risk of total loss'
  },
  {
    name: 'Delta Neutral',
    description: 'Market neutral strategy using delta hedging',
    complexity: 'Expert',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '⚖️',
    popularity: 55,
    category: 'Algorithmic',
    maxRisk: 'Gamma Risk',
    maxProfit: 'Theta Decay',
    breakeven: 'Dynamic',
    legs: 'Multiple',
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },
  {
    name: 'Gamma Scalping',
    description: 'Profit from gamma exposure through dynamic hedging',
    complexity: 'Expert',
    profitPotential: 'Medium',
    riskLevel: 'Medium',
    icon: '🔄',
    popularity: 40,
    category: 'Algorithmic',
    maxRisk: 'Theta Decay',
    maxProfit: 'Gamma Profits',
    breakeven: 'Dynamic',
    legs: 'Multiple',
    marketOutlook: 'Neutral',
    volatilityOutlook: 'High',
    timeDecay: 'Negative',
    marginRequired: true
  },
  {
    name: 'Volatility Arbitrage',
    description: 'Exploit differences in implied vs realized volatility',
    complexity: 'Expert',
    profitPotential: 'Medium',
    riskLevel: 'Medium',
    icon: '📊',
    popularity: 30,
    category: 'Algorithmic',
    maxRisk: 'Model Risk',
    maxProfit: 'Volatility Edge',
    breakeven: 'Model Dependent',
    legs: 'Multiple',
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Variable',
    timeDecay: 'Variable',
    marginRequired: true
  },

  // Income Strategies
  {
    name: 'Wheel Strategy',
    description: 'Systematic covered call and cash-secured put cycle',
    complexity: 'Intermediate',
    profitPotential: 'Medium',
    riskLevel: 'Medium',
    icon: '🎡',
    popularity: 85,
    category: 'Income',
    maxRisk: 'Stock Assignment Risk',
    maxProfit: 'Premium Collection',
    breakeven: 'Variable',
    legs: 'Variable',
    marketOutlook: 'Neutral to Bullish',
    volatilityOutlook: 'Medium',
    timeDecay: 'Positive',
    marginRequired: true
  },
  {
    name: 'Covered Strangle',
    description: 'Enhanced income from stock positions',
    complexity: 'Advanced',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🎪',
    popularity: 60,
    category: 'Income',
    maxRisk: 'Stock Assignment Risk',
    maxProfit: 'Net Premium Received',
    breakeven: 'Two breakeven points',
    legs: 3,
    marketOutlook: 'Neutral',
    volatilityOutlook: 'Medium',
    timeDecay: 'Positive',
    marginRequired: true
  },

  // Exotic Strategies
  {
    name: 'Jade Lizard',
    description: 'Bullish strategy with no upside risk',
    complexity: 'Advanced',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🦎',
    popularity: 25,
    category: 'Advanced',
    maxRisk: 'Put Strike - Net Credit',
    maxProfit: 'Net Credit Received',
    breakeven: 'Put Strike - Net Credit',
    legs: 3,
    marketOutlook: 'Bullish',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },
  {
    name: 'Reverse Jade Lizard',
    description: 'Bearish strategy with no downside risk',
    complexity: 'Advanced',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🐉',
    popularity: 20,
    category: 'Advanced',
    maxRisk: 'Call Strike - Net Credit',
    maxProfit: 'Net Credit Received',
    breakeven: 'Call Strike + Net Credit',
    legs: 3,
    marketOutlook: 'Bearish',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  },
  {
    name: 'Christmas Tree',
    description: 'Ratio spread with limited risk',
    complexity: 'Expert',
    profitPotential: 'Limited',
    riskLevel: 'Medium',
    icon: '🎄',
    popularity: 15,
    category: 'Advanced',
    maxRisk: 'Limited',
    maxProfit: 'Limited',
    breakeven: 'Multiple points',
    legs: 4,
    marketOutlook: 'Directional',
    volatilityOutlook: 'Low',
    timeDecay: 'Positive',
    marginRequired: true
  }
];

export function OptionsContent() {
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showNewTradeModal, setShowNewTradeModal] = useState(false);
  const [showAlertsModal, setShowAlertsModal] = useState(false);
  const [showStrategyBuilder, setShowStrategyBuilder] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<any>(null);
  const router = useRouter();

  const handleNavigation = (path: string) => {
    router.push(path);
  };

  const handleRefreshData = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: "Data refreshed",
        description: "Options data has been updated successfully.",
      });
    }, 2000);
  };

  const handleFilterOptions = () => {
    toast({
      title: "Filter applied",
      description: "Options filtered by your criteria.",
    });
  };

  const handleOpenSettings = () => {
    toast({
      title: "Settings",
      description: "Options settings panel opened.",
    });
  };

  const handleAddPosition = (symbol: string) => {
    toast({
      title: "Position added",
      description: `Added ${symbol} options position to your portfolio.`,
    });
  };

  const handleBuildStrategy = (strategyName: string) => {
    const strategy = strategyTemplates.find(s => s.name === strategyName);
    if (strategy) {
      setSelectedStrategy(strategy);
      setShowStrategyBuilder(true);
    }
  };

  const handleViewDetails = (symbol: string) => {
    toast({
      title: "Option Details",
      description: `Viewing detailed analysis for ${symbol}.`,
    });
  };

  const handleLearnMore = () => {
    toast({
      title: "Educational Resources",
      description: "Opening options trading educational content.",
    });
    handleNavigation('/help');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 particles">
      {/* Header */}
      <div className="sticky top-0 z-50 glass-card border-0 rounded-none backdrop-blur-2xl">
        <div className="flex items-center justify-between p-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Target className="h-8 w-8 text-blue-400 animate-pulse-glow" />
              <h1 className="text-2xl font-bold gradient-text">Options Trading</h1>
            </div>
            <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 animate-scale-in">
              Live Market
            </Badge>
          </div>
          
          {/* Search & Controls */}
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search options..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-64 glass-card border-white/10 focus-modern"
              />
            </div>
            <Button className="btn-morphic" onClick={() => setShowAlertsModal(true)}>
              <Bell className="h-4 w-4 mr-2" />
              Alerts
            </Button>
            <Button className="btn-liquid text-white" onClick={() => setShowNewTradeModal(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Trade
            </Button>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-8">
        {/* Portfolio Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-cyan-600/20 blur-3xl" />
          <Card className="relative glass-card card-hover animate-slide-in-bottom">
            <CardContent className="p-8">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                {/* Total Portfolio Value */}
                <div className="text-center space-y-2">
                  <p className="text-sm text-gray-400 uppercase tracking-wider">Portfolio Value</p>
                  <p className="text-4xl font-bold gradient-text animate-float">
                    ${portfolioMetrics.totalValue.toLocaleString()}
                  </p>
                  <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
                    <TrendingUp className="h-3 w-3 mr-1" />
                    Active
                  </Badge>
                </div>

                {/* Daily P&L */}
                <div className="text-center space-y-2">
                  <p className="text-sm text-gray-400 uppercase tracking-wider">Daily P&L</p>
                  <div className="flex items-center justify-center space-x-2">
                    <p className={`text-2xl font-bold ${portfolioMetrics.dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${Math.abs(portfolioMetrics.dailyPnL).toLocaleString()}
                    </p>
                    {portfolioMetrics.dailyPnL >= 0 ? (
                      <ArrowUpRight className="h-5 w-5 text-green-400" />
                    ) : (
                      <ArrowDownLeft className="h-5 w-5 text-red-400" />
                    )}
                  </div>
                  <p className={`text-sm ${portfolioMetrics.dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {portfolioMetrics.dailyPnLPercent >= 0 ? '+' : ''}{portfolioMetrics.dailyPnLPercent}%
                  </p>
                </div>

                {/* Positions */}
                <div className="text-center space-y-2">
                  <p className="text-sm text-gray-400 uppercase tracking-wider">Open Positions</p>
                  <p className="text-2xl font-bold text-blue-400">{portfolioMetrics.positions}</p>
                  <p className="text-sm text-gray-400">{portfolioMetrics.contracts} contracts</p>
                </div>

                {/* Monthly P&L */}
                <div className="text-center space-y-2">
                  <p className="text-sm text-gray-400 uppercase tracking-wider">Monthly P&L</p>
                  <p className="text-2xl font-bold text-cyan-400">${portfolioMetrics.monthlyPnL.toLocaleString()}</p>
                  <p className="text-sm text-cyan-400">+{portfolioMetrics.monthlyPnLPercent}%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>



        {/* Professional Strategy Builder */}
        <Card className="glass-card card-hover">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="h-5 w-5 text-emerald-400" />
                  <span className="holographic">Professional Strategy Builder</span>
                </CardTitle>
                <CardDescription>Deploy advanced options strategies including Martingale and algorithmic systems</CardDescription>
              </div>
              <div className="flex space-x-2">
                <Select defaultValue="all">
                  <SelectTrigger className="w-44">
                    <SelectValue placeholder="Filter strategies" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    <SelectItem value="Algorithmic">🤖 Algorithmic</SelectItem>
                    <SelectItem value="Volatility">⚡ Volatility</SelectItem>
                    <SelectItem value="Income">💰 Income</SelectItem>
                    <SelectItem value="Directional">📈 Directional</SelectItem>
                    <SelectItem value="Neutral">⚖️ Neutral</SelectItem>
                    <SelectItem value="Advanced">🧠 Advanced</SelectItem>
                  </SelectContent>
                </Select>
                <Button className="btn-morphic">
                  <Sparkles className="h-4 w-4 mr-2" />
                  AI Suggest
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {strategyTemplates.map((strategy, index) => (
                <motion.div
                  key={`${strategy.name}-pro`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className="neomorphic-inset p-6 space-y-4 hover:scale-105 transition-all duration-300 cursor-pointer group relative"
                  onClick={() => handleBuildStrategy(strategy.name)}
                >
                  {/* Special indicator for high-risk strategies */}
                  {strategy.riskLevel === 'Very High' && (
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center animate-pulse">
                      <span className="text-white text-xs font-bold">!</span>
                    </div>
                  )}
                  
                  {/* Category badge */}
                  <div className="absolute top-2 left-2">
                    <Badge className={`text-xs ${
                      strategy.category === 'Algorithmic' ? 'bg-purple-500/20 text-purple-300' :
                      strategy.category === 'Volatility' ? 'bg-yellow-500/20 text-yellow-300' :
                      strategy.category === 'Income' ? 'bg-green-500/20 text-green-300' :
                      strategy.category === 'Directional' ? 'bg-blue-500/20 text-blue-300' :
                      'bg-gray-500/20 text-gray-300'
                    }`}>
                      {strategy.category}
                    </Badge>
                  </div>

                  <div className="text-center pt-4">
                    <div className="text-4xl group-hover:animate-bounce mb-2">{strategy.icon}</div>
                    <h3 className="font-bold text-lg gradient-text group-hover:text-purple-300 transition-colors">
                      {strategy.name}
                    </h3>
                    <p className="text-sm text-gray-400 line-clamp-2 mb-3">{strategy.description}</p>
                  </div>

                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-center">
                        <p className="text-gray-400">Complexity</p>
                        <Badge className={`text-xs ${
                          strategy.complexity === 'Beginner' ? 'bg-green-500/20 text-green-300' :
                          strategy.complexity === 'Intermediate' ? 'bg-yellow-500/20 text-yellow-300' :
                          strategy.complexity === 'Advanced' ? 'bg-orange-500/20 text-orange-300' :
                          'bg-red-500/20 text-red-300'
                        }`}>
                          {strategy.complexity}
                        </Badge>
                      </div>
                      <div className="text-center">
                        <p className="text-gray-400">Risk</p>
                        <Badge className={`text-xs ${
                          strategy.riskLevel === 'Low' ? 'bg-green-500/20 text-green-300' :
                          strategy.riskLevel === 'Medium' ? 'bg-yellow-500/20 text-yellow-300' :
                          strategy.riskLevel === 'High' ? 'bg-orange-500/20 text-orange-300' :
                          'bg-red-500/20 text-red-300'
                        }`}>
                          {strategy.riskLevel}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Popularity</span>
                        <span className="text-blue-400">{strategy.popularity}%</span>
                      </div>
                      <Progress 
                        value={strategy.popularity} 
                        className="h-2 bg-gray-700"
                      />
                    </div>

                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Max Profit:</span>
                        <span className="text-green-400">{strategy.maxProfit}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Max Risk:</span>
                        <span className="text-red-400">{strategy.maxRisk}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Legs:</span>
                        <span className="text-blue-400">{strategy.legs}</span>
                      </div>
                    </div>
                    
                    {strategy.warning && (
                      <div className="text-xs text-red-400 bg-red-500/10 p-2 rounded border border-red-500/20">
                        ⚠️ High Risk
                      </div>
                    )}
                  </div>

                  <Button 
                    className="w-full btn-morphic group-hover:btn-liquid group-hover:text-white transition-all duration-300"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleBuildStrategy(strategy.name);
                    }}
                  >
                    <PlayCircle className="h-4 w-4 mr-2" />
                    Deploy Strategy
                  </Button>
                </motion.div>
              ))}
            </div>

            {/* Quick Stats */}
            <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 neomorphic-inset rounded-lg">
                <p className="text-2xl font-bold text-green-400">{strategyTemplates.filter(s => s.category === 'Income').length}</p>
                <p className="text-sm text-gray-400">Income Strategies</p>
              </div>
              <div className="text-center p-4 neomorphic-inset rounded-lg">
                <p className="text-2xl font-bold text-purple-400">{strategyTemplates.filter(s => s.category === 'Algorithmic').length}</p>
                <p className="text-sm text-gray-400">Algorithmic</p>
              </div>
              <div className="text-center p-4 neomorphic-inset rounded-lg">
                <p className="text-2xl font-bold text-yellow-400">{strategyTemplates.filter(s => s.category === 'Volatility').length}</p>
                <p className="text-sm text-gray-400">Volatility Plays</p>
              </div>
              <div className="text-center p-4 neomorphic-inset rounded-lg">
                <p className="text-2xl font-bold text-red-400">{strategyTemplates.filter(s => s.riskLevel === 'Very High').length}</p>
                <p className="text-sm text-gray-400">High Risk</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Modals */}
      <Dialog open={showNewTradeModal} onOpenChange={setShowNewTradeModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Trade</DialogTitle>
            <DialogDescription>
              <NewTradeForm onSuccess={() => setShowNewTradeModal(false)} />
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>

      <Dialog open={showAlertsModal} onOpenChange={setShowAlertsModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Alert Rule</DialogTitle>
            <DialogDescription>
              <NewAlertForm onSuccess={() => setShowAlertsModal(false)} />
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>

      {/* Strategy Builder Modal */}
      <Dialog open={showStrategyBuilder} onOpenChange={setShowStrategyBuilder}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              {selectedStrategy?.icon && <span className="text-2xl">{selectedStrategy.icon}</span>}
              <span>Strategy Builder - {selectedStrategy?.name}</span>
            </DialogTitle>
            <DialogDescription>
              Configure and deploy your {selectedStrategy?.name} strategy
            </DialogDescription>
          </DialogHeader>
          {selectedStrategy && (
            <StrategyBuilderContent 
              strategy={selectedStrategy} 
              onClose={() => setShowStrategyBuilder(false)}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function NewTradeForm({ onSuccess }: { onSuccess?: () => void }) {
  const [formData, setFormData] = useState<PositionFormData>({
    symbol: '',
    optionType: 'call',
    action: 'buy',
    strike: '',
    expiration: '',
    premium: '',
    quantity: ''
  });

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    toast({
      title: "Trade executed successfully!",
      description: `${formData.action.toUpperCase()} ${formData.quantity} ${formData.symbol} ${formData.strike} ${formData.optionType.toUpperCase()} @ $${formData.premium}`,
    });
    
    onSuccess?.();
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 pt-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="symbol">Symbol</Label>
          <Input 
            id="symbol"
            name="symbol"
            value={formData.symbol}
            onChange={handleChange}
            placeholder="AAPL"
            required
          />
        </div>
        <div>
          <Label htmlFor="optionType">Option Type</Label>
          <Select name="optionType" value={formData.optionType} onValueChange={(value) => setFormData(prev => ({ ...prev, optionType: value as 'call' | 'put' }))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="call">Call</SelectItem>
              <SelectItem value="put">Put</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="action">Action</Label>
          <Select name="action" value={formData.action} onValueChange={(value) => setFormData(prev => ({ ...prev, action: value as 'buy' | 'sell' }))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="buy">Buy to Open</SelectItem>
              <SelectItem value="sell">Sell to Open</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="strike">Strike Price</Label>
          <Input 
            id="strike"
            name="strike"
            type="number"
            value={formData.strike}
            onChange={handleChange}
            placeholder="175.00"
            step="0.01"
            required
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="expiration">Expiration</Label>
          <Input 
            id="expiration"
            name="expiration"
            type="date"
            value={formData.expiration}
            onChange={handleChange}
            required
          />
        </div>
        <div>
          <Label htmlFor="premium">Premium</Label>
          <Input 
            id="premium"
            name="premium"
            type="number"
            value={formData.premium}
            onChange={handleChange}
            placeholder="8.45"
            step="0.01"
            required
          />
        </div>
      </div>

      <div>
        <Label htmlFor="quantity">Quantity</Label>
        <Input 
          id="quantity"
          name="quantity"
          type="number"
          value={formData.quantity}
          onChange={handleChange}
          placeholder="1"
          min="1"
          required
        />
      </div>

      <Button type="submit" className="w-full btn-liquid text-white">
        Execute Trade
      </Button>
    </form>
  );
}

function StrategyBuilderContent({ strategy, onClose }: { strategy: any; onClose: () => void }) {
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploymentProgress, setDeploymentProgress] = useState(0);

  const handleDeploy = async () => {
    setIsDeploying(true);
    setDeploymentProgress(0);

    // Simulate deployment process
    for (let i = 0; i <= 100; i += 10) {
      setDeploymentProgress(i);
      await new Promise(resolve => setTimeout(resolve, 200));
    }

    toast({
      title: "Strategy deployed successfully!",
      description: `${strategy.name} strategy is now active in your portfolio.`,
    });

    setIsDeploying(false);
    onClose();
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'Low': return 'text-green-400';
      case 'Medium': return 'text-yellow-400';
      case 'High': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Beginner': return 'bg-green-500/20 text-green-300';
      case 'Intermediate': return 'bg-yellow-500/20 text-yellow-300';
      case 'Advanced': return 'bg-red-500/20 text-red-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  return (
    <div className="space-y-6">
      {/* Strategy Overview */}
      <div className="neomorphic p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-4xl">{strategy.icon}</span>
            <div>
              <h2 className="text-2xl font-bold">{strategy.name}</h2>
              <p className="text-gray-400">{strategy.description}</p>
            </div>
          </div>
          <Badge className={getComplexityColor(strategy.complexity)}>
            {strategy.complexity}
          </Badge>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-400">Risk Level</p>
            <p className={`font-bold ${getRiskLevelColor(strategy.riskLevel)}`}>
              {strategy.riskLevel}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Profit Potential</p>
            <p className="font-bold text-blue-400">{strategy.profitPotential}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Market Outlook</p>
            <p className="font-bold text-purple-400">{strategy.marketOutlook}</p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Legs</p>
            <p className="font-bold text-cyan-400">{strategy.legs}</p>
          </div>
        </div>
      </div>

      {/* Strategy Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Risk Profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Max Risk:</span>
              <span className="font-semibold">{strategy.maxRisk}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Profit:</span>
              <span className="font-semibold">{strategy.maxProfit}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Breakeven:</span>
              <span className="font-semibold">{strategy.breakeven}</span>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Market Conditions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility:</span>
              <span className="font-semibold">{strategy.volatilityOutlook}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Time Decay:</span>
              <span className={`font-semibold ${strategy.timeDecay === 'Positive' ? 'text-green-400' : 'text-red-400'}`}>
                {strategy.timeDecay}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Margin Required:</span>
              <span className={`font-semibold ${strategy.marginRequired ? 'text-red-400' : 'text-green-400'}`}>
                {strategy.marginRequired ? 'Yes' : 'No'}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Deployment Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">Deploy Strategy</CardTitle>
          <CardDescription>Configure and activate this strategy in your portfolio</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {isDeploying ? (
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-400"></div>
                <span>Deploying strategy...</span>
              </div>
              <Progress value={deploymentProgress} className="w-full" />
              <p className="text-sm text-gray-400">
                {deploymentProgress < 30 ? 'Analyzing market conditions...' :
                 deploymentProgress < 60 ? 'Calculating optimal parameters...' :
                 deploymentProgress < 90 ? 'Setting up positions...' :
                 'Finalizing deployment...'}
              </p>
            </div>
          ) : (
            <div className="flex space-x-4">
              <Button 
                onClick={handleDeploy}
                className="btn-liquid text-white flex-1"
              >
                <PlayCircle className="h-4 w-4 mr-2" />
                Deploy Strategy
              </Button>
              <Button 
                onClick={onClose}
                variant="outline"
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function NewAlertForm({ onSuccess }: { onSuccess?: () => void }): JSX.Element {
  const [formData, setFormData] = useState({
    name: '',
    symbol: '',
    condition: 'price_above',
    value: '',
    email: true,
    push: false,
    sms: false,
    category: 'Price Movement',
    metric: 'stock_price_usd'
  });

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  }

  function handleCheckboxChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, checked } = e.target;
    setFormData(prev => ({ ...prev, [name]: checked }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/alerts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          symbol: formData.symbol,
          condition: formData.condition,
          value: parseFloat(formData.value),
          email_enabled: formData.email,
          push_enabled: formData.push,
          sms_enabled: formData.sms,
          category: formData.category,
          metric: formData.metric
        }),
      });

      if (response.ok) {
        toast({
          title: "Alert created successfully!",
          description: `Alert for ${formData.symbol} has been set up.`,
        });
        onSuccess?.();
      } else {
        throw new Error('Failed to create alert');
      }
    } catch (error) {
      toast({
        title: "Error creating alert",
        description: "Please try again later.",
      });
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 pt-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="name">Alert Name</Label>
          <Input 
            id="name"
            name="name"
            value={formData.name}
            onChange={handleInputChange}
            placeholder="AAPL Price Alert"
            required
          />
        </div>
        <div>
          <Label htmlFor="symbol">Symbol</Label>
          <Input 
            id="symbol"
            name="symbol"
            value={formData.symbol}
            onChange={handleInputChange}
            placeholder="AAPL"
            required
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="condition">Condition</Label>
          <Select name="condition" value={formData.condition} onValueChange={(value) => setFormData(prev => ({ ...prev, condition: value }))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="price_above">Price Above</SelectItem>
              <SelectItem value="price_below">Price Below</SelectItem>
              <SelectItem value="volume_above">Volume Above</SelectItem>
              <SelectItem value="iv_above">IV Above</SelectItem>
              <SelectItem value="iv_below">IV Below</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="value">Value</Label>
          <Input 
            id="value"
            name="value"
            type="number"
            value={formData.value}
            onChange={handleInputChange}
            placeholder="175.00"
            step="0.01"
            required
          />
        </div>
      </div>

      <div>
        <Label>Notification Methods</Label>
        <div className="flex space-x-6 mt-2">
          <label className="flex items-center space-x-2">
            <input 
              type="checkbox" 
              name="email"
              checked={formData.email}
              onChange={handleCheckboxChange}
              className="rounded"
            />
            <span>Email</span>
          </label>
          <label className="flex items-center space-x-2">
            <input 
              type="checkbox" 
              name="push"
              checked={formData.push}
              onChange={handleCheckboxChange}
              className="rounded"
            />
            <span>Push</span>
          </label>
          <label className="flex items-center space-x-2">
            <input 
              type="checkbox" 
              name="sms"
              checked={formData.sms}
              onChange={handleCheckboxChange}
              className="rounded"
            />
            <span>SMS</span>
          </label>
        </div>
      </div>

      <Button type="submit" className="w-full btn-liquid text-white">
        Create Alert
      </Button>
    </form>
  );
}