'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatCurrency } from '@/lib/utils';
import { 
  BarChart, 
  LineChart, 
  PieChart, 
  TrendingUp, 
  Download, 
  Settings, 
  Activity,
  Zap,
  Target,
  Globe,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  Eye,
  Filter,
  RefreshCw
} from 'lucide-react';

interface ChartData {
  id: string;
  name: string;
  type: 'line' | 'bar' | 'pie' | 'candlestick';
  data: any[];
  timeframe: string;
  symbol?: string;
  description: string;
  icon: any;
  gradient: string;
}

export function VisualizationContent() {
  const [selectedChart, setSelectedChart] = useState<string>('portfolio-performance');
  const [isLoading, setIsLoading] = useState(false);
  
  const [charts] = useState<ChartData[]>([
    {
      id: 'portfolio-performance',
      name: 'Portfolio Performance',
      type: 'line',
      data: [],
      timeframe: '1Y',
      description: 'Track your portfolio growth over time',
      icon: TrendingUp,
      gradient: 'from-blue-600 via-purple-600 to-blue-800'
    },
    {
      id: 'asset-allocation',
      name: 'Asset Allocation',
      type: 'pie',
      data: [],
      timeframe: 'Current',
      description: 'Visualize your investment distribution',
      icon: Target,
      gradient: 'from-emerald-500 via-teal-600 to-cyan-600'
    },
    {
      id: 'price-chart',
      name: 'Market Analysis',
      type: 'candlestick',
      data: [],
      timeframe: '1M',
      symbol: 'AAPL',
      description: 'Real-time market movements and trends',
      icon: Activity,
      gradient: 'from-amber-500 via-orange-600 to-red-600'
    },
    {
      id: 'sector-performance',
      name: 'Sector Insights',
      type: 'bar',
      data: [],
      timeframe: '1M',
      description: 'Compare sector performance metrics',
      icon: Globe,
      gradient: 'from-violet-600 via-purple-600 to-indigo-800'
    }
  ]);

  // Enhanced mock data
  const portfolioData = [
    { date: '2024-01-01', value: 100000, change: 0 },
    { date: '2024-01-05', value: 102500, change: 2.5 },
    { date: '2024-01-10', value: 98750, change: -1.25 },
    { date: '2024-01-15', value: 105200, change: 5.2 },
    { date: '2024-01-20', value: 108900, change: 8.9 },
  ];

  const allocationData = [
    { name: 'Technology', value: 45.2, amount: 49232, color: '#3b82f6', trend: '+12.3%' },
    { name: 'Healthcare', value: 23.1, amount: 25152, color: '#10b981', trend: '+8.7%' },
    { name: 'Financial', value: 18.7, amount: 20360, color: '#f59e0b', trend: '-2.1%' },
    { name: 'Energy', value: 8.9, amount: 9692, color: '#ef4444', trend: '+15.6%' },
    { name: 'Consumer', value: 4.1, amount: 4464, color: '#8b5cf6', trend: '+5.2%' },
  ];

  const sectorData = [
    { sector: 'Technology', performance: 12.5, volume: '2.4B', momentum: 'Strong' },
    { sector: 'Healthcare', performance: 8.3, volume: '1.8B', momentum: 'Moderate' },
    { sector: 'Financial', performance: -2.1, volume: '3.1B', momentum: 'Weak' },
    { sector: 'Energy', performance: 15.7, volume: '1.2B', momentum: 'Very Strong' },
    { sector: 'Consumer', performance: 5.2, volume: '2.0B', momentum: 'Moderate' },
  ];

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1500);
  };

  const renderChart = () => {
    const currentChart = charts.find(c => c.id === selectedChart);
    
    switch (selectedChart) {
      case 'portfolio-performance':
        return (
          <div className="relative">
            {/* Background with animated gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5 rounded-xl" />
            
            {/* Content */}
            <div className="relative p-8">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Chart Area */}
                <div className="lg:col-span-2">
                  <div className="h-72 bg-gradient-to-br from-gray-900/50 to-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 relative overflow-hidden">
                    {/* Animated background pattern */}
                    <div className="absolute inset-0 opacity-5">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-600 transform rotate-12 scale-150" />
                    </div>
                    
                    <div className="relative z-10 flex flex-col h-full">
                      <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-blue-500/20 rounded-lg">
                            <TrendingUp className="h-6 w-6 text-blue-400" />
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-white">Portfolio Timeline</h3>
                            <p className="text-gray-400 text-sm">YTD Performance</p>
                          </div>
                        </div>
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                          +8.9% YTD
                        </Badge>
                      </div>
                      
                      {/* Simulated line chart with points */}
                      <div className="flex-1 flex items-end space-x-4">
                        {portfolioData.map((point, index) => (
                          <div key={index} className="flex-1 flex flex-col items-center group">
                            <div className="relative mb-2">
                              <div 
                                className="bg-gradient-to-t from-blue-500 to-purple-500 rounded-t transition-all duration-300 group-hover:scale-105"
                                style={{ 
                                  height: `${(point.value - 95000) / 1000}px`,
                                  minHeight: '20px',
                                  width: '24px'
                                }}
                              />
                              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white whitespace-nowrap">
                                  {formatCurrency(point.value)}
                                </div>
                              </div>
                            </div>
                            <span className="text-xs text-gray-400 transform rotate-45 origin-bottom-left">
                              {point.date.split('-').slice(1).join('/')}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Stats Panel */}
                <div className="space-y-4">
                  <div className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border border-green-500/20 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400 text-sm">Total Value</span>
                      <ArrowUpRight className="h-4 w-4 text-green-400" />
                    </div>
                    <div className="text-2xl font-bold text-white">$108,900</div>
                    <div className="text-green-400 text-sm">+$8,900 (+8.9%)</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-500/10 to-cyan-600/10 border border-blue-500/20 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400 text-sm">Day's Change</span>
                      <Sparkles className="h-4 w-4 text-blue-400" />
                    </div>
                    <div className="text-xl font-bold text-white">+$1,250</div>
                    <div className="text-blue-400 text-sm">+1.15% today</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-purple-500/10 to-violet-600/10 border border-purple-500/20 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400 text-sm">Best Performer</span>
                      <Zap className="h-4 w-4 text-purple-400" />
                    </div>
                    <div className="text-lg font-bold text-white">NVDA</div>
                    <div className="text-purple-400 text-sm">+15.7% this month</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'asset-allocation':
        return (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 via-teal-500/5 to-cyan-500/5 rounded-xl" />
            
            <div className="relative p-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Pie Chart Visualization */}
                <div className="flex items-center justify-center">
                  <div className="relative">
                    {/* Outer ring */}
                    <div className="w-64 h-64 relative">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                        {allocationData.map((item, index) => {
                          const offset = allocationData.slice(0, index).reduce((sum, d) => sum + d.value, 0);
                          const circumference = 2 * Math.PI * 40;
                          const strokeDasharray = `${(item.value / 100) * circumference} ${circumference}`;
                          const strokeDashoffset = `-${(offset / 100) * circumference}`;
                          
                          return (
                            <circle
                              key={index}
                              cx="50"
                              cy="50"
                              r="40"
                              fill="none"
                              stroke={item.color}
                              strokeWidth="8"
                              strokeDasharray={strokeDasharray}
                              strokeDashoffset={strokeDashoffset}
                              className="transition-all duration-300 hover:stroke-width-10"
                              style={{ filter: 'drop-shadow(0 0 6px rgba(59, 130, 246, 0.3))' }}
                            />
                          );
                        })}
                      </svg>
                      
                      {/* Center content */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">100%</div>
                          <div className="text-gray-400 text-sm">Allocated</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Legend and Details */}
                <div className="space-y-4">
                  <div className="flex items-center space-x-2 mb-6">
                    <Target className="h-6 w-6 text-emerald-400" />
                    <h3 className="text-xl font-bold text-white">Asset Breakdown</h3>
                  </div>
                  
                  {allocationData.map((item, index) => (
                    <div key={index} className="group">
                      <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4 transition-all duration-300 hover:bg-gray-700/50 hover:border-gray-600/50 hover:transform hover:scale-[1.02]">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-3">
                            <div 
                              className="w-4 h-4 rounded-full shadow-lg"
                              style={{ 
                                backgroundColor: item.color,
                                boxShadow: `0 0 12px ${item.color}40`
                              }}
                            />
                            <span className="font-medium text-white">{item.name}</span>
                          </div>
                          <Badge 
                            className={`${
                              item.trend.startsWith('+') 
                                ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                                : 'bg-red-500/20 text-red-400 border-red-500/30'
                            }`}
                          >
                            {item.trend}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="text-2xl font-bold text-white">{item.value}%</div>
                          <div className="text-right">
                            <div className="text-gray-300 font-medium">{formatCurrency(item.amount)}</div>
                            <div className="text-gray-400 text-sm">Market Value</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'price-chart':
        return (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 via-orange-500/5 to-red-500/5 rounded-xl" />
            
            <div className="relative p-8">
              <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                {/* Main Price Display */}
                <div className="xl:col-span-3 space-y-6">
                  <div className="bg-gradient-to-br from-gray-900/80 to-gray-800/80 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center space-x-4">
                        <div className="p-3 bg-amber-500/20 rounded-xl">
                          <Activity className="h-8 w-8 text-amber-400" />
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold text-white">AAPL</h3>
                          <p className="text-gray-400">Apple Inc.</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold text-white">$175.23</div>
                        <div className="flex items-center text-green-400">
                          <ArrowUpRight className="h-4 w-4 mr-1" />
                          <span>+$2.34 (+1.35%)</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Simulated candlestick chart */}
                    <div className="h-48 flex items-end justify-between space-x-2">
                      {Array.from({ length: 20 }, (_, i) => {
                        const height = 40 + Math.random() * 80;
                        const isGreen = Math.random() > 0.5;
                        return (
                          <div key={i} className="flex-1 flex flex-col items-center group">
                            <div 
                              className={`w-full transition-all duration-300 group-hover:scale-110 ${
                                isGreen 
                                  ? 'bg-gradient-to-t from-green-500 to-green-400' 
                                  : 'bg-gradient-to-t from-red-500 to-red-400'
                              }`}
                              style={{ height: `${height}px` }}
                            />
                            {/* Volume bars at bottom */}
                            <div 
                              className="w-full bg-gray-600 mt-1"
                              style={{ height: `${5 + Math.random() * 15}px` }}
                            />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Market Indicators */}
                <div className="space-y-4">
                  <div className="bg-gradient-to-br from-green-500/10 to-emerald-600/10 border border-green-500/20 rounded-xl p-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">24h Volume</div>
                      <div className="text-lg font-bold text-white">52.3M</div>
                      <div className="text-green-400 text-xs">+12.5%</div>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-500/10 to-cyan-600/10 border border-blue-500/20 rounded-xl p-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">Market Cap</div>
                      <div className="text-lg font-bold text-white">$2.7T</div>
                      <div className="text-blue-400 text-xs">Rank #1</div>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-purple-500/10 to-violet-600/10 border border-purple-500/20 rounded-xl p-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">P/E Ratio</div>
                      <div className="text-lg font-bold text-white">28.5</div>
                      <div className="text-purple-400 text-xs">Fair Value</div>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-orange-500/10 to-red-600/10 border border-orange-500/20 rounded-xl p-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">52W Range</div>
                      <div className="text-sm font-bold text-white">$164-$199</div>
                      <div className="text-orange-400 text-xs">Mid Range</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'sector-performance':
        return (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 via-purple-500/5 to-indigo-500/5 rounded-xl" />
            
            <div className="relative p-8">
              <div className="flex items-center space-x-3 mb-8">
                <div className="p-3 bg-violet-500/20 rounded-xl">
                  <Globe className="h-6 w-6 text-violet-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white">Sector Analysis</h3>
                  <p className="text-gray-400">Market performance by sector</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {sectorData.map((item, index) => {
                  const isPositive = item.performance >= 0;
                  const momentum = item.momentum;
                  const momentumColor = 
                    momentum === 'Very Strong' ? 'text-emerald-400 bg-emerald-500/20' :
                    momentum === 'Strong' ? 'text-green-400 bg-green-500/20' :
                    momentum === 'Moderate' ? 'text-blue-400 bg-blue-500/20' :
                    'text-red-400 bg-red-500/20';
                  
                  return (
                    <div key={index} className="group">
                      <div className="bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 transition-all duration-300 hover:border-violet-500/30 hover:transform hover:scale-[1.02] hover:shadow-2xl">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h4 className="text-xl font-bold text-white">{item.sector}</h4>
                            <p className="text-gray-400 text-sm">Volume: {item.volume}</p>
                          </div>
                          <Badge className={momentumColor}>
                            {momentum}
                          </Badge>
                        </div>
                        
                        <div className="space-y-4">
                          {/* Performance bar */}
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-gray-300 text-sm">Performance</span>
                              <span className={`font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{item.performance}%
                              </span>
                            </div>
                            <div className="relative">
                              <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                                <div 
                                  className={`h-full transition-all duration-1000 ease-out ${
                                    isPositive 
                                      ? 'bg-gradient-to-r from-green-500 to-emerald-400' 
                                      : 'bg-gradient-to-r from-red-500 to-red-400'
                                  }`}
                                  style={{ 
                                    width: `${Math.min(Math.abs(item.performance) * 4, 100)}%`,
                                    boxShadow: isPositive 
                                      ? '0 0 12px rgba(34, 197, 94, 0.5)' 
                                      : '0 0 12px rgba(239, 68, 68, 0.5)'
                                  }}
                                />
                              </div>
                              {/* Glow effect */}
                              <div 
                                className={`absolute top-0 h-full rounded-full blur-sm ${
                                  isPositive 
                                    ? 'bg-gradient-to-r from-green-400/30 to-emerald-300/30' 
                                    : 'bg-gradient-to-r from-red-400/30 to-red-300/30'
                                }`}
                                style={{ width: `${Math.min(Math.abs(item.performance) * 4, 100)}%` }}
                              />
                            </div>
                          </div>
                          
                          {/* Additional metrics */}
                          <div className="grid grid-cols-2 gap-4 pt-2">
                            <div className="text-center">
                              <div className="text-gray-400 text-xs">RSI</div>
                              <div className="text-white font-semibold">{(Math.random() * 40 + 30).toFixed(1)}</div>
                            </div>
                            <div className="text-center">
                              <div className="text-gray-400 text-xs">Beta</div>
                              <div className="text-white font-semibold">{(Math.random() * 0.8 + 0.6).toFixed(2)}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="h-96 bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-xl flex items-center justify-center">
            <div className="text-center">
              <Eye className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-400 text-lg">Select a visualization to explore</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="space-y-8">
      {/* Enhanced Header */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-pink-600/10 rounded-2xl blur-xl" />
        <Card className="relative bg-gray-900/80 backdrop-blur-sm border-gray-800/50">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
                  <Sparkles className="h-8 w-8 text-white" />
                </div>
                <div>
                  <CardTitle className="text-2xl text-white">Advanced Analytics</CardTitle>
                  <p className="text-gray-400">Interactive market visualizations and insights</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                  onClick={handleRefresh}
                  disabled={isLoading}
                >
                  <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
                <Button variant="outline" size="sm" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  <Filter className="h-4 w-4 mr-2" />
                  Filters
                </Button>
              </div>
            </div>
          </CardHeader>
          
          <CardContent>
            {/* Stylish Chart Selection */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {charts.map((chart) => {
                const IconComponent = chart.icon;
                const isSelected = selectedChart === chart.id;
                
                return (
                  <button
                    key={chart.id}
                    onClick={() => setSelectedChart(chart.id)}
                    className={`group relative p-6 rounded-xl border transition-all duration-300 text-left ${
                      isSelected 
                        ? 'border-blue-500/50 bg-gradient-to-br from-blue-500/10 to-purple-500/10 shadow-lg shadow-blue-500/25' 
                        : 'border-gray-700 bg-gray-800/50 hover:border-gray-600 hover:bg-gray-700/50'
                    }`}
                  >
                    {/* Background gradient effect */}
                    {isSelected && (
                      <div className={`absolute inset-0 bg-gradient-to-br ${chart.gradient} opacity-5 rounded-xl`} />
                    )}
                    
                    <div className="relative">
                      <div className={`p-3 rounded-lg mb-3 transition-all duration-300 ${
                        isSelected 
                          ? `bg-gradient-to-br ${chart.gradient} shadow-lg` 
                          : 'bg-gray-700 group-hover:bg-gray-600'
                      }`}>
                        <IconComponent className="h-6 w-6 text-white" />
                      </div>
                      
                      <h3 className={`font-semibold mb-1 transition-colors ${
                        isSelected ? 'text-white' : 'text-gray-300 group-hover:text-white'
                      }`}>
                        {chart.name}
                      </h3>
                      
                      <p className="text-gray-400 text-sm">{chart.description}</p>
                      
                      <Badge 
                        variant="outline" 
                        className={`mt-3 ${
                          isSelected 
                            ? 'border-blue-400/50 text-blue-400' 
                            : 'border-gray-600 text-gray-400'
                        }`}
                      >
                        {chart.timeframe}
                      </Badge>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3">
              <Button variant="outline" size="sm" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                <Download className="h-4 w-4 mr-2" />
                Export Data
              </Button>
              <Button variant="outline" size="sm" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                <Settings className="h-4 w-4 mr-2" />
                Customize View
              </Button>
              <Button variant="outline" size="sm" className="border-gray-600 text-gray-300 hover:bg-gray-700">
                <Activity className="h-4 w-4 mr-2" />
                Real-time Mode
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart Display with enhanced styling */}
      <Card className="bg-gray-900/80 backdrop-blur-sm border-gray-800/50 overflow-hidden">
        <CardHeader className="border-b border-gray-800/50">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center space-x-3">
              {(() => {
                const currentChart = charts.find(c => c.id === selectedChart);
                const IconComponent = currentChart?.icon || Activity;
                return (
                  <>
                    <IconComponent className="h-6 w-6 text-blue-400" />
                    <span>{currentChart?.name || 'Chart'}</span>
                  </>
                );
              })()}
            </CardTitle>
            
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-green-400 text-sm">Live</span>
              </div>
              <Badge variant="outline" className="border-gray-600 text-gray-400">
                {charts.find(c => c.id === selectedChart)?.timeframe || 'N/A'}
              </Badge>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="p-0">
          {renderChart()}
        </CardContent>
      </Card>
    </div>
  );
} 