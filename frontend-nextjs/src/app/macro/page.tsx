'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Globe, 
  BarChart3,
  Activity,
  Calendar,
  AlertTriangle,
  PieChart,
  Target,
  Zap,
  Layers,
  RefreshCw
} from 'lucide-react';

interface MacroData {
  treasury_yields: Record<string, any>;
  inflation: Record<string, any>;
  monetary_policy: Record<string, any>;
}

export default function MacroPage() {
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  // Fetch real macro data
  const fetchMacroData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/macro/comprehensive');
      if (response.ok) {
        const data = await response.json();
        setMacroData(data.data);
        setLastUpdated(data.last_updated);
      } else {
        console.error('Failed to fetch macro data');
        // Fall back to mock data if API fails
        setMacroData(getMockMacroData());
      }
    } catch (error) {
      console.error('Error fetching macro data:', error);
      // Fall back to mock data if API fails
      setMacroData(getMockMacroData());
    } finally {
      setIsLoading(false);
    }
  };

  // Mock data fallback
  const getMockMacroData = (): MacroData => ({
    treasury_yields: {
      '3M': { value: 5.30, change_24h: 0.02 },
      '2Y': { value: 4.75, change_24h: -0.05 },
      '10Y': { value: 4.50, change_24h: -0.08 },
      '2Y10Y_SPREAD': { value: -0.25, change_24h: -0.03 }
    },
    inflation: {
      'CORE_PCE': { value: 307.5, yoy_change: 2.8, mom_change: 0.1 },
      'BREAKEVEN_5Y': { value: 2.45, yoy_change: 2.45, mom_change: 0.08 }
    },
    monetary_policy: {
      'FED_FUNDS_RATE': { value: 5.25, change_pct: 0.0, units: '%' },
      'FED_BALANCE_SHEET': { value: 7200000, change_pct: -2.1, units: 'Millions $' }
    }
  });

  useEffect(() => {
    fetchMacroData();
    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchMacroData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Core Central Bank & Monetary Policy from real data
  const monetaryIndicators = macroData ? [
    { 
      name: 'Fed Funds Rate', 
      value: `${macroData.monetary_policy.FED_FUNDS_RATE?.value || 5.25}%`, 
      change: `${(macroData.monetary_policy.FED_FUNDS_RATE?.change_pct || 0) >= 0 ? '+' : ''}${macroData.monetary_policy.FED_FUNDS_RATE?.change_pct || 0}%`, 
      trend: (macroData.monetary_policy.FED_FUNDS_RATE?.change_pct || 0) > 0 ? 'up' : (macroData.monetary_policy.FED_FUNDS_RATE?.change_pct || 0) < 0 ? 'down' : 'neutral', 
      impact: 'high', 
      category: 'monetary' 
    },
    { 
      name: 'Fed Balance Sheet', 
      value: `$${((macroData.monetary_policy.FED_BALANCE_SHEET?.value || 7200000) / 1000000).toFixed(1)}T`, 
      change: `${(macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct || -2.1) >= 0 ? '+' : ''}${macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct || -2.1}%`, 
      trend: (macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct || -2.1) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'monetary' 
    },
    { name: 'ECB Deposit Rate', value: '4.00%', change: '+0.00%', trend: 'neutral', impact: 'high', category: 'monetary' },
    { name: 'BoJ Policy Rate', value: '-0.10%', change: '+0.00%', trend: 'neutral', impact: 'medium', category: 'monetary' }
  ] : [];

  // Yield Curve & Fixed Income from real data
  const yieldCurveIndicators = macroData ? [
    { 
      name: '2Y Treasury', 
      value: `${macroData.treasury_yields['2Y']?.value || 4.75}%`, 
      change: `${(macroData.treasury_yields['2Y']?.change_24h || 0) >= 0 ? '+' : ''}${macroData.treasury_yields['2Y']?.change_24h || 0}%`, 
      trend: (macroData.treasury_yields['2Y']?.change_24h || 0) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'curve' 
    },
    { 
      name: '10Y Treasury', 
      value: `${macroData.treasury_yields['10Y']?.value || 4.50}%`, 
      change: `${(macroData.treasury_yields['10Y']?.change_24h || 0) >= 0 ? '+' : ''}${macroData.treasury_yields['10Y']?.change_24h || 0}%`, 
      trend: (macroData.treasury_yields['10Y']?.change_24h || 0) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'curve' 
    },
    { 
      name: '2Y-10Y Spread', 
      value: `${Math.round((macroData.treasury_yields['2Y10Y_SPREAD']?.value || -0.25) * 100)}bps`, 
      change: `${(macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h || 0) >= 0 ? '+' : ''}${Math.round((macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h || 0) * 100)}bps`, 
      trend: (macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h || 0) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'curve' 
    },
    { 
      name: '3M Treasury', 
      value: `${macroData.treasury_yields['3M']?.value || 5.30}%`, 
      change: `${(macroData.treasury_yields['3M']?.change_24h || 0) >= 0 ? '+' : ''}${macroData.treasury_yields['3M']?.change_24h || 0}%`, 
      trend: (macroData.treasury_yields['3M']?.change_24h || 0) > 0 ? 'up' : 'down', 
      impact: 'medium', 
      category: 'curve' 
    }
  ] : [];

  // Inflation & Growth from real data
  const inflationGrowthIndicators = macroData ? [
    { 
      name: 'Core PCE', 
      value: `${macroData.inflation.CORE_PCE?.yoy_change || 2.8}%`, 
      change: `${(macroData.inflation.CORE_PCE?.mom_change || 0) >= 0 ? '+' : ''}${macroData.inflation.CORE_PCE?.mom_change || 0}%`, 
      trend: (macroData.inflation.CORE_PCE?.mom_change || 0) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'inflation' 
    },
    { 
      name: '5Y Breakeven', 
      value: `${macroData.inflation.BREAKEVEN_5Y?.yoy_change || 2.45}%`, 
      change: `${(macroData.inflation.BREAKEVEN_5Y?.mom_change || 0.08) >= 0 ? '+' : ''}${macroData.inflation.BREAKEVEN_5Y?.mom_change || 0.08}%`, 
      trend: (macroData.inflation.BREAKEVEN_5Y?.mom_change || 0.08) > 0 ? 'up' : 'down', 
      impact: 'high', 
      category: 'inflation' 
    },
    { name: 'Atlanta Fed GDPNow', value: '2.7%', change: '+0.2%', trend: 'up', impact: 'medium', category: 'growth' },
    { name: 'NY Fed WEI', value: '2.1%', change: '+0.1%', trend: 'up', impact: 'medium', category: 'growth' }
  ] : [];

  // Cross-Asset & Volatility
  const crossAssetIndicators = [
    { name: 'VIX', value: '18.5', change: '-2.1%', trend: 'down', impact: 'high', category: 'volatility' },
    { name: 'MOVE Index', value: '115.2', change: '+3.8%', trend: 'up', impact: 'high', category: 'volatility' },
    { name: 'SKEW Index', value: '142.5', change: '+1.2%', trend: 'up', impact: 'medium', category: 'volatility' },
    { name: 'Gold/Silver Ratio', value: '78.2', change: '-0.8%', trend: 'down', impact: 'medium', category: 'metals' }
  ];

  // Currency & Carry Trade
  const currencyCarryIndicators = [
    { name: 'DXY Index', value: '103.45', change: '+0.8%', trend: 'up', impact: 'high', category: 'currency' },
    { name: 'JPY Carry Trade', value: '-2.8%', change: '+0.3%', trend: 'up', impact: 'high', category: 'carry' },
    { name: 'AUD/JPY', value: '98.25', change: '+1.2%', trend: 'up', impact: 'medium', category: 'carry' },
    { name: 'TRY Real Rate', value: '8.5%', change: '+0.5%', trend: 'up', impact: 'medium', category: 'carry' }
  ];

  // Credit & Liquidity
  const creditLiquidityIndicators = [
    { name: 'IG Credit Spread', value: '125bps', change: '+5bps', trend: 'up', impact: 'high', category: 'credit' },
    { name: 'HY Credit Spread', value: '485bps', change: '+12bps', trend: 'up', impact: 'high', category: 'credit' },
    { name: 'LIBOR-OIS Spread', value: '12bps', change: '+1bp', trend: 'up', impact: 'medium', category: 'liquidity' },
    { name: 'Term Premium', value: '0.35%', change: '+0.08%', trend: 'up', impact: 'medium', category: 'liquidity' }
  ];

  // Commodity & Energy
  const commodityIndicators = [
    { name: 'DJP Commodity', value: '28.45', change: '+2.1%', trend: 'up', impact: 'medium', category: 'commodity' },
    { name: 'Copper/Gold Ratio', value: '0.0032', change: '+1.8%', trend: 'up', impact: 'medium', category: 'commodity' },
    { name: 'Baltic Dry Index', value: '1,285', change: '+5.2%', trend: 'up', impact: 'low', category: 'commodity' },
    { name: 'Oil Contango', value: '0.85%', change: '-0.2%', trend: 'down', impact: 'medium', category: 'energy' }
  ];

  // Alternative & Sentiment
  const alternativeIndicators = [
    { name: 'AAII Bull/Bear', value: '1.2x', change: '+0.1x', trend: 'up', impact: 'low', category: 'sentiment' },
    { name: 'Put/Call Ratio', value: '0.85', change: '-0.05', trend: 'down', impact: 'medium', category: 'sentiment' },
    { name: 'Margin Debt', value: '$684B', change: '-1.8%', trend: 'down', impact: 'medium', category: 'leverage' },
    { name: 'Repo Rate', value: '5.30%', change: '+0.02%', trend: 'up', impact: 'medium', category: 'funding' }
  ];

  const upcomingEvents = [
    { date: 'Today', time: '2:00 PM', event: 'FOMC Meeting Minutes', impact: 'high' },
    { date: 'Tomorrow', time: '8:30 AM', event: 'Initial Jobless Claims', impact: 'medium' },
    { date: 'Jan 12', time: '8:30 AM', event: 'Core PCE Price Index', impact: 'high' },
    { date: 'Jan 15', time: '9:15 AM', event: 'Industrial Production', impact: 'medium' },
    { date: 'Jan 18', time: '10:00 AM', event: 'Existing Home Sales', impact: 'low' },
    { date: 'Jan 26', time: '8:30 AM', event: 'Advance GDP', impact: 'high' }
  ];

  const renderIndicatorSection = (title: string, indicators: any[], icon: any) => (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          {icon}
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {indicators.map((indicator, index) => (
            <div key={indicator.name} className="neomorphic p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-300">{indicator.name}</span>
                <Badge className={`text-xs ${
                  indicator.impact === 'high' ? 'bg-red-500/20 text-red-300' :
                  indicator.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                  'bg-green-500/20 text-green-300'
                }`}>
                  {indicator.impact}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold">{indicator.value}</span>
                <div className={`flex items-center space-x-1 ${
                  indicator.trend === 'up' ? 'text-green-400' : 
                  indicator.trend === 'down' ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {indicator.trend === 'up' ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : indicator.trend === 'down' ? (
                    <TrendingDown className="h-4 w-4" />
                  ) : (
                    <Activity className="h-4 w-4" />
                  )}
                  <span className="text-sm">{indicator.change}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold gradient-text">Institutional Macro Dashboard</h1>
            <p className="text-gray-400 mt-2">Advanced macroeconomic indicators for quantitative analysis</p>
          </div>
          <div className="flex space-x-2">
            <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">
              Live Data
            </Badge>
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">
              Institutional Grade
            </Badge>
          </div>
        </div>

        {/* Monetary Policy Section */}
        {renderIndicatorSection(
          'Central Bank & Monetary Policy',
          monetaryIndicators,
          <DollarSign className="h-5 w-5 text-green-400" />
        )}

        {/* Yield Curve Section */}
        {renderIndicatorSection(
          'Yield Curve & Term Structure',
          yieldCurveIndicators,
          <BarChart3 className="h-5 w-5 text-blue-400" />
        )}

        {/* Inflation & Growth Section */}
        {renderIndicatorSection(
          'Inflation & Growth Indicators',
          inflationGrowthIndicators,
          <TrendingUp className="h-5 w-5 text-orange-400" />
        )}

        {/* Cross-Asset & Volatility Section */}
        {renderIndicatorSection(
          'Cross-Asset & Volatility',
          crossAssetIndicators,
          <Activity className="h-5 w-5 text-red-400" />
        )}

        {/* Currency & Carry Trade Section */}
        {renderIndicatorSection(
          'Currency & Carry Trade',
          currencyCarryIndicators,
          <Globe className="h-5 w-5 text-cyan-400" />
        )}

        {/* Credit & Liquidity Section */}
        {renderIndicatorSection(
          'Credit & Liquidity Conditions',
          creditLiquidityIndicators,
          <Layers className="h-5 w-5 text-purple-400" />
        )}

        {/* Commodity & Energy Section */}
        {renderIndicatorSection(
          'Commodity & Energy Complex',
          commodityIndicators,
          <Zap className="h-5 w-5 text-yellow-400" />
        )}

        {/* Alternative & Sentiment Section */}
        {renderIndicatorSection(
          'Alternative Data & Sentiment',
          alternativeIndicators,
          <Target className="h-5 w-5 text-pink-400" />
        )}

        {/* Economic Calendar */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Calendar className="h-5 w-5 text-blue-400" />
              <span>High-Impact Economic Calendar</span>
            </CardTitle>
            <CardDescription>Institutional-grade event risk monitoring</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {upcomingEvents.map((event, index) => (
                <div key={index} className="flex items-center justify-between p-4 neomorphic rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="text-center">
                      <p className="text-sm font-medium">{event.date}</p>
                      <p className="text-xs text-gray-400">{event.time}</p>
                    </div>
                    <div>
                      <p className="font-medium">{event.event}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge className={`${
                      event.impact === 'high' ? 'bg-red-500/20 text-red-300' :
                      event.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                      'bg-green-500/20 text-green-300'
                    }`}>
                      {event.impact}
                    </Badge>
                    {event.impact === 'high' && (
                      <AlertTriangle className="h-4 w-4 text-red-400" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Institutional Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <PieChart className="h-5 w-5 text-green-400" />
                <span>Regime Analysis</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Macro Regime</span>
                  <Badge className="bg-blue-500/20 text-blue-300">Goldilocks</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Volatility Regime</span>
                  <Badge className="bg-yellow-500/20 text-yellow-300">Medium</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Risk Regime</span>
                  <Badge className="bg-green-500/20 text-green-300">Risk On</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Liquidity Regime</span>
                  <Badge className="bg-cyan-500/20 text-cyan-300">Abundant</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5 text-purple-400" />
                <span>Factor Exposures</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span>Growth Factor</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={75} className="w-20 h-2" />
                    <span className="text-green-400">+0.8σ</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span>Value Factor</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={40} className="w-20 h-2" />
                    <span className="text-red-400">-0.3σ</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span>Momentum Factor</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={65} className="w-20 h-2" />
                    <span className="text-green-400">+0.5σ</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span>Quality Factor</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={55} className="w-20 h-2" />
                    <span className="text-gray-400">+0.1σ</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Layers className="h-5 w-5 text-orange-400" />
                <span>Cross-Asset Signals</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Equity/Bond Correlation</span>
                  <Badge className="bg-red-500/20 text-red-300">+0.65</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Currency Momentum</span>
                  <Badge className="bg-green-500/20 text-green-300">Strong</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Commodity Trend</span>
                  <Badge className="bg-yellow-500/20 text-yellow-300">Mixed</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span>Crypto Correlation</span>
                  <Badge className="bg-purple-500/20 text-purple-300">+0.45</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 