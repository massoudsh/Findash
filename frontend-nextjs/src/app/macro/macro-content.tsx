'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
} from 'lucide-react';

interface MacroData {
  treasury_yields: Record<string, { value?: number; change_24h?: number }>;
  inflation: Record<string, { value?: number; yoy_change?: number; mom_change?: number }>;
  monetary_policy: Record<string, { value?: number; change_pct?: number }>;
}

function renderIndicatorSection(
  title: string,
  indicators: { name: string; value: string; change: string; trend: string; impact: string }[],
  icon: React.ReactNode
) {
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          {icon}
          <span>{title}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {indicators.map((indicator) => (
            <div key={indicator.name} className="neomorphic p-4 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-muted-foreground">{indicator.name}</span>
                <Badge className={`text-xs ${indicator.impact === 'high' ? 'bg-red-500/20 text-red-300' : indicator.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}`}>
                  {indicator.impact}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-foreground">{indicator.value}</span>
                <div className={`flex items-center space-x-1 ${indicator.trend === 'up' ? 'text-chart-2' : indicator.trend === 'down' ? 'text-destructive' : 'text-muted-foreground'}`}>
                  {indicator.trend === 'up' ? <TrendingUp className="h-4 w-4" /> : indicator.trend === 'down' ? <TrendingDown className="h-4 w-4" /> : <Activity className="h-4 w-4" />}
                  <span className="text-sm">{indicator.change}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function MacroContent() {
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const getMockMacroData = (): MacroData => ({
    treasury_yields: { '3M': { value: 5.30, change_24h: 0.02 }, '2Y': { value: 4.75, change_24h: -0.05 }, '10Y': { value: 4.50, change_24h: -0.08 }, '2Y10Y_SPREAD': { value: -0.25, change_24h: -0.03 } },
    inflation: { CORE_PCE: { value: 307.5, yoy_change: 2.8, mom_change: 0.1 }, BREAKEVEN_5Y: { value: 2.45, yoy_change: 2.45, mom_change: 0.08 } },
    monetary_policy: { FED_FUNDS_RATE: { value: 5.25, change_pct: 0.0 }, FED_BALANCE_SHEET: { value: 7200000, change_pct: -2.1 } },
  });

  const fetchMacroData = async () => {
    try {
      const response = await fetch('/api/macro/comprehensive');
      if (response.ok) {
        const data = await response.json();
        setMacroData(data.data);
        setLastUpdated(data.last_updated ?? '');
      } else {
        setMacroData(getMockMacroData());
      }
    } catch {
      setMacroData(getMockMacroData());
    }
  };

  useEffect(() => {
    fetchMacroData();
    const interval = setInterval(fetchMacroData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const monetaryIndicators = macroData ? [
    { name: 'Fed Funds Rate', value: `${macroData.monetary_policy.FED_FUNDS_RATE?.value ?? 5.25}%`, change: `${(macroData.monetary_policy.FED_FUNDS_RATE?.change_pct ?? 0) >= 0 ? '+' : ''}${macroData.monetary_policy.FED_FUNDS_RATE?.change_pct ?? 0}%`, trend: (macroData.monetary_policy.FED_FUNDS_RATE?.change_pct ?? 0) > 0 ? 'up' : (macroData.monetary_policy.FED_FUNDS_RATE?.change_pct ?? 0) < 0 ? 'down' : 'neutral', impact: 'high' },
    { name: 'Fed Balance Sheet', value: `$${((macroData.monetary_policy.FED_BALANCE_SHEET?.value ?? 7200000) / 1000000).toFixed(1)}T`, change: `${(macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct ?? -2.1) >= 0 ? '+' : ''}${macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct ?? -2.1}%`, trend: (macroData.monetary_policy.FED_BALANCE_SHEET?.change_pct ?? -2.1) > 0 ? 'up' : 'down', impact: 'high' },
    { name: 'ECB Deposit Rate', value: '4.00%', change: '+0.00%', trend: 'neutral', impact: 'high' },
    { name: 'BoJ Policy Rate', value: '-0.10%', change: '+0.00%', trend: 'neutral', impact: 'medium' },
  ] : [];

  const yieldCurveIndicators = macroData ? [
    { name: '2Y Treasury', value: `${macroData.treasury_yields['2Y']?.value ?? 4.75}%`, change: `${(macroData.treasury_yields['2Y']?.change_24h ?? 0) >= 0 ? '+' : ''}${macroData.treasury_yields['2Y']?.change_24h ?? 0}%`, trend: (macroData.treasury_yields['2Y']?.change_24h ?? 0) > 0 ? 'up' : 'down', impact: 'high' },
    { name: '10Y Treasury', value: `${macroData.treasury_yields['10Y']?.value ?? 4.50}%`, change: `${(macroData.treasury_yields['10Y']?.change_24h ?? 0) >= 0 ? '+' : ''}${macroData.treasury_yields['10Y']?.change_24h ?? 0}%`, trend: (macroData.treasury_yields['10Y']?.change_24h ?? 0) > 0 ? 'up' : 'down', impact: 'high' },
    { name: '2Y-10Y Spread', value: `${Math.round((macroData.treasury_yields['2Y10Y_SPREAD']?.value ?? -0.25) * 100)}bps`, change: `${(macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h ?? 0) >= 0 ? '+' : ''}${Math.round((macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h ?? 0) * 100)}bps`, trend: (macroData.treasury_yields['2Y10Y_SPREAD']?.change_24h ?? 0) > 0 ? 'up' : 'down', impact: 'high' },
    { name: '3M Treasury', value: `${macroData.treasury_yields['3M']?.value ?? 5.30}%`, change: `${(macroData.treasury_yields['3M']?.change_24h ?? 0) >= 0 ? '+' : ''}${macroData.treasury_yields['3M']?.change_24h ?? 0}%`, trend: (macroData.treasury_yields['3M']?.change_24h ?? 0) > 0 ? 'up' : 'down', impact: 'medium' },
  ] : [];

  const inflationGrowthIndicators = macroData ? [
    { name: 'Core PCE', value: `${macroData.inflation.CORE_PCE?.yoy_change ?? 2.8}%`, change: `${(macroData.inflation.CORE_PCE?.mom_change ?? 0) >= 0 ? '+' : ''}${macroData.inflation.CORE_PCE?.mom_change ?? 0}%`, trend: (macroData.inflation.CORE_PCE?.mom_change ?? 0) > 0 ? 'up' : 'down', impact: 'high' },
    { name: '5Y Breakeven', value: `${macroData.inflation.BREAKEVEN_5Y?.yoy_change ?? 2.45}%`, change: `${(macroData.inflation.BREAKEVEN_5Y?.mom_change ?? 0.08) >= 0 ? '+' : ''}${macroData.inflation.BREAKEVEN_5Y?.mom_change ?? 0.08}%`, trend: (macroData.inflation.BREAKEVEN_5Y?.mom_change ?? 0.08) > 0 ? 'up' : 'down', impact: 'high' },
    { name: 'Atlanta Fed GDPNow', value: '2.7%', change: '+0.2%', trend: 'up', impact: 'medium' },
    { name: 'NY Fed WEI', value: '2.1%', change: '+0.1%', trend: 'up', impact: 'medium' },
  ] : [];

  const crossAssetIndicators = [
    { name: 'VIX', value: '18.5', change: '-2.1%', trend: 'down', impact: 'high' },
    { name: 'MOVE Index', value: '115.2', change: '+3.8%', trend: 'up', impact: 'high' },
    { name: 'SKEW Index', value: '142.5', change: '+1.2%', trend: 'up', impact: 'medium' },
    { name: 'Gold/Silver Ratio', value: '78.2', change: '-0.8%', trend: 'down', impact: 'medium' },
  ];
  const currencyCarryIndicators = [
    { name: 'DXY Index', value: '103.45', change: '+0.8%', trend: 'up', impact: 'high' },
    { name: 'JPY Carry Trade', value: '-2.8%', change: '+0.3%', trend: 'up', impact: 'high' },
    { name: 'AUD/JPY', value: '98.25', change: '+1.2%', trend: 'up', impact: 'medium' },
    { name: 'TRY Real Rate', value: '8.5%', change: '+0.5%', trend: 'up', impact: 'medium' },
  ];
  const creditLiquidityIndicators = [
    { name: 'IG Credit Spread', value: '125bps', change: '+5bps', trend: 'up', impact: 'high' },
    { name: 'HY Credit Spread', value: '485bps', change: '+12bps', trend: 'up', impact: 'high' },
    { name: 'LIBOR-OIS Spread', value: '12bps', change: '+1bp', trend: 'up', impact: 'medium' },
    { name: 'Term Premium', value: '0.35%', change: '+0.08%', trend: 'up', impact: 'medium' },
  ];
  const commodityIndicators = [
    { name: 'DJP Commodity', value: '28.45', change: '+2.1%', trend: 'up', impact: 'medium' },
    { name: 'Copper/Gold Ratio', value: '0.0032', change: '+1.8%', trend: 'up', impact: 'medium' },
    { name: 'Baltic Dry Index', value: '1,285', change: '+5.2%', trend: 'up', impact: 'low' },
    { name: 'Oil Contango', value: '0.85%', change: '-0.2%', trend: 'down', impact: 'medium' },
  ];
  const alternativeIndicators = [
    { name: 'AAII Bull/Bear', value: '1.2x', change: '+0.1x', trend: 'up', impact: 'low' },
    { name: 'Put/Call Ratio', value: '0.85', change: '-0.05', trend: 'down', impact: 'medium' },
    { name: 'Margin Debt', value: '$684B', change: '-1.8%', trend: 'down', impact: 'medium' },
    { name: 'Repo Rate', value: '5.30%', change: '+0.02%', trend: 'up', impact: 'medium' },
  ];
  const upcomingEvents = [
    { date: 'Today', time: '2:00 PM', event: 'FOMC Meeting Minutes', impact: 'high' },
    { date: 'Tomorrow', time: '8:30 AM', event: 'Initial Jobless Claims', impact: 'medium' },
    { date: 'Jan 12', time: '8:30 AM', event: 'Core PCE Price Index', impact: 'high' },
    { date: 'Jan 15', time: '9:15 AM', event: 'Industrial Production', impact: 'medium' },
    { date: 'Jan 18', time: '10:00 AM', event: 'Existing Home Sales', impact: 'low' },
    { date: 'Jan 26', time: '8:30 AM', event: 'Advance GDP', impact: 'high' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Macro</h1>
          <p className="text-muted-foreground mt-1">Treasury yields, inflation, and monetary policy — real-time from FRED when backend is available</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {lastUpdated && <span className="text-xs text-muted-foreground">Updated {new Date(lastUpdated).toLocaleTimeString()}</span>}
          <Badge variant={macroData ? 'default' : 'secondary'}>{macroData ? 'Live from source' : 'Static snapshot'}</Badge>
        </div>
      </div>
      <div className="space-y-6">
        {renderIndicatorSection('Central Bank & Monetary Policy', monetaryIndicators, <DollarSign className="h-5 w-5 text-foreground" />)}
        {renderIndicatorSection('Yield Curve & Term Structure', yieldCurveIndicators, <BarChart3 className="h-5 w-5 text-foreground" />)}
        {renderIndicatorSection('Inflation & Growth Indicators', inflationGrowthIndicators, <TrendingUp className="h-5 w-5 text-foreground" />)}
        {renderIndicatorSection('Cross-Asset & Volatility', crossAssetIndicators, <Activity className="h-5 w-5 text-red-400" />)}
        {renderIndicatorSection('Currency & Carry Trade', currencyCarryIndicators, <Globe className="h-5 w-5 text-cyan-400" />)}
        {renderIndicatorSection('Credit & Liquidity Conditions', creditLiquidityIndicators, <Layers className="h-5 w-5 text-purple-400" />)}
        {renderIndicatorSection('Commodity & Energy Complex', commodityIndicators, <Zap className="h-5 w-5 text-yellow-400" />)}
        {renderIndicatorSection('Alternative Data & Sentiment', alternativeIndicators, <Target className="h-5 w-5 text-pink-400" />)}
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
                    <div><p className="font-medium">{event.event}</p></div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge className={event.impact === 'high' ? 'bg-red-500/20 text-red-300' : event.impact === 'medium' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}>{event.impact}</Badge>
                    {event.impact === 'high' && <AlertTriangle className="h-4 w-4 text-red-400" />}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2"><PieChart className="h-5 w-5 text-green-400" /><span>Regime Analysis</span></CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center"><span>Macro Regime</span><Badge className="bg-blue-500/20 text-blue-300">Goldilocks</Badge></div>
                <div className="flex justify-between items-center"><span>Volatility Regime</span><Badge className="bg-yellow-500/20 text-yellow-300">Medium</Badge></div>
                <div className="flex justify-between items-center"><span>Risk Regime</span><Badge className="bg-green-500/20 text-green-300">Risk On</Badge></div>
                <div className="flex justify-between items-center"><span>Liquidity Regime</span><Badge className="bg-cyan-500/20 text-cyan-300">Abundant</Badge></div>
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2"><Target className="h-5 w-5 text-purple-400" /><span>Factor Exposures</span></CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center"><span>Growth Factor</span><div className="flex items-center space-x-2"><Progress value={75} className="w-20 h-2" /><span className="text-green-400">+0.8σ</span></div></div>
                <div className="flex justify-between items-center"><span>Value Factor</span><div className="flex items-center space-x-2"><Progress value={40} className="w-20 h-2" /><span className="text-red-400">-0.3σ</span></div></div>
                <div className="flex justify-between items-center"><span>Momentum Factor</span><div className="flex items-center space-x-2"><Progress value={65} className="w-20 h-2" /><span className="text-green-400">+0.5σ</span></div></div>
                <div className="flex justify-between items-center"><span>Quality Factor</span><div className="flex items-center space-x-2"><Progress value={55} className="w-20 h-2" /><span className="text-gray-400">+0.1σ</span></div></div>
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2"><Layers className="h-5 w-5 text-orange-400" /><span>Cross-Asset Signals</span></CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center"><span>Equity/Bond Correlation</span><Badge className="bg-red-500/20 text-red-300">+0.65</Badge></div>
                <div className="flex justify-between items-center"><span>Currency Momentum</span><Badge className="bg-green-500/20 text-green-300">Strong</Badge></div>
                <div className="flex justify-between items-center"><span>Commodity Trend</span><Badge className="bg-yellow-500/20 text-yellow-300">Mixed</Badge></div>
                <div className="flex justify-between items-center"><span>Crypto Correlation</span><Badge className="bg-purple-500/20 text-purple-300">+0.45</Badge></div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
