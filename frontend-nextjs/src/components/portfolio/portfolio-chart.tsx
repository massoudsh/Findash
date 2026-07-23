'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PieChart, BarChart3 } from 'lucide-react';

interface Asset {
  id: string;
  symbol: string;
  name: string;
  marketValue: number;
  allocation: number;
  sector: string;
  type: 'stock' | 'crypto' | 'etf' | 'bond' | 'commodity';
}

interface PortfolioChartProps {
  assets: Asset[];
  totalValue: number;
}

export function PortfolioChart({ assets, totalValue }: PortfolioChartProps) {
  // Calculate sector breakdown
  const sectorData = assets.reduce((acc, asset) => {
    if (!acc[asset.sector]) {
      acc[asset.sector] = {
        value: 0,
        percentage: 0,
        count: 0
      };
    }
    acc[asset.sector].value += asset.marketValue;
    acc[asset.sector].count += 1;
    return acc;
  }, {} as Record<string, { value: number; percentage: number; count: number }>);

  // Calculate percentages
  Object.keys(sectorData).forEach(sector => {
    sectorData[sector].percentage = (sectorData[sector].value / totalValue) * 100;
  });

  // Sort sectors by value
  const sortedSectors = Object.entries(sectorData)
    .sort(([, a], [, b]) => b.value - a.value)
    .slice(0, 8); // Top 8 sectors

  // Pie chart colors
  const pieColors = [
    '#3b82f6', // blue-500
    '#10b981', // emerald-500
    '#f59e0b', // amber-500
    '#ef4444', // red-500
    '#8b5cf6', // violet-500
    '#06b6d4', // cyan-500
    '#84cc16', // lime-500
    '#f97316', // orange-500
    '#ec4899', // pink-500
    '#6366f1', // indigo-500
  ];

  // Calculate pie chart segments
  const pieSegments = assets.slice(0, 10).map((asset, index) => ({
    ...asset,
    color: pieColors[index % pieColors.length],
    startAngle: assets.slice(0, index).reduce((sum, a) => sum + (a.allocation * 3.6), 0),
    endAngle: assets.slice(0, index + 1).reduce((sum, a) => sum + (a.allocation * 3.6), 0)
  }));

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Pie Chart - Asset Allocation */}
      <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <PieChart className="h-5 w-5 text-blue-400" />
            تخصیص دارایی
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center">
            <div className="relative">
              {/* SVG Pie Chart */}
              <svg width="280" height="280" viewBox="0 0 280 280" className="transform -rotate-90">
                <circle
                  cx="140"
                  cy="140"
                  r="120"
                  fill="none"
                  stroke="rgb(51, 65, 85)"
                  strokeWidth="2"
                />
                {pieSegments.map((segment, index) => {
                  const radius = 120;
                  const centerX = 140;
                  const centerY = 140;
                  const startAngle = (segment.startAngle - 90) * (Math.PI / 180);
                  const endAngle = (segment.endAngle - 90) * (Math.PI / 180);
                  
                  const x1 = centerX + radius * Math.cos(startAngle);
                  const y1 = centerY + radius * Math.sin(startAngle);
                  const x2 = centerX + radius * Math.cos(endAngle);
                  const y2 = centerY + radius * Math.sin(endAngle);
                  
                  const largeArcFlag = segment.allocation > 50 ? 1 : 0;
                  
                  const pathData = [
                    `M ${centerX} ${centerY}`,
                    `L ${x1} ${y1}`,
                    `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
                    'Z'
                  ].join(' ');
                  
                  return (
                    <path
                      key={segment.id}
                      d={pathData}
                      fill={segment.color}
                      stroke="rgb(15, 23, 42)"
                      strokeWidth="2"
                      className="hover:opacity-80 transition-opacity"
                    />
                  );
                })}
                {/* Center circle */}
                <circle
                  cx="140"
                  cy="140"
                  r="60"
                  fill="rgb(15, 23, 42)"
                  stroke="rgb(51, 65, 85)"
                  strokeWidth="2"
                />
                {/* Center text */}
                <text
                  x="140"
                  y="135"
                  textAnchor="middle"
                  className="fill-blue-400 text-sm font-semibold transform rotate-90"
                  style={{ transformOrigin: '140px 135px' }}
                >
                  مجموع
                </text>
                <text
                  x="140"
                  y="150"
                  textAnchor="middle"
                  className="fill-slate-300 text-xs font-medium transform rotate-90"
                  style={{ transformOrigin: '140px 150px' }}
                >
                  {formatCurrency(totalValue)}
                </text>
              </svg>
            </div>
          </div>
          
          {/* Legend */}
          <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
            {pieSegments.slice(0, 8).map((segment) => (
              <div key={segment.id} className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: segment.color }}
                />
                <span className="text-slate-300 text-xs truncate">
                  {segment.symbol} ({segment.allocation.toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Horizontal Bar Chart - Sector Breakdown */}
      <Card className="bg-slate-900/50 border-slate-700/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-slate-200">
            <BarChart3 className="h-5 w-5 text-blue-400" />
            تفکیک بخش‌ها
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {sortedSectors.map(([sector, data], index) => (
              <div key={sector} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: pieColors[index % pieColors.length] }}
                    />
                    <span className="text-slate-300 text-sm font-medium">{sector}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-slate-200 text-sm font-semibold">
                      {data.percentage.toFixed(1)}%
                    </div>
                    <div className="text-slate-400 text-xs">
                      {formatCurrency(data.value)}
                    </div>
                  </div>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: `${data.percentage}%`,
                      backgroundColor: pieColors[index % pieColors.length]
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          
          {/* Summary Stats */}
          <div className="mt-6 pt-4 border-t border-slate-700/50">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400">کل بخش‌ها</div>
                <div className="text-slate-200 font-semibold">{Object.keys(sectorData).length}</div>
              </div>
              <div>
                <div className="text-slate-400">برترین بخش</div>
                <div className="text-slate-200 font-semibold">
                  {sortedSectors[0]?.[0] || 'نامشخص'}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 