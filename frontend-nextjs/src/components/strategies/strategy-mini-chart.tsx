'use client';

import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';

/** Mock data shapes that visually suggest how each strategy works */
const STRATEGY_CHART_DATA: Record<string, { name: string; value: number }[]> = {
  momentum: [
    { name: '1', value: 98 }, { name: '2', value: 100 }, { name: '3', value: 104 },
    { name: '4', value: 107 }, { name: '5', value: 112 }, { name: '6', value: 118 },
    { name: '7', value: 122 }, { name: '8', value: 128 }, { name: '9', value: 132 },
    { name: '10', value: 138 },
  ],
  mean_reversion: [
    { name: '1', value: 100 }, { name: '2', value: 92 }, { name: '3', value: 88 },
    { name: '4', value: 94 }, { name: '5', value: 98 }, { name: '6', value: 102 },
    { name: '7', value: 99 }, { name: '8', value: 101 }, { name: '9', value: 100 },
    { name: '10', value: 100 },
  ],
  technical: [
    { name: '1', value: 100 }, { name: '2', value: 102 }, { name: '3', value: 98 },
    { name: '4', value: 105 }, { name: '5', value: 103 }, { name: '6', value: 108 },
    { name: '7', value: 106 }, { name: '8', value: 110 }, { name: '9', value: 107 },
    { name: '10', value: 112 },
  ],
  risk_aware: [
    { name: '1', value: 100 }, { name: '2', value: 99.5 }, { name: '3', value: 100.2 },
    { name: '4', value: 100.8 }, { name: '5', value: 100.1 }, { name: '6', value: 101 },
    { name: '7', value: 100.5 }, { name: '8', value: 101.2 }, { name: '9', value: 101 },
    { name: '10', value: 101.5 },
  ],
  breakout: [
    { name: '1', value: 98 }, { name: '2', value: 99 }, { name: '3', value: 99 },
    { name: '4', value: 100 }, { name: '5', value: 100 }, { name: '6', value: 105 },
    { name: '7', value: 112 }, { name: '8', value: 118 }, { name: '9', value: 122 },
    { name: '10', value: 128 },
  ],
  volatility_spread: [
    { name: 'IV', value: 45 }, { name: 'HV', value: 38 }, { name: 'IV', value: 42 },
    { name: 'HV', value: 40 }, { name: 'IV', value: 48 }, { name: 'HV', value: 35 },
    { name: 'IV', value: 44 }, { name: 'HV', value: 41 }, { name: 'IV', value: 46 },
    { name: 'HV', value: 39 },
  ],
};

const DEFAULT_CHART_DATA = [
  { name: '1', value: 100 }, { name: '2', value: 101 }, { name: '3', value: 102 },
  { name: '4', value: 101 }, { name: '5', value: 103 }, { name: '6', value: 104 },
  { name: '7', value: 103 }, { name: '8', value: 105 }, { name: '9', value: 106 },
  { name: '10', value: 105 },
];

interface StrategyMiniChartProps {
  strategyType: string;
  className?: string;
  height?: number;
}

export function StrategyMiniChart({ strategyType, className = '', height = 64 }: StrategyMiniChartProps) {
  const data = useMemo(
    () => STRATEGY_CHART_DATA[strategyType] ?? DEFAULT_CHART_DATA,
    [strategyType]
  );

  const isVolatility = strategyType === 'volatility_spread';
  const color = isVolatility ? '#8b5cf6' : undefined;
  const stroke = isVolatility ? '#a78bfa' : '#22c55e';

  return (
    <div className={`w-full rounded-md overflow-hidden bg-black/20 ${className}`} style={{ height }}>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
          <defs>
            <linearGradient id={`grad-${strategyType}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.4} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="name" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ fontSize: 10, padding: '4px 8px' }}
            formatter={(value: number) => [value, isVolatility ? 'Vol %' : 'Level']}
            labelFormatter={(label) => `Point ${label}`}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={stroke}
            fill={`url(#grad-${strategyType})`}
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
