'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatCurrency } from '@/lib/utils';

interface BacktestChartProps {
  data: {
    date: string;
    value: number;
  }[];
}

export function BacktestChart({ data }: BacktestChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis tickFormatter={(value) => formatCurrency(value)} />
        <Tooltip
          formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
          labelStyle={{ fontWeight: 'bold' }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#8884d8"
          activeDot={{ r: 8 }}
          dot={false}
          name="Portfolio Value"
        />
      </LineChart>
    </ResponsiveContainer>
  );
} 