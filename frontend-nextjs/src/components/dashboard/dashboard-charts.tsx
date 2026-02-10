'use client';

import {
  PieChart as RechartsPie,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCurrency, formatPercentage } from '@/lib/utils';

const PIE_COLORS = [
  '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
  '#ef4444', '#ec4899', '#6366f1', '#14b8a6', '#84cc16',
];

interface PieSlice {
  name: string;
  value: number;
  amount?: number;
}

interface DashboardDonutChartProps {
  data: PieSlice[];
  title: string;
  subtitle?: string;
  centerLabel?: string;
  centerSublabel?: string;
  showLegend?: boolean;
  height?: number;
  formatValue?: (value: number, name: string) => string;
  className?: string;
}

export function DashboardDonutChart({
  data,
  title,
  subtitle,
  centerLabel,
  centerSublabel,
  showLegend = true,
  height = 280,
  formatValue,
  className = '',
}: DashboardDonutChartProps) {
  const total = data.reduce((s, d) => s + d.value, 0);
  const showHeader = title || subtitle;
  const defaultFormat = (value: number, name: string) => {
    const pct = total ? (value / total) * 100 : 0;
    const slice = data.find((d) => d.name === name);
    if (slice?.amount != null) return `${name}: ${formatCurrency(slice.amount)} (${pct.toFixed(1)}%)`;
    return `${name}: ${pct.toFixed(1)}%`;
  };
  const formatter = formatValue ?? defaultFormat;

  return (
    <Card className={`overflow-hidden ${className}`}>
      {showHeader && (
        <CardHeader className="pb-2">
          {title && <CardTitle className="text-base font-semibold">{title}</CardTitle>}
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </CardHeader>
      )}
      <CardContent className={showHeader ? 'pt-0' : ''}>
        <div className="relative" style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPie>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={height * 0.28}
                outerRadius={height * 0.38}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
                stroke="var(--card)"
                strokeWidth={2}
              >
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid hsl(var(--border))',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                }}
                formatter={(value: number, name: string) => [formatter(value, name), '']}
                labelFormatter={(name) => name}
              />
              {showLegend && (
                <Legend
                  layout="horizontal"
                  align="center"
                  verticalAlign="bottom"
                  formatter={(value) => (
                    <span className="text-muted-foreground text-xs">{value}</span>
                  )}
                />
              )}
            </RechartsPie>
          </ResponsiveContainer>
          {centerLabel != null && (
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-2xl font-bold text-foreground">{centerLabel}</span>
              {centerSublabel && (
                <span className="text-xs text-muted-foreground mt-0.5">{centerSublabel}</span>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface DashboardPieChartProps {
  data: PieSlice[];
  title: string;
  subtitle?: string;
  height?: number;
  className?: string;
}

export function DashboardPieChart({
  data,
  title,
  subtitle,
  height = 280,
  className = '',
}: DashboardPieChartProps) {
  const total = data.reduce((s, d) => s + d.value, 0);

  const showHeader = title || subtitle;

  return (
    <Card className={`overflow-hidden ${className}`}>
      {showHeader && (
        <CardHeader className="pb-2">
          {title && <CardTitle className="text-base font-semibold">{title}</CardTitle>}
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </CardHeader>
      )}
      <CardContent className={showHeader ? 'pt-0' : ''}>
        <div style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPie>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                outerRadius={height * 0.36}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={{ strokeWidth: 1 }}
                stroke="var(--card)"
                strokeWidth={2}
              >
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid hsl(var(--border))',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                }}
                formatter={(value: number, name: string) => {
                  const slice = data.find((d) => d.name === name);
                  const pct = total ? ((value / total) * 100).toFixed(1) : '0';
                  if (slice?.amount != null) return [formatCurrency(slice.amount) + ` (${pct}%)`, ''];
                  return [`${pct}%`, ''];
                }}
              />
              <Legend
                layout="horizontal"
                align="center"
                verticalAlign="bottom"
                formatter={(value) => <span className="text-muted-foreground text-xs">{value}</span>}
              />
            </RechartsPie>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
