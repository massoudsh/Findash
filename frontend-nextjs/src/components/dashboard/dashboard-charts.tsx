'use client';

import {
  PieChart as RechartsPie,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
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

// --- Bar chart (one per dashboard) ---
interface BarDataPoint {
  name: string;
  value: number;
  amount?: number;
}

interface DashboardBarChartProps {
  data: BarDataPoint[];
  title: string;
  subtitle?: string;
  height?: number;
  dataKey?: string;
  barColor?: string;
  formatTick?: (value: number) => string;
  className?: string;
}

export function DashboardBarChart({
  data,
  title,
  subtitle,
  height = 280,
  dataKey = 'value',
  barColor = '#3b82f6',
  formatTick = (v) => formatCurrency(v),
  className = '',
}: DashboardBarChartProps) {
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
            <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : String(v))} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                formatter={(value: number) => [formatTick(value), '']}
                labelFormatter={(name) => name}
              />
              <Bar dataKey={dataKey} fill={barColor} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// --- Waterfall chart (one per dashboard) ---
interface WaterfallPoint {
  name: string;
  start: number;
  delta: number;
}

interface DashboardWaterfallChartProps {
  data: WaterfallPoint[];
  title: string;
  subtitle?: string;
  height?: number;
  className?: string;
}

export function DashboardWaterfallChart({
  data,
  title,
  subtitle,
  height = 280,
  className = '',
}: DashboardWaterfallChartProps) {
  const showHeader = title || subtitle;
  const maxVal = Math.max(...data.map((d) => d.start + Math.max(0, d.delta)), 1);
  const minVal = Math.min(...data.map((d) => d.start + Math.min(0, d.delta)), 0);
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
            <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}k` : `$${v}`)} domain={[minVal, maxVal]} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                formatter={(value: number) => [value >= 0 ? `+${formatCurrency(value)}` : formatCurrency(value), 'Delta']}
                labelFormatter={(name) => name}
              />
              <Bar dataKey="start" stackId="wf" fill="transparent" />
              <Bar dataKey="delta" stackId="wf" radius={[4, 4, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell
                    key={`wf-${index}`}
                    fill={entry.delta > 0 ? '#22c55e' : entry.delta < 0 ? '#ef4444' : '#94a3b8'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// --- Line chart (one per dashboard) ---
interface LineDataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface DashboardLineChartProps {
  data: LineDataPoint[];
  title: string;
  subtitle?: string;
  height?: number;
  dataKey?: string;
  strokeColor?: string;
  formatTick?: (value: number) => string;
  className?: string;
}

export function DashboardLineChart({
  data,
  title,
  subtitle,
  height = 280,
  dataKey = 'value',
  strokeColor = '#3b82f6',
  formatTick = (v) => formatCurrency(v),
  className = '',
}: DashboardLineChartProps) {
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
            <LineChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : String(v))} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                formatter={(value: number) => [formatTick(value), '']}
                labelFormatter={(name) => name}
              />
              <Line type="monotone" dataKey={dataKey} stroke={strokeColor} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
