"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { formatToman } from "@/lib/assets";

interface SparklineProps {
  data: { timestamp: string; close: number }[];
  positive: boolean;
}

/** Compact sparkline for use inside AssetCard */
export function SparklineChart({ data, positive }: SparklineProps) {
  const color = positive ? "#34d399" : "#f87171";

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 2, right: 4, left: 4, bottom: 2 }}>
        <defs>
          <linearGradient id={`sparkGrad-${positive}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0}   />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="close"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#sparkGrad-${positive})`}
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

interface FullChartProps {
  data: { timestamp: string; close: number; open?: number; high?: number; low?: number }[];
  symbol: string;
  nameFa: string;
}

/** Full-size price chart with tooltip — used in detail modal/page */
export function AssetFullChart({ data, symbol, nameFa }: FullChartProps) {
  const first = data[0]?.close ?? 0;
  const last  = data[data.length - 1]?.close ?? 0;
  const isPositive = last >= first;
  const color = isPositive ? "#34d399" : "#f87171";

  const formatted = data.map((d) => ({
    ...d,
    date: new Date(d.timestamp).toLocaleDateString("fa-IR", {
      month: "short",
      day: "numeric",
    }),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={formatted} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="fullGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0}    />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#888", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={["auto", "auto"]}
          tick={{ fill: "#888", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => formatToman(v)}
          width={90}
        />
        <Tooltip
          contentStyle={{
            background: "#1a1d27",
            border: "1px solid #ffffff15",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#aaa" }}
          formatter={(value: number) => [formatToman(value), "قیمت"]}
        />
        <Area
          type="monotone"
          dataKey="close"
          stroke={color}
          strokeWidth={2}
          fill="url(#fullGrad)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
