'use client';

import { useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Treemap,
  Cell,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';

const TIME_FILTERS = ['1h', '24h', '7d'] as const;

function GaugeSegment({
  segments,
  value,
  className,
}: {
  segments: { label: string; color: string; width: number }[];
  value: number;
  className?: string;
}) {
  const total = segments.reduce((s, seg) => s + seg.width, 0);
  const pct = Math.min(Math.max(value, 0), 100);
  return (
    <div className={cn('w-full relative', className)}>
      <div className="flex h-8 rounded-lg overflow-hidden bg-muted/50">
        {segments.map((seg) => {
          const segPct = (seg.width / total) * 100;
          return (
            <div
              key={seg.label}
              className="flex items-center justify-center text-[10px] font-medium text-white truncate"
              style={{ width: `${segPct}%`, backgroundColor: seg.color, minWidth: 0 }}
            >
              {seg.label}
            </div>
          );
        })}
      </div>
      <div
        className="absolute top-0 bottom-0 w-1 rounded-full bg-foreground shadow-md z-10 transition-all pointer-events-none"
        style={{ left: `calc(${pct}% - 2px)` }}
      />
    </div>
  );
}

const marketCapData = [
  { t: 'Jan', v: 2.8 },
  { t: 'Feb', v: 2.65 },
  { t: 'Mar', v: 2.9 },
  { t: 'Apr', v: 2.5 },
  { t: 'May', v: 2.41 },
];

const futuresSpotData = [
  { name: 'Mon', spot: 18, futures: 22 },
  { name: 'Tue', spot: 20, futures: 24 },
  { name: 'Wed', spot: 19, futures: 26 },
  { name: 'Thu', spot: 21, futures: 23 },
  { name: 'Fri', spot: 22, futures: 25 },
];

const spotVolumeData = [
  { name: 'BTC', size: 6400, fill: '#22c55e' },
  { name: 'ETH', size: 3490, fill: '#ef4444' },
  { name: 'USDC', size: 1130, fill: '#3b82f6' },
  { name: 'SOL', size: 686, fill: '#8b5cf6' },
  { name: 'XRP', size: 756, fill: '#64748b' },
  { name: 'BNB', size: 276, fill: '#f59e0b' },
  { name: 'USD1', size: 304, fill: '#14b8a6' },
];
const spotVolumeTree = [{ name: 'Volume', children: spotVolumeData }];

const topGainers = [
  { symbol: 'TRX', price: 0.284057, change: 1.5 },
  { symbol: 'BTC', price: 67118.99, change: 0.96 },
  { symbol: 'SOL', price: 82.11, change: 0.82 },
  { symbol: 'BCH', price: 560.86, change: 0.1 },
  { symbol: 'WBT', price: 50.4, change: -0.08 },
  { symbol: 'ETH', price: 1945.44, change: -0.6 },
  { symbol: 'BNB', price: 606.23, change: -0.82 },
  { symbol: 'DOGE', price: 0.09834, change: -1.1 },
];

const sectorPerformanceData = [
  { t: 'Mon', BTC: 2, ETH: 1.5, L2: 3, DeFi: 1, AI: 5, Meme: 8 },
  { t: 'Tue', BTC: 2.5, ETH: 2, L2: 4, DeFi: 2, AI: 6, Meme: 10 },
  { t: 'Wed', BTC: 1.8, ETH: 1.2, L2: 2.5, DeFi: 0.5, AI: 4, Meme: 6 },
  { t: 'Thu', BTC: 3, ETH: 2.5, L2: 5, DeFi: 2.5, AI: 7, Meme: 12 },
  { t: 'Fri', BTC: 2.2, ETH: 1.8, L2: 3.5, DeFi: 1.5, AI: 5.5, Meme: 9 },
];

const supplyInProfitData = [
  { asset: 'BTC', value: 98, fill: '#f97316' },
  { asset: 'ETH', value: 95, fill: '#6366f1' },
  { asset: 'XRP', value: 88, fill: '#64748b' },
  { asset: 'BNB', value: 92, fill: '#f59e0b' },
  { asset: 'SOL', value: 94, fill: '#8b5cf6' },
  { asset: 'TRX', value: 85, fill: '#ef4444' },
  { asset: 'DOGE', value: 82, fill: '#eab308' },
  { asset: 'LINK', value: 90, fill: '#06b6d4' },
];

const CATEGORIES = [
  'Stablecoins',
  'Layer 1',
  'Layer 2',
  'Web3',
  'Meme',
  'Tokenized',
  'AI',
  'Staking',
  'DeFi',
  'DePIN',
  'Exchange',
  'Gaming',
  'NFT',
  'RWA',
  'Governance',
];

export function CryptoAnalyticsDashboard() {
  const [timeFilter, setTimeFilter] = useState<(typeof TIME_FILTERS)[number]>('24h');
  const [sectionTab, setSectionTab] = useState('Overview');
  const [category, setCategory] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <Tabs value={sectionTab} onValueChange={setSectionTab} className="w-full">
        <TabsList className="flex flex-wrap h-auto gap-1 bg-muted/50 p-1">
          {['Overview', 'Fundamentals', 'Profit & Loss', 'Supply Dynamics', 'Futures'].map(
            (tab) => (
              <TabsTrigger key={tab} value={tab} className="text-sm">
                {tab}
              </TabsTrigger>
            )
          )}
        </TabsList>

        <div className="flex flex-wrap gap-2 mt-3">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setCategory(category === cat ? null : cat)}
              className={cn(
                'px-3 py-1.5 rounded-md text-xs font-medium transition-colors',
                category === cat
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted/70 text-muted-foreground hover:bg-muted'
              )}
            >
              {cat}
            </button>
          ))}
        </div>

        <TabsContent value="Overview" className="mt-4 space-y-4">
          {/* Row 1: Market Cap, Futures vs Spot, Altcoin Cycle, BTC Sharpe */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm font-semibold">Market Cap</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-16">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={marketCapData}>
                      <Line type="monotone" dataKey="v" stroke="#ef4444" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-lg font-bold text-foreground mt-1">$2.41T</p>
                <p className="text-xs text-red-500 font-medium">-23.9%</p>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm font-semibold">Futures vs Spot Vol</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={futuresSpotData} layout="vertical" margin={{ left: 0, right: 0 }}>
                      <XAxis type="number" hide />
                      <YAxis type="category" dataKey="name" width={28} tick={{ fontSize: 10 }} />
                      <Bar dataKey="spot" fill="#3b82f6" radius={[0, 2, 2, 0]} />
                      <Bar dataKey="futures" fill="#8b5cf6" radius={[0, 2, 2, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm font-semibold">Altcoin Cycle Index</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <GaugeSegment
                  segments={[
                    { label: 'Bitcoin', color: '#f97316', width: 33 },
                    { label: 'Neutral', color: '#eab308', width: 34 },
                    { label: 'Altcoin', color: '#3b82f6', width: 33 },
                  ]}
                  value={75}
                />
                <p className="text-xs text-muted-foreground mt-2 text-center">Altcoin phase</p>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm font-semibold">BTC Sharpe Signal</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <GaugeSegment
                  segments={[
                    { label: 'Risk Off', color: '#ef4444', width: 33 },
                    { label: 'Neutral', color: '#eab308', width: 34 },
                    { label: 'Risk On', color: '#22c55e', width: 33 },
                  ]}
                  value={58}
                />
                <p className="text-xs text-muted-foreground mt-2 text-center">Risk On</p>
              </CardContent>
            </Card>
          </div>

          {/* Row 2: Spot Volume, Top Gainers, Sector Performance, Supply In Profit */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-semibold">Spot Volume</CardTitle>
                <div className="flex gap-1">
                  {TIME_FILTERS.map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setTimeFilter(t)}
                      className={cn(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        timeFilter === t ? 'bg-primary text-primary-foreground' : 'bg-muted/70'
                      )}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-48 rounded-lg overflow-hidden">
                  <ResponsiveContainer width="100%" height="100%">
                    <Treemap
                      data={spotVolumeTree}
                      dataKey="size"
                      stroke="hsl(var(--border))"
                      content={((props: TreemapContentProps) => <CustomTreemapContent {...props} />) as unknown as React.ReactElement}
                    />
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-semibold">Top Gainers - Large</CardTitle>
                <div className="flex gap-1">
                  {TIME_FILTERS.map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setTimeFilter(t)}
                      className={cn(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        timeFilter === t ? 'bg-primary text-primary-foreground' : 'bg-muted/70'
                      )}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <ul className="space-y-2">
                  {topGainers.map((row) => (
                    <li
                      key={row.symbol}
                      className="flex items-center justify-between text-sm py-1 border-b border-border/50 last:border-0"
                    >
                      <span className="font-medium">{row.symbol}</span>
                      <span className="text-muted-foreground">
                        ${row.price < 1 ? row.price.toFixed(6) : row.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                      <span className={row.change >= 0 ? 'text-green-500' : 'text-red-500'}>
                        {row.change >= 0 ? '+' : ''}{row.change}%
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1 flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-semibold">Sector Performance</CardTitle>
                <div className="flex gap-1">
                  {TIME_FILTERS.map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => setTimeFilter(t)}
                      className={cn(
                        'px-2 py-0.5 rounded text-xs font-medium',
                        timeFilter === t ? 'bg-primary text-primary-foreground' : 'bg-muted/70'
                      )}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={sectorPerformanceData}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="t" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                      <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line type="monotone" dataKey="BTC" stroke="#f97316" strokeWidth={1.5} dot={false} name="BTC" />
                      <Line type="monotone" dataKey="ETH" stroke="#6366f1" strokeWidth={1.5} dot={false} name="ETH" />
                      <Line type="monotone" dataKey="L2" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="L2" />
                      <Line type="monotone" dataKey="DeFi" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="DeFi" />
                      <Line type="monotone" dataKey="AI" stroke="#22c55e" strokeWidth={1.5} dot={false} name="AI" />
                      <Line type="monotone" dataKey="Meme" stroke="#ec4899" strokeWidth={1.5} dot={false} name="Meme" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/80">
              <CardHeader className="pb-1">
                <CardTitle className="text-sm font-semibold">Supply In Profit</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={supplyInProfitData} layout="vertical" margin={{ left: 40, right: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" horizontal={false} />
                      <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                      <YAxis type="category" dataKey="asset" width={36} tick={{ fontSize: 10 }} />
                      <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                        {supplyInProfitData.map((entry, i) => (
                          <Cell key={entry.asset} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="Fundamentals" className="mt-4">
          <Card className="border-border/50 bg-card/80">
            <CardHeader>
              <CardTitle className="text-base">Fundamentals</CardTitle>
              <p className="text-sm text-muted-foreground">
                On-chain and market fundamental metrics. Connect data sources in Data & Charts for live data.
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['NVT Ratio', 'MVRV', 'SOPR', 'Active Addresses'].map((m) => (
                  <div key={m} className="p-4 rounded-lg bg-muted/50 text-center">
                    <p className="text-xs text-muted-foreground">{m}</p>
                    <p className="text-lg font-semibold mt-1">—</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="Profit & Loss" className="mt-4">
          <Card className="border-border/50 bg-card/80">
            <CardHeader>
              <CardTitle className="text-base">Profit & Loss</CardTitle>
              <p className="text-sm text-muted-foreground">
                Realized and unrealized P&L by cohort. Available when profit/loss data is connected.
              </p>
            </CardHeader>
          </Card>
        </TabsContent>

        <TabsContent value="Supply Dynamics" className="mt-4">
          <Card className="border-border/50 bg-card/80">
            <CardHeader>
              <CardTitle className="text-base">Supply Dynamics</CardTitle>
              <p className="text-sm text-muted-foreground">
                Supply distribution, HODL waves, and transfer volume.
              </p>
            </CardHeader>
          </Card>
        </TabsContent>

        <TabsContent value="Futures" className="mt-4">
          <Card className="border-border/50 bg-card/80">
            <CardHeader>
              <CardTitle className="text-base">Futures</CardTitle>
              <p className="text-sm text-muted-foreground">
                Open interest, funding rates, and liquidations. See Command Center → Options for funding.
              </p>
            </CardHeader>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface TreemapContentProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  name?: string;
  value?: number;
  depth?: number;
  index?: number;
  fill?: string;
}

function CustomTreemapContent(props: TreemapContentProps) {
  const { x = 0, y = 0, width = 0, height = 0, name, value = 0, fill } = props;
  if (width < 24 || height < 18) return null;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={fill} stroke="hsl(var(--border))" strokeWidth={1} rx={2} />
      <text x={x + width / 2} y={y + height / 2 - 4} textAnchor="middle" fill="white" fontSize={10} fontWeight="600">
        {name}
      </text>
      <text x={x + width / 2} y={y + height / 2 + 6} textAnchor="middle" fill="rgba(255,255,255,0.9)" fontSize={9}>
        {value >= 1000 ? `$${(value / 1000).toFixed(2)}B` : `$${value}M`}
      </text>
    </g>
  );
}
