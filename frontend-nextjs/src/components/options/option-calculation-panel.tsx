'use client';

import { useMemo, useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Link2, X } from 'lucide-react';

export interface OptionContractSelection {
  underlying: string;
  strike: number;
  type: 'call' | 'put';
  expiry: string;
  premium: number;       // mark for selected type
  spot: number;
  callPremium?: number;  // when opening from chain (both sides)
  putPremium?: number;
  bid?: number;
  ask?: number;
}

interface OptionCalculationPanelProps {
  contract: OptionContractSelection;
  onClose?: () => void;
}

/** Long call: max loss = premium, break even = strike + premium, max profit = unlimited */
function payoffLongCall(spot: number, strike: number, premium: number, underlyingPrice: number): number {
  const intrinsic = Math.max(0, underlyingPrice - strike);
  return intrinsic - premium;
}

/** Long put: max loss = premium, break even = strike - premium, max profit = strike - premium (capped at strike) */
function payoffLongPut(spot: number, strike: number, premium: number, underlyingPrice: number): number {
  const intrinsic = Math.max(0, strike - underlyingPrice);
  return intrinsic - premium;
}

function formatCurrency(value: number, compact = false): string {
  if (Math.abs(value) >= 1000) return `$${value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function OptionCalculationPanel({ contract, onClose }: OptionCalculationPanelProps) {
  const [quantity, setQuantity] = useState(1);
  const [reduce, setReduce] = useState(false);
  const [post, setPost] = useState(false);
  const [timeInForce, setTimeInForce] = useState('GTC');
  const [side, setSide] = useState<'call' | 'put'>(contract.type);

  const { underlying, strike, expiry, spot, callPremium, putPremium } = contract;
  const premium = side === 'call' ? (callPremium ?? contract.premium) : (putPremium ?? contract.premium);
  const type = side;
  const isCrypto = underlying === 'BTC' || underlying === 'ETH';
  const contractMultiplier = isCrypto ? 1 : 100;

  const payoffFn = type === 'call' ? payoffLongCall : payoffLongPut;
  const breakEven = type === 'call' ? strike + premium : strike - premium;
  const maxLoss = premium * quantity * contractMultiplier;
  const maxProfit = type === 'call' ? Infinity : breakEven * quantity * contractMultiplier;

  // Keep side in sync when user selects a different contract from the chain
  useEffect(() => {
    setSide(contract.type);
  }, [contract.strike, contract.expiry, contract.type]);

  const chartData = useMemo(() => {
    const minP = Math.min(spot, strike, breakEven, strike * 0.5) * 0.9;
    const maxP = type === 'call' ? Math.max(spot, breakEven, strike * 1.5) * 1.15 : Math.max(spot, strike) * 1.1;
    const steps = 60;
    const dx = (maxP - minP) / steps;
    const points: { price: number; pl: number }[] = [];
    for (let i = 0; i <= steps; i++) {
      const p = minP + i * dx;
      const pl = payoffFn(spot, strike, premium, p) * quantity * contractMultiplier;
      points.push({ price: p, pl });
    }
    return points;
  }, [spot, strike, premium, breakEven, type, quantity, contractMultiplier]);

  const yDomain = useMemo(() => {
    if (chartData.length === 0) return [0, 0];
    const pls = chartData.map((d) => d.pl);
    const minPl = Math.min(...pls);
    const maxPl = Math.max(...pls);
    const padding = Math.max(2000, (maxPl - minPl) * 0.1);
    return [minPl - padding, maxPl + padding];
  }, [chartData]);

  const maxCost = premium * quantity * contractMultiplier;
  const currentPl = payoffFn(spot, strike, premium, spot) * quantity * contractMultiplier;

  const expiryLabel = expiry.length >= 6 ? `${expiry.slice(0, 2)} ${expiry.slice(2, 5)} ${expiry.slice(5)}` : expiry;

  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        {/* Header: Contract + Close */}
        <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30 flex-wrap gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-lg">
              {underlying} ${strike.toLocaleString()} {type === 'call' ? 'Call' : 'Put'} {expiryLabel}
            </h3>
            {(contract.callPremium != null || contract.putPremium != null) && (
              <div className="flex rounded-md border bg-background p-0.5">
                <button
                  type="button"
                  onClick={() => setSide('call')}
                  className={`px-2 py-0.5 text-xs rounded ${side === 'call' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'}`}
                >
                  Call
                </button>
                <button
                  type="button"
                  onClick={() => setSide('put')}
                  className={`px-2 py-0.5 text-xs rounded ${side === 'put' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground'}`}
                >
                  Put
                </button>
              </div>
            )}
          </div>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Order entry */}
        <div className="px-4 py-3 border-b flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <Checkbox checked={reduce} onCheckedChange={(v) => setReduce(!!v)} />
            Reduce
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <Checkbox checked={post} onCheckedChange={(v) => setPost(!!v)} />
            Post
          </label>
          <Select value={timeInForce} onValueChange={setTimeInForce}>
            <SelectTrigger className="w-[100px] h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="GTC">GTC</SelectItem>
              <SelectItem value="IOC">IOC</SelectItem>
              <SelectItem value="FOK">FOK</SelectItem>
            </SelectContent>
          </Select>
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-muted-foreground">Qty</span>
            <Select value={String(quantity)} onValueChange={(v) => setQuantity(Number(v))}>
              <SelectTrigger className="w-[70px] h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[1, 2, 5, 10, 25, 50, 100].map((n) => (
                  <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="px-4 py-2 border-b">
          <Button className="w-full gap-2" variant="secondary">
            <Link2 className="h-4 w-4" />
            Connect a Wallet
          </Button>
        </div>

        {/* Cost / Margin */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 px-4 py-3 border-b text-sm">
          <div>
            <p className="text-xs text-muted-foreground">Max Cost</p>
            <p className="font-mono font-medium">{formatCurrency(maxCost)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Margin Required</p>
            <p className="font-mono font-medium">$0.00</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Buying Power</p>
            <p className="font-mono font-medium">$0.00</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Est. Fee</p>
            <p className="font-mono font-medium">$0.00</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Est. Rewards</p>
            <p className="font-mono font-medium">0 DRV</p>
          </div>
        </div>

        {/* Tabs: Payoff | Greeks | Trades | Book */}
        <Tabs defaultValue="payoff" className="w-full">
          <div className="px-4 border-b">
            <TabsList className="h-9 bg-transparent p-0 gap-1">
              <TabsTrigger value="payoff" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none">Payoff</TabsTrigger>
              <TabsTrigger value="greeks" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none">Greeks</TabsTrigger>
              <TabsTrigger value="trades" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none">Trades</TabsTrigger>
              <TabsTrigger value="book" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none">Book</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="payoff" className="m-0 p-4">
            <div className="flex flex-wrap gap-4 mb-4">
              <div>
                <p className="text-xs text-muted-foreground">Max Loss</p>
                <p className="text-lg font-semibold text-red-600 dark:text-red-400">{formatCurrency(maxLoss)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Break Even</p>
                <p className="text-lg font-semibold">{formatCurrency(breakEven)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Max Profit</p>
                <p className="text-lg font-semibold text-green-600 dark:text-green-400">
                  {type === 'call' ? '∞' : formatCurrency(maxProfit)}
                </p>
              </div>
            </div>
            <div className="h-[280px] w-full" role="img" aria-label="Profit and loss at expiry chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  key={`payoff-${strike}-${type}-${quantity}-${premium}-${spot}`}
                  data={chartData}
                  margin={{ top: 8, right: 8, left: 8, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis
                    dataKey="price"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(v) => (v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v}`)}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis
                    type="number"
                    domain={yDomain}
                    tickFormatter={(v) => (Math.abs(v) >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v}`)}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(value: number) => [formatCurrency(value), 'P/L']}
                    labelFormatter={(label) => `Price: ${formatCurrency(Number(label))}`}
                  />
                  <ReferenceLine x={spot} stroke="var(--primary)" strokeDasharray="4 4" />
                  <Line
                    type="monotone"
                    dataKey="pl"
                    stroke="var(--chart-1)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={true}
                    name="P/L at expiry"
                  />
                  <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeDasharray="2 2" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Current {underlying}: {formatCurrency(spot)} · P/L at expiry: {formatCurrency(currentPl)}
            </p>
          </TabsContent>

          <TabsContent value="greeks" className="m-0 p-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Delta</p>
                <p className="font-mono font-semibold">—</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Gamma</p>
                <p className="font-mono font-semibold">—</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Theta</p>
                <p className="font-mono font-semibold">—</p>
              </div>
              <div className="rounded-lg bg-muted/50 p-3 text-center">
                <p className="text-xs text-muted-foreground">Vega</p>
                <p className="font-mono font-semibold">—</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-3">Connect or use backend analyze for live Greeks.</p>
          </TabsContent>

          <TabsContent value="trades" className="m-0 p-4 min-h-[120px] text-sm text-muted-foreground">
            No trades. Connect a wallet to view history.
          </TabsContent>

          <TabsContent value="book" className="m-0 p-4 min-h-[120px] text-sm text-muted-foreground">
            Order book for this contract. Connect to stream.
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
