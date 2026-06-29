'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { formatCurrency } from '@/lib/utils';
import { cn } from '@/lib/utils';
import { Plus, Trash2, TrendingUp, TrendingDown, Wallet, X } from 'lucide-react';

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: string;
  note?: string;
}

interface Position {
  symbol: string;
  quantity: number;
  avgCost: number;
  totalCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPct: number;
}

const STORAGE_KEY = 'octopus_trades_v1';

function loadTrades(): Trade[] {
  if (typeof window === 'undefined') return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveTrades(trades: Trade[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trades));
}

function calcPositions(trades: Trade[], prices: Record<string, number>): Position[] {
  const map: Record<string, { qty: number; cost: number }> = {};
  for (const t of trades) {
    if (!map[t.symbol]) map[t.symbol] = { qty: 0, cost: 0 };
    if (t.side === 'buy') {
      map[t.symbol].cost += t.price * t.quantity;
      map[t.symbol].qty += t.quantity;
    } else {
      map[t.symbol].cost -= t.price * t.quantity;
      map[t.symbol].qty -= t.quantity;
    }
  }
  return Object.entries(map)
    .filter(([, v]) => v.qty > 0)
    .map(([symbol, v]) => {
      const avgCost = v.cost / v.qty;
      const currentPrice = prices[symbol] ?? avgCost;
      const marketValue = currentPrice * v.qty;
      const unrealizedPnl = marketValue - v.cost;
      const unrealizedPnlPct = v.cost > 0 ? (unrealizedPnl / v.cost) * 100 : 0;
      return { symbol, quantity: v.qty, avgCost, totalCost: v.cost, currentPrice, marketValue, unrealizedPnl, unrealizedPnlPct };
    });
}

interface TradeTrackerProps {
  /** optional current prices from realtime hook */
  prices?: Record<string, number>;
}

export function TradeTracker({ prices = {} }: TradeTrackerProps) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ symbol: '', side: 'buy' as 'buy' | 'sell', quantity: '', price: '', note: '' });

  useEffect(() => {
    setTrades(loadTrades());
  }, []);

  const addTrade = useCallback(() => {
    if (!form.symbol || !form.quantity || !form.price) return;
    const t: Trade = {
      id: crypto.randomUUID(),
      symbol: form.symbol.toUpperCase(),
      side: form.side,
      quantity: parseFloat(form.quantity),
      price: parseFloat(form.price),
      timestamp: new Date().toISOString(),
      note: form.note || undefined,
    };
    const updated = [t, ...trades];
    setTrades(updated);
    saveTrades(updated);
    setForm({ symbol: '', side: 'buy', quantity: '', price: '', note: '' });
    setShowForm(false);
  }, [form, trades]);

  const removeTrade = useCallback((id: string) => {
    const updated = trades.filter((t) => t.id !== id);
    setTrades(updated);
    saveTrades(updated);
  }, [trades]);

  const positions = calcPositions(trades, prices);
  const totalPnl = positions.reduce((s, p) => s + p.unrealizedPnl, 0);
  const totalValue = positions.reduce((s, p) => s + p.marketValue, 0);

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Portfolio Value</p>
            <p className="text-lg font-bold">{formatCurrency(totalValue)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Unrealized P&L</p>
            <p className={cn('text-lg font-bold', totalPnl >= 0 ? 'text-green-600' : 'text-red-500')}>
              {totalPnl >= 0 ? '+' : ''}{formatCurrency(totalPnl)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Positions</p>
            <p className="text-lg font-bold">{positions.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Total Trades</p>
            <p className="text-lg font-bold">{trades.length}</p>
          </CardContent>
        </Card>
      </div>

      {/* Positions Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Wallet className="h-4 w-4" /> Positions
          </CardTitle>
          <Button size="sm" onClick={() => setShowForm((v) => !v)}>
            {showForm ? <X className="h-4 w-4 mr-1" /> : <Plus className="h-4 w-4 mr-1" />}
            {showForm ? 'Cancel' : 'Log Trade'}
          </Button>
        </CardHeader>
        <CardContent>
          {/* Trade Form */}
          {showForm && (
            <div className="mb-4 p-4 border rounded-lg bg-muted/30 space-y-3">
              <h4 className="font-semibold text-sm">New Trade</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <Label className="text-xs">Symbol</Label>
                  <Input
                    placeholder="AAPL"
                    value={form.symbol}
                    onChange={(e) => setForm((f) => ({ ...f, symbol: e.target.value.toUpperCase() }))}
                    className="h-8 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs">Side</Label>
                  <div className="flex gap-2 mt-1">
                    <Button
                      size="sm"
                      variant={form.side === 'buy' ? 'default' : 'outline'}
                      className={cn('flex-1 h-8 text-xs', form.side === 'buy' && 'bg-green-600 hover:bg-green-700')}
                      onClick={() => setForm((f) => ({ ...f, side: 'buy' }))}
                    >
                      BUY
                    </Button>
                    <Button
                      size="sm"
                      variant={form.side === 'sell' ? 'default' : 'outline'}
                      className={cn('flex-1 h-8 text-xs', form.side === 'sell' && 'bg-red-600 hover:bg-red-700')}
                      onClick={() => setForm((f) => ({ ...f, side: 'sell' }))}
                    >
                      SELL
                    </Button>
                  </div>
                </div>
                <div>
                  <Label className="text-xs">Quantity</Label>
                  <Input
                    type="number"
                    placeholder="10"
                    value={form.quantity}
                    onChange={(e) => setForm((f) => ({ ...f, quantity: e.target.value }))}
                    className="h-8 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs">Price (USD)</Label>
                  <Input
                    type="number"
                    placeholder="150.00"
                    value={form.price}
                    onChange={(e) => setForm((f) => ({ ...f, price: e.target.value }))}
                    className="h-8 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs">Note (optional)</Label>
                  <Input
                    placeholder="reason..."
                    value={form.note}
                    onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
                    className="h-8 text-sm"
                  />
                </div>
              </div>
              <Button size="sm" onClick={addTrade} className="mt-1">Save Trade</Button>
            </div>
          )}

          {positions.length === 0 ? (
            <p className="text-center text-muted-foreground py-6 text-sm">No open positions. Log your first trade above.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground text-xs">
                    <th className="text-left py-2">Symbol</th>
                    <th className="text-right py-2">Qty</th>
                    <th className="text-right py-2">Avg Cost</th>
                    <th className="text-right py-2">Current</th>
                    <th className="text-right py-2">Market Value</th>
                    <th className="text-right py-2">P&L</th>
                    <th className="text-right py-2">P&L %</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((pos) => (
                    <tr key={pos.symbol} className="border-b">
                      <td className="py-2 font-semibold">{pos.symbol}</td>
                      <td className="text-right py-2">{pos.quantity}</td>
                      <td className="text-right py-2 font-mono">{formatCurrency(pos.avgCost)}</td>
                      <td className="text-right py-2 font-mono">{formatCurrency(pos.currentPrice)}</td>
                      <td className="text-right py-2 font-mono">{formatCurrency(pos.marketValue)}</td>
                      <td className={cn('text-right py-2 font-mono', pos.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-500')}>
                        <span className="inline-flex items-center gap-0.5">
                          {pos.unrealizedPnl >= 0
                            ? <TrendingUp className="h-3 w-3" />
                            : <TrendingDown className="h-3 w-3" />}
                          {pos.unrealizedPnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl)}
                        </span>
                      </td>
                      <td className={cn('text-right py-2 font-mono', pos.unrealizedPnlPct >= 0 ? 'text-green-600' : 'text-red-500')}>
                        {pos.unrealizedPnlPct >= 0 ? '+' : ''}{pos.unrealizedPnlPct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trade History */}
      {trades.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">Trade History</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {trades.map((t) => (
                <div key={t.id} className="flex items-center justify-between p-2 rounded-lg border text-sm">
                  <div className="flex items-center gap-3">
                    <Badge variant={t.side === 'buy' ? 'default' : 'destructive'} className="uppercase text-xs">
                      {t.side}
                    </Badge>
                    <span className="font-semibold">{t.symbol}</span>
                    <span className="text-muted-foreground">
                      {t.quantity} @ {formatCurrency(t.price)}
                    </span>
                    {t.note && <span className="text-muted-foreground italic text-xs">{t.note}</span>}
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-muted-foreground">{new Date(t.timestamp).toLocaleDateString()}</span>
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => removeTrade(t.id)}>
                      <Trash2 className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
