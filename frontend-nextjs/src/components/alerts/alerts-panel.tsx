'use client';

import { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { formatCurrency } from '@/lib/utils';
import { cn } from '@/lib/utils';
import { Bell, BellOff, Plus, Trash2, X, CheckCircle, ArrowUp, ArrowDown } from 'lucide-react';
import { useMarketWS } from '@/lib/hooks/use-market-ws';
import { usePriceAlerts } from '@/lib/hooks/use-price-alerts';
import { useToast } from '@/components/ui/toast';

const WATCHED_SYMBOLS = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA', 'BTC-USD', 'ETH-USD', 'GLD', 'SPY'];

export function AlertsPanel() {
  const { ticks } = useMarketWS(WATCHED_SYMBOLS);
  const prices = Object.fromEntries(Object.entries(ticks).map(([s, t]) => [s, t.price]));
  const { toast } = useToast();

  const onTrigger = useCallback(
    (alert: { symbol: string; targetPrice: number; direction: 'above' | 'below'; note?: string }, price: number) => {
      toast({
        title: `🔔 Alert: ${alert.symbol}`,
        description: `Price ${alert.direction === 'above' ? 'reached ≥' : 'dropped ≤'} ${formatCurrency(alert.targetPrice)} — now at ${formatCurrency(price)}${alert.note ? ` · ${alert.note}` : ''}`,
        duration: 8000,
      });
    },
    [toast]
  );

  const { alerts, addAlert, removeAlert, clearTriggered } = usePriceAlerts(prices, onTrigger);

  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ symbol: 'AAPL', direction: 'above' as 'above' | 'below', targetPrice: '', note: '' });

  const handleAdd = () => {
    if (!form.symbol || !form.targetPrice) return;
    addAlert({ symbol: form.symbol.toUpperCase(), direction: form.direction, targetPrice: parseFloat(form.targetPrice), note: form.note || undefined });
    setForm({ symbol: 'AAPL', direction: 'above', targetPrice: '', note: '' });
    setShowForm(false);
  };

  const active = alerts.filter((a) => !a.triggered);
  const triggered = alerts.filter((a) => a.triggered);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Bell className="h-4 w-4" />
            Price Alerts
            {active.length > 0 && (
              <Badge variant="default" className="ml-1 text-xs">{active.length} active</Badge>
            )}
          </CardTitle>
          <div className="flex gap-2">
            {triggered.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clearTriggered} className="text-xs">
                Clear triggered
              </Button>
            )}
            <Button size="sm" onClick={() => setShowForm((v) => !v)}>
              {showForm ? <X className="h-4 w-4 mr-1" /> : <Plus className="h-4 w-4 mr-1" />}
              {showForm ? 'Cancel' : 'New Alert'}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* New Alert Form */}
          {showForm && (
            <div className="mb-4 p-4 border rounded-lg bg-muted/30 space-y-3">
              <h4 className="font-semibold text-sm">Create Price Alert</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
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
                  <Label className="text-xs">Direction</Label>
                  <div className="flex gap-2 mt-1">
                    <Button
                      size="sm"
                      variant={form.direction === 'above' ? 'default' : 'outline'}
                      className="flex-1 h-8 text-xs"
                      onClick={() => setForm((f) => ({ ...f, direction: 'above' }))}
                    >
                      <ArrowUp className="h-3 w-3 mr-0.5" /> Above
                    </Button>
                    <Button
                      size="sm"
                      variant={form.direction === 'below' ? 'default' : 'outline'}
                      className="flex-1 h-8 text-xs"
                      onClick={() => setForm((f) => ({ ...f, direction: 'below' }))}
                    >
                      <ArrowDown className="h-3 w-3 mr-0.5" /> Below
                    </Button>
                  </div>
                </div>
                <div>
                  <Label className="text-xs">Target Price (USD)</Label>
                  <Input
                    type="number"
                    placeholder="150.00"
                    value={form.targetPrice}
                    onChange={(e) => setForm((f) => ({ ...f, targetPrice: e.target.value }))}
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
              {form.symbol && prices[form.symbol] && (
                <p className="text-xs text-muted-foreground">
                  Current {form.symbol}: <span className="font-semibold">{formatCurrency(prices[form.symbol])}</span>
                </p>
              )}
              <Button size="sm" onClick={handleAdd}>Set Alert</Button>
            </div>
          )}

          {/* Active Alerts */}
          {active.length === 0 && triggered.length === 0 ? (
            <p className="text-center text-muted-foreground py-6 text-sm">No alerts yet. Create one above.</p>
          ) : (
            <div className="space-y-2">
              {active.map((alert) => (
                <div key={alert.id} className="flex items-center justify-between p-3 rounded-lg border text-sm">
                  <div className="flex items-center gap-3">
                    <Bell className="h-4 w-4 text-yellow-500" />
                    <span className="font-semibold">{alert.symbol}</span>
                    <Badge variant="outline" className="text-xs">
                      {alert.direction === 'above' ? '≥' : '≤'} {formatCurrency(alert.targetPrice)}
                    </Badge>
                    {alert.note && <span className="text-muted-foreground text-xs italic">{alert.note}</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    {prices[alert.symbol] && (
                      <span className="text-xs text-muted-foreground">
                        now: {formatCurrency(prices[alert.symbol])}
                      </span>
                    )}
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => removeAlert(alert.id)}>
                      <Trash2 className="h-3 w-3 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              ))}
              {triggered.map((alert) => (
                <div key={alert.id} className={cn('flex items-center justify-between p-3 rounded-lg border text-sm opacity-60 bg-green-500/5')}>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="font-semibold line-through">{alert.symbol}</span>
                    <Badge variant="secondary" className="text-xs">triggered</Badge>
                    <span className="text-xs text-muted-foreground">
                      {alert.direction === 'above' ? '≥' : '≤'} {formatCurrency(alert.targetPrice)}
                    </span>
                  </div>
                  <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => removeAlert(alert.id)}>
                    <Trash2 className="h-3 w-3 text-muted-foreground" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Current Prices Quick View */}
      {Object.keys(prices).length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-sm flex items-center gap-2"><BellOff className="h-4 w-4" />Watched Prices</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
              {WATCHED_SYMBOLS.map((sym) => (
                prices[sym] ? (
                  <div key={sym} className="text-center p-2 border rounded-lg">
                    <p className="text-xs text-muted-foreground">{sym}</p>
                    <p className="text-sm font-mono font-semibold">{formatCurrency(prices[sym])}</p>
                  </div>
                ) : null
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
