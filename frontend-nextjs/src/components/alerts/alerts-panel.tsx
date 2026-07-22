'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { formatCurrency } from '@/lib/utils';
import { cn } from '@/lib/utils';
import { Bell, BellRing, Plus, Trash2, X, CheckCircle2, ArrowUpCircle, ArrowDownCircle, Sparkles } from 'lucide-react';
import { useMarketWS } from '@/lib/hooks/use-market-ws';
import { usePriceAlerts } from '@/lib/hooks/use-price-alerts';
import { useToast } from '@/components/ui/toast';

const WATCHED_SYMBOLS = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA', 'BTC-USD', 'ETH-USD', 'GLD', 'SPY'];

const DIRECTION_LABEL: Record<'above' | 'below', string> = {
  above: 'بالاتر از',
  below: 'پایین‌تر از',
};

export function AlertsPanel() {
  const { ticks } = useMarketWS(WATCHED_SYMBOLS);
  const prices = Object.fromEntries(Object.entries(ticks).map(([s, t]) => [s, t.price]));
  const { toast } = useToast();

  const onTrigger = useCallback(
    (alert: { symbol: string; targetPrice: number; direction: 'above' | 'below'; note?: string }, price: number) => {
      toast({
        title: `🔔 هشدار: ${alert.symbol}`,
        description: `قیمت ${alert.direction === 'above' ? 'به بالای' : 'به پایین'} ${formatCurrency(alert.targetPrice)} رسید — اکنون ${formatCurrency(price)}${alert.note ? ` · ${alert.note}` : ''}`,
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
  const watchedWithPrice = WATCHED_SYMBOLS.filter((s) => prices[s] !== undefined);

  return (
    <div className="space-y-5" dir="rtl">
      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="persian-card rounded-2xl p-4 border border-border/40">
          <div className="flex items-center gap-2 mb-1.5">
            <Bell className="h-4 w-4 text-green-400" />
            <span className="text-[11px] text-muted-foreground">هشدارهای فعال</span>
          </div>
          <div className="text-xl font-black text-green-400">{active.length.toLocaleString('fa-IR')}</div>
        </div>
        <div className="persian-card rounded-2xl p-4 border border-border/40">
          <div className="flex items-center gap-2 mb-1.5">
            <CheckCircle2 className="h-4 w-4 text-amber-400" />
            <span className="text-[11px] text-muted-foreground">فعال‌شده</span>
          </div>
          <div className="text-xl font-black text-amber-400">{triggered.length.toLocaleString('fa-IR')}</div>
        </div>
        <div className="persian-card rounded-2xl p-4 border border-border/40">
          <div className="flex items-center gap-2 mb-1.5">
            <Sparkles className="h-4 w-4 text-blue-400" />
            <span className="text-[11px] text-muted-foreground">نماد تحت نظر</span>
          </div>
          <div className="text-xl font-black text-blue-400">{WATCHED_SYMBOLS.length.toLocaleString('fa-IR')}</div>
        </div>
      </div>

      {/* Main card */}
      <div className="persian-card rounded-2xl border border-border/40 p-5">
        <div className="flex items-center justify-between gap-3 pb-4 border-b border-border/40">
          <h3 className="flex items-center gap-2 text-sm font-bold">
            <BellRing className="h-4 w-4 text-green-400" />
            هشدارهای قیمت
          </h3>
          <div className="flex gap-2">
            {triggered.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clearTriggered} className="text-xs">
                پاک‌کردن فعال‌شده‌ها
              </Button>
            )}
            <Button
              size="sm"
              className={cn('gap-1.5', !showForm && 'bg-green-600 hover:bg-green-700 text-white')}
              variant={showForm ? 'outline' : 'default'}
              onClick={() => setShowForm((v) => !v)}
            >
              {showForm ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
              {showForm ? 'انصراف' : 'هشدار جدید'}
            </Button>
          </div>
        </div>

        {/* New Alert Form */}
        {showForm && (
          <div className="mt-4 mb-2 p-4 rounded-xl bg-muted/30 border border-border/40 space-y-3">
            <h4 className="font-bold text-sm">ساخت هشدار قیمت</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <Label className="text-xs">نماد</Label>
                <Input
                  placeholder="AAPL"
                  value={form.symbol}
                  onChange={(e) => setForm((f) => ({ ...f, symbol: e.target.value.toUpperCase() }))}
                  className="h-9 text-sm mt-1"
                />
              </div>
              <div>
                <Label className="text-xs">جهت</Label>
                <div className="flex gap-2 mt-1">
                  <Button
                    type="button"
                    size="sm"
                    variant={form.direction === 'above' ? 'default' : 'outline'}
                    className={cn('flex-1 h-9 text-xs gap-1', form.direction === 'above' && 'bg-green-600 hover:bg-green-700 text-white')}
                    onClick={() => setForm((f) => ({ ...f, direction: 'above' }))}
                  >
                    <ArrowUpCircle className="h-3.5 w-3.5" /> بالاتر
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={form.direction === 'below' ? 'default' : 'outline'}
                    className={cn('flex-1 h-9 text-xs gap-1', form.direction === 'below' && 'bg-red-600 hover:bg-red-700 text-white')}
                    onClick={() => setForm((f) => ({ ...f, direction: 'below' }))}
                  >
                    <ArrowDownCircle className="h-3.5 w-3.5" /> پایین‌تر
                  </Button>
                </div>
              </div>
              <div>
                <Label className="text-xs">قیمت هدف (دلار)</Label>
                <Input
                  type="number"
                  placeholder="150.00"
                  value={form.targetPrice}
                  onChange={(e) => setForm((f) => ({ ...f, targetPrice: e.target.value }))}
                  className="h-9 text-sm mt-1"
                />
              </div>
              <div>
                <Label className="text-xs">یادداشت (اختیاری)</Label>
                <Input
                  placeholder="دلیل..."
                  value={form.note}
                  onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
                  className="h-9 text-sm mt-1"
                />
              </div>
            </div>
            {form.symbol && prices[form.symbol] && (
              <p className="text-xs text-muted-foreground">
                قیمت فعلی {form.symbol}: <span className="font-semibold text-foreground">{formatCurrency(prices[form.symbol])}</span>
              </p>
            )}
            <Button size="sm" className="bg-green-600 hover:bg-green-700 text-white" onClick={handleAdd}>
              ثبت هشدار
            </Button>
          </div>
        )}

        {/* Alerts list */}
        <div className="mt-4">
          {active.length === 0 && triggered.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 py-10 text-muted-foreground text-sm">
              <Bell className="h-8 w-8 opacity-30" />
              <p>هنوز هشداری ثبت نشده — یک هشدار جدید بساز.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {active.map((alert) => (
                <div
                  key={alert.id}
                  className={cn(
                    'flex items-center justify-between gap-3 p-3 rounded-xl border text-sm transition-colors',
                    alert.direction === 'above'
                      ? 'border-green-500/20 bg-green-500/[0.04] hover:bg-green-500/[0.08]'
                      : 'border-red-500/20 bg-red-500/[0.04] hover:bg-red-500/[0.08]'
                  )}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    {alert.direction === 'above' ? (
                      <ArrowUpCircle className="h-4 w-4 text-green-400 flex-shrink-0" />
                    ) : (
                      <ArrowDownCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
                    )}
                    <span className="font-bold flex-shrink-0">{alert.symbol}</span>
                    <span
                      className={cn(
                        'text-xs px-2 py-0.5 rounded-full font-semibold flex-shrink-0',
                        alert.direction === 'above' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
                      )}
                    >
                      {DIRECTION_LABEL[alert.direction]} {formatCurrency(alert.targetPrice)}
                    </span>
                    {alert.note && <span className="text-muted-foreground text-xs italic truncate">{alert.note}</span>}
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {prices[alert.symbol] && (
                      <span className="text-xs text-muted-foreground font-mono">
                        اکنون: {formatCurrency(prices[alert.symbol])}
                      </span>
                    )}
                    <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => removeAlert(alert.id)}>
                      <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                    </Button>
                  </div>
                </div>
              ))}
              {triggered.map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between gap-3 p-3 rounded-xl border border-border/30 bg-muted/20 text-sm opacity-70"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                    <span className="font-bold line-through flex-shrink-0">{alert.symbol}</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground flex-shrink-0">
                      فعال‌شده
                    </span>
                    <span className="text-xs text-muted-foreground flex-shrink-0">
                      {DIRECTION_LABEL[alert.direction]} {formatCurrency(alert.targetPrice)}
                    </span>
                  </div>
                  <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => removeAlert(alert.id)}>
                    <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Watched prices */}
      {watchedWithPrice.length > 0 && (
        <div className="persian-card rounded-2xl border border-border/40 p-5">
          <h3 className="flex items-center gap-2 text-sm font-bold mb-4">
            <Sparkles className="h-4 w-4 text-blue-400" />
            نمادهای تحت نظر
          </h3>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-2.5">
            {watchedWithPrice.map((sym) => (
              <div key={sym} className="text-center p-2.5 rounded-xl border border-border/40 bg-muted/10">
                <p className="text-[11px] text-muted-foreground mb-0.5">{sym}</p>
                <p className="text-sm font-mono font-bold">{formatCurrency(prices[sym])}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
