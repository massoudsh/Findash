'use client';

import { useState, useCallback, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

export type AssetType =
  | 'gold'
  | 'silver'
  | 'currency'
  | 'crypto'
  | 'stock'
  | 'bond'
  | 'real_estate'
  | 'cash';

export interface IranPortfolioAsset {
  id: string;
  type: AssetType;
  name: string;
  code: string;
  quantity: number;
  unitPrice: number;
  totalValue: number;
  currency: 'IRT' | 'USD';
  side: 'buy' | 'sell';
  timestamp: string;
  note?: string;
}

// ─── Presets ─────────────────────────────────────────────────────────────────

export const ASSET_PRESETS: Array<{
  type: AssetType;
  name: string;
  code: string;
  color: string;
  icon: string;
}> = [
  { type: 'gold',         name: 'طلا ۱۸ عیار',   code: 'XAU18',     color: '#f59e0b', icon: '🥇' },
  { type: 'gold',         name: 'طلا ۲۴ عیار',   code: 'XAU24',     color: '#d97706', icon: '🥇' },
  { type: 'gold',         name: 'سکه تمام بهار', code: 'COIN_FULL', color: '#eab308', icon: '🪙' },
  { type: 'gold',         name: 'نیم سکه',        code: 'COIN_HALF', color: '#ca8a04', icon: '🪙' },
  { type: 'silver',       name: 'نقره',            code: 'XAG',       color: '#94a3b8', icon: '🥈' },
  { type: 'currency',     name: 'دلار آمریکا',    code: 'USD',       color: '#10b981', icon: '💵' },
  { type: 'currency',     name: 'یورو',            code: 'EUR',       color: '#06b6d4', icon: '€' },
  { type: 'currency',     name: 'درهم',            code: 'AED',       color: '#0ea5e9', icon: '🇦🇪' },
  { type: 'crypto',       name: 'بیت‌کوین',        code: 'BTC',       color: '#f97316', icon: '₿' },
  { type: 'crypto',       name: 'اتریوم',          code: 'ETH',       color: '#6366f1', icon: 'Ξ' },
  { type: 'crypto',       name: 'تتر',             code: 'USDT',      color: '#34d399', icon: '₮' },
  { type: 'stock',        name: 'سهام بورس',       code: 'STOCK',     color: '#3b82f6', icon: '📈' },
  { type: 'bond',         name: 'اوراق بدهی',      code: 'BOND',      color: '#8b5cf6', icon: '📋' },
  { type: 'real_estate',  name: 'مسکن',            code: 'RE_TEHRAN', color: '#ef4444', icon: '🏠' },
  { type: 'cash',         name: 'نقدی / صندوق',    code: 'CASH',      color: '#6b7280', icon: '💰' },
];

// ─── Category labels (for grouping) ──────────────────────────────────────────

export const CATEGORY_LABEL: Record<AssetType, string> = {
  gold:         'طلا و سکه',
  silver:       'نقره',
  currency:     'ارز',
  crypto:       'دیجیتال',
  stock:        'سهام',
  bond:         'اوراق',
  real_estate:  'مسکن',
  cash:         'نقدی',
};

// ─── localStorage helpers ─────────────────────────────────────────────────────

const STORAGE_KEY = 'iran_portfolio_v1';

export function loadIranAssets(): IranPortfolioAsset[] {
  if (typeof window === 'undefined') return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

export function saveIranAssets(assets: IranPortfolioAsset[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(assets));
}

// ─── Toman formatter ──────────────────────────────────────────────────────────

export function formatToman(n: number): string {
  return new Intl.NumberFormat('fa-IR').format(Math.round(n));
}

// ─── Modal Component ──────────────────────────────────────────────────────────

interface AddAssetModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAdd: (asset: IranPortfolioAsset) => void;
}

const EMPTY_FORM = {
  presetName: '',
  code: '',
  quantity: '',
  unitPrice: '',
  totalValue: '',
  currency: 'IRT' as 'IRT' | 'USD',
  side: 'buy' as 'buy' | 'sell',
  note: '',
};

export function AddAssetModal({ open, onOpenChange, onAdd }: AddAssetModalProps) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [lastChanged, setLastChanged] = useState<'unit' | 'total' | null>(null);

  const selectedPreset = ASSET_PRESETS.find((p) => p.name === form.presetName) ?? null;

  // Auto-calculate total when unit price or quantity changes
  useEffect(() => {
    if (lastChanged !== 'unit') return;
    const qty = parseFloat(form.quantity);
    const unit = parseFloat(form.unitPrice);
    if (!isNaN(qty) && !isNaN(unit) && qty > 0 && unit > 0) {
      setForm((f) => ({ ...f, totalValue: String(qty * unit) }));
    }
  }, [form.quantity, form.unitPrice, lastChanged]);

  // Auto-calculate unit price when total or quantity changes
  useEffect(() => {
    if (lastChanged !== 'total') return;
    const qty = parseFloat(form.quantity);
    const total = parseFloat(form.totalValue);
    if (!isNaN(qty) && !isNaN(total) && qty > 0 && total > 0) {
      setForm((f) => ({ ...f, unitPrice: String(total / qty) }));
    }
  }, [form.quantity, form.totalValue, lastChanged]);

  const handlePresetChange = (name: string) => {
    const preset = ASSET_PRESETS.find((p) => p.name === name);
    setForm((f) => ({ ...f, presetName: name, code: preset?.code ?? '' }));
  };

  const handleSubmit = useCallback(() => {
    if (!selectedPreset || !form.quantity) return;
    const qty = parseFloat(form.quantity);
    const unit = parseFloat(form.unitPrice) || 0;
    const total = parseFloat(form.totalValue) || unit * qty;

    const asset: IranPortfolioAsset = {
      id: crypto.randomUUID(),
      type: selectedPreset.type,
      name: selectedPreset.name,
      code: form.code || selectedPreset.code,
      quantity: qty,
      unitPrice: unit,
      totalValue: total,
      currency: form.currency,
      side: form.side,
      timestamp: new Date().toISOString(),
      note: form.note || undefined,
    };
    onAdd(asset);
    setForm(EMPTY_FORM);
    setLastChanged(null);
    onOpenChange(false);
  }, [selectedPreset, form, onAdd, onOpenChange]);

  const isValid = !!selectedPreset && !!form.quantity && (!!form.totalValue || !!form.unitPrice);
  const currencyLabel = form.currency === 'IRT' ? 'تومان' : 'USD';
  const totalNum = parseFloat(form.totalValue);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md" dir="rtl">
        <DialogHeader>
          <DialogTitle className="text-right text-base font-semibold">
            افزودن دارایی به پرتفوی
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 pt-1">
          {/* Asset Type */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">انتخاب دارایی</Label>
            <Select value={form.presetName} onValueChange={handlePresetChange}>
              <SelectTrigger>
                <SelectValue placeholder="نوع دارایی را انتخاب کنید..." />
              </SelectTrigger>
              <SelectContent>
                {ASSET_PRESETS.map((p) => (
                  <SelectItem key={`${p.type}-${p.code}`} value={p.name}>
                    <span className="flex items-center gap-2">
                      <span>{p.icon}</span>
                      <span>{p.name}</span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Asset Code */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">کد دارایی</Label>
            <Input
              placeholder="GOLD"
              value={form.code}
              onChange={(e) => setForm((f) => ({ ...f, code: e.target.value.toUpperCase() }))}
              className="text-left ltr"
              dir="ltr"
            />
          </div>

          {/* Transaction Type */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">نوع تراکنش</Label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                size="sm"
                variant={form.side === 'buy' ? 'default' : 'outline'}
                className={cn(form.side === 'buy' && 'bg-green-600 hover:bg-green-700 text-white')}
                onClick={() => setForm((f) => ({ ...f, side: 'buy' }))}
              >
                افزودن (خرید)
              </Button>
              <Button
                type="button"
                size="sm"
                variant={form.side === 'sell' ? 'default' : 'outline'}
                className={cn(form.side === 'sell' && 'bg-red-600 hover:bg-red-700 text-white')}
                onClick={() => setForm((f) => ({ ...f, side: 'sell' }))}
              >
                برداشت (فروش)
              </Button>
            </div>
          </div>

          {/* Quantity */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">مقدار / تعداد</Label>
            <Input
              type="number"
              placeholder="۱"
              value={form.quantity}
              onChange={(e) => setForm((f) => ({ ...f, quantity: e.target.value }))}
              min="0"
              dir="ltr"
            />
          </div>

          {/* Currency */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">واحد پول</Label>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                size="sm"
                variant={form.currency === 'IRT' ? 'default' : 'outline'}
                onClick={() => setForm((f) => ({ ...f, currency: 'IRT' }))}
              >
                تومان (IRT)
              </Button>
              <Button
                type="button"
                size="sm"
                variant={form.currency === 'USD' ? 'default' : 'outline'}
                onClick={() => setForm((f) => ({ ...f, currency: 'USD' }))}
              >
                دلار (USD)
              </Button>
            </div>
          </div>

          {/* Unit Price + Total Value in a grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-muted-foreground">
                قیمت واحد ({currencyLabel})
              </Label>
              <Input
                type="number"
                placeholder="0"
                value={form.unitPrice}
                onChange={(e) => {
                  setForm((f) => ({ ...f, unitPrice: e.target.value }));
                  setLastChanged('unit');
                }}
                min="0"
                dir="ltr"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-muted-foreground">
                ارزش کل ({currencyLabel})
              </Label>
              <Input
                type="number"
                placeholder="0"
                value={form.totalValue}
                onChange={(e) => {
                  setForm((f) => ({ ...f, totalValue: e.target.value }));
                  setLastChanged('total');
                }}
                min="0"
                dir="ltr"
              />
            </div>
          </div>

          {/* Toman display */}
          {form.currency === 'IRT' && !isNaN(totalNum) && totalNum > 0 && (
            <p className="text-xs text-muted-foreground text-right -mt-1">
              {formatToman(totalNum)} تومان
            </p>
          )}

          {/* Note */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">یادداشت (اختیاری)</Label>
            <Input
              placeholder="توضیحات..."
              value={form.note}
              onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
            />
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-1">
            <Button
              className={cn(
                'flex-1',
                form.side === 'buy'
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              )}
              onClick={handleSubmit}
              disabled={!isValid}
            >
              {form.side === 'buy' ? 'افزودن دارایی' : 'ثبت فروش'}
            </Button>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              انصراف
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
