"use client";

/**
 * PortfolioAssetsSection
 * نمایش دارایی‌های فیزیکی کاربر در صفحه پورتفولیو.
 * داده از POST /api/assets/portfolio خوانده می‌شود.
 */

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Plus, Trash2 } from "lucide-react";
import { formatToman, CATEGORY_ICONS, CATEGORY_LABELS } from "@/lib/assets";

interface PortfolioEntry {
  id: number;
  symbol: string;
  name_fa: string;
  category: string;
  quantity: number;
  buy_price: number;
  buy_date: string;
  current_price: number;
  current_value: number;
  profit_loss: number;
  profit_loss_percent: number;
}

interface AddAssetForm {
  symbol: string;
  quantity: string;
  buy_price: string;
  buy_date: string;
}

const EMPTY_FORM: AddAssetForm = {
  symbol: "",
  quantity: "",
  buy_price: "",
  buy_date: new Date().toISOString().split("T")[0],
};

// Sample symbols for dropdown
const AVAILABLE_SYMBOLS = [
  { symbol: "XAU18",       label: "طلای ۱۸ عیار",      category: "gold"        },
  { symbol: "COIN_FULL",   label: "سکه بهار آزادی",    category: "gold"        },
  { symbol: "COIN_HALF",   label: "نیم‌سکه",            category: "gold"        },
  { symbol: "COIN_QUARTER",label: "ربع‌سکه",            category: "gold"        },
  { symbol: "XAG",         label: "نقره (گرم)",         category: "silver"      },
  { symbol: "USD",         label: "دلار آمریکا",        category: "currency"    },
  { symbol: "EUR",         label: "یورو",               category: "currency"    },
  { symbol: "RE_TEHRAN",   label: "مسکن تهران",        category: "real_estate" },
  { symbol: "BTC",         label: "بیت‌کوین",           category: "crypto"      },
  { symbol: "ETH",         label: "اتریوم",             category: "crypto"      },
];

export function PortfolioAssetsSection() {
  const [entries, setEntries]       = useState<PortfolioEntry[]>([]);
  const [showForm, setShowForm]     = useState(false);
  const [form, setForm]             = useState<AddAssetForm>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const totalValue = entries.reduce((s, e) => s + e.current_value, 0);
  const totalPnL   = entries.reduce((s, e) => s + e.profit_loss, 0);
  const totalPnLPct = entries.length
    ? entries.reduce((s, e) => s + e.profit_loss_percent, 0) / entries.length
    : 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch("/api/assets/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol:    form.symbol,
          quantity:  parseFloat(form.quantity),
          buy_price: parseFloat(form.buy_price),
          buy_date:  form.buy_date,
        }),
      });

      if (!res.ok) throw new Error("خطا در ثبت دارایی");

      const newEntry: PortfolioEntry = await res.json();
      const meta = AVAILABLE_SYMBOLS.find((s) => s.symbol === form.symbol);

      setEntries((prev) => [
        ...prev,
        {
          ...newEntry,
          name_fa:  meta?.label    ?? form.symbol,
          category: meta?.category ?? "gold",
        },
      ]);
      setForm(EMPTY_FORM);
      setShowForm(false);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "خطای ناشناخته");
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = (id: number) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  };

  return (
    <section className="space-y-4" dir="rtl">

      {/* Section header */}
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-white/80">
          <span>🏦</span>
          <span>دارایی‌های فیزیکی</span>
        </h2>
        <Button
          size="sm"
          variant="outline"
          className="text-xs h-7 border-white/15 hover:border-white/30 bg-white/5"
          onClick={() => setShowForm((v) => !v)}
        >
          <Plus className="w-3 h-3 ml-1" />
          افزودن دارایی
        </Button>
      </div>

      {/* Summary bar */}
      {entries.length > 0 && (
        <div className="flex gap-4 px-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/5 text-sm">
          <div>
            <p className="text-[11px] text-muted-foreground">ارزش کل</p>
            <p className="font-semibold tabular-nums">{formatToman(totalValue)}</p>
          </div>
          <div className="h-8 w-px bg-white/10 self-center" />
          <div>
            <p className="text-[11px] text-muted-foreground">سود/زیان</p>
            <p className={`font-semibold tabular-nums ${totalPnL >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {totalPnL >= 0 ? "+" : ""}{formatToman(totalPnL)}
              <span className="text-xs mr-1 opacity-70">
                ({totalPnL >= 0 ? "+" : ""}{totalPnLPct.toFixed(1)}%)
              </span>
            </p>
          </div>
        </div>
      )}

      {/* Add form */}
      {showForm && (
        <Card className="bg-[#0f1117] border border-white/10">
          <CardContent className="p-4">
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                {/* Symbol */}
                <div className="col-span-2">
                  <label className="text-xs text-muted-foreground block mb-1">نوع دارایی</label>
                  <select
                    required
                    value={form.symbol}
                    onChange={(e) => setForm((f) => ({ ...f, symbol: e.target.value }))}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
                  >
                    <option value="" disabled>انتخاب کنید...</option>
                    {AVAILABLE_SYMBOLS.map((s) => (
                      <option key={s.symbol} value={s.symbol}>
                        {CATEGORY_ICONS[s.category as keyof typeof CATEGORY_ICONS]} {s.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Quantity */}
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">مقدار</label>
                  <input
                    type="number"
                    required
                    min="0"
                    step="any"
                    placeholder="مثال: 2.5"
                    value={form.quantity}
                    onChange={(e) => setForm((f) => ({ ...f, quantity: e.target.value }))}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/25"
                  />
                </div>

                {/* Buy price */}
                <div>
                  <label className="text-xs text-muted-foreground block mb-1">قیمت خرید (تومان)</label>
                  <input
                    type="number"
                    required
                    min="0"
                    step="any"
                    placeholder="مثال: 3500000"
                    value={form.buy_price}
                    onChange={(e) => setForm((f) => ({ ...f, buy_price: e.target.value }))}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-white/25"
                  />
                </div>

                {/* Buy date */}
                <div className="col-span-2">
                  <label className="text-xs text-muted-foreground block mb-1">تاریخ خرید</label>
                  <input
                    type="date"
                    required
                    value={form.buy_date}
                    onChange={(e) => setForm((f) => ({ ...f, buy_date: e.target.value }))}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
                  />
                </div>
              </div>

              {error && (
                <p className="text-xs text-red-400">{error}</p>
              )}

              <div className="flex gap-2 justify-end">
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="text-xs h-8 text-muted-foreground"
                  onClick={() => { setShowForm(false); setError(null); }}
                >
                  انصراف
                </Button>
                <Button
                  type="submit"
                  size="sm"
                  disabled={submitting}
                  className="text-xs h-8 bg-white/10 hover:bg-white/15 text-white border border-white/10"
                >
                  {submitting ? "در حال ثبت..." : "ثبت دارایی"}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Entries table */}
      {entries.length > 0 ? (
        <div className="space-y-2">
          {entries.map((entry) => {
            const isPositive = entry.profit_loss >= 0;
            return (
              <Card key={entry.id} className="bg-[#0f1117] border border-white/5 hover:border-white/10 transition-colors">
                <CardContent className="p-3">
                  <div className="flex items-center gap-3">
                    {/* Icon + Name */}
                    <span className="text-lg leading-none shrink-0">
                      {CATEGORY_ICONS[entry.category as keyof typeof CATEGORY_ICONS] ?? "📦"}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">{entry.name_fa}</p>
                      <p className="text-[11px] text-muted-foreground">
                        {entry.quantity} واحد × {formatToman(entry.buy_price)}
                      </p>
                    </div>

                    {/* Current value */}
                    <div className="text-left shrink-0">
                      <p className="text-sm font-semibold tabular-nums">{formatToman(entry.current_value)}</p>
                      <p className={`text-[11px] tabular-nums flex items-center gap-0.5 justify-end ${isPositive ? "text-emerald-400" : "text-red-400"}`}>
                        {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {isPositive ? "+" : ""}{formatToman(Math.abs(entry.profit_loss))}
                        <span className="opacity-70">({isPositive ? "+" : ""}{entry.profit_loss_percent.toFixed(1)}%)</span>
                      </p>
                    </div>

                    {/* Delete */}
                    <button
                      onClick={() => handleDelete(entry.id)}
                      className="text-white/20 hover:text-red-400 transition-colors p-1 shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        !showForm && (
          <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
            <span className="text-3xl">🏦</span>
            <p className="text-sm text-muted-foreground">هنوز دارایی ثبت نشده</p>
            <p className="text-xs text-muted-foreground/60">
              طلا، سکه، دلار، نقره یا مسکن خود را اضافه کنید
            </p>
          </div>
        )
      )}
    </section>
  );
}
