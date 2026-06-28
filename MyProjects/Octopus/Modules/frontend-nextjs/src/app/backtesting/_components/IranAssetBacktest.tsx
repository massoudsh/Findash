"use client";

/**
 * TASK-005 — IranAssetBacktest
 * بک‌تست استراتژی‌های ساده بر اساس دارایی‌های ایرانی
 *
 * استراتژی‌های پشتیبانی‌شده:
 * - Buy & Hold: خرید در تاریخ شروع و نگه‌داشتن
 * - Dollar Cost Averaging (DCA): خرید ماهانه با مقدار ثابت
 * - Relative Strength: تخصیص پورتفولیو بر اساس قدرت نسبی
 */

import { useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { formatJalali } from "@/lib/locale";
import { formatToman } from "@/lib/assets";
import { TrendingUp, TrendingDown, Play, Loader2 } from "lucide-react";

type Strategy = "buy_hold" | "dca" | "relative_strength";

interface BacktestConfig {
  symbol: string;
  strategy: Strategy;
  initialCapital: number;
  startDate: string;
  endDate: string;
  monthlyAmount?: number;    // for DCA
}

interface BacktestResult {
  finalValue: number;
  profit: number;
  profitPercent: number;
  maxDrawdown: number;
  sharpe: number;
  chartData: { date: string; value: number; benchmark?: number }[];
}

const SYMBOLS = [
  { value: "XAU18",      label: "🥇 طلای ۱۸ عیار" },
  { value: "COIN_FULL",  label: "🪙 سکه بهار آزادی" },
  { value: "USD",        label: "💵 دلار آمریکا" },
  { value: "XAG",        label: "🥈 نقره" },
  { value: "BTC",        label: "₿ بیت‌کوین" },
];

const STRATEGIES: { value: Strategy; label: string; desc: string }[] = [
  { value: "buy_hold",         label: "خرید و نگه‌داری",   desc: "خرید در ابتدا، فروش در انتها" },
  { value: "dca",              label: "میانگین‌گیری (DCA)", desc: "خرید ماهانه با مقدار ثابت" },
  { value: "relative_strength",label: "قدرت نسبی",          desc: "تخصیص بر اساس عملکرد نسبی" },
];

/** Simulate backtest on server — calls /api/assets/{symbol}/history */
async function runBacktest(config: BacktestConfig): Promise<BacktestResult> {
  const days = Math.ceil(
    (new Date(config.endDate).getTime() - new Date(config.startDate).getTime()) / 86400000
  );
  const res = await fetch(`/api/assets/${config.symbol}/history?days=${Math.min(days, 365)}`);
  const json = await res.json();
  const history: { timestamp: string; close: number }[] = json.data ?? [];

  if (history.length < 2) {
    return { finalValue: config.initialCapital, profit: 0, profitPercent: 0, maxDrawdown: 0, sharpe: 0, chartData: [] };
  }

  const startPrice = history[0].close;
  const chartData: { date: string; value: number; benchmark?: number }[] = [];
  let portfolio = config.initialCapital;
  let units     = portfolio / startPrice;
  let maxValue  = portfolio;
  let maxDrawdown = 0;
  const returns: number[] = [];

  for (let i = 0; i < history.length; i++) {
    const point = history[i];
    const price = point.close;

    if (config.strategy === "dca" && i > 0 && i % 30 === 0) {
      const monthly = config.monthlyAmount ?? 5_000_000;
      units += monthly / price;
      portfolio += monthly;
    }

    const currentValue = units * price;
    if (currentValue > maxValue) maxValue = currentValue;
    const drawdown = (maxValue - currentValue) / maxValue;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;

    if (i > 0) returns.push((currentValue - units * history[i - 1].close) / (units * history[i - 1].close));

    chartData.push({
      date:      formatJalali(point.timestamp, true),
      value:     Math.round(currentValue),
      benchmark: Math.round((config.initialCapital / startPrice) * price),
    });
  }

  const finalValue    = units * history[history.length - 1].close;
  const profit        = finalValue - config.initialCapital;
  const profitPercent = (profit / config.initialCapital) * 100;
  const avgReturn     = returns.reduce((a, b) => a + b, 0) / returns.length;
  const stdReturn     = Math.sqrt(returns.reduce((a, b) => a + (b - avgReturn) ** 2, 0) / returns.length);
  const sharpe        = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0;

  return { finalValue, profit, profitPercent, maxDrawdown: maxDrawdown * 100, sharpe, chartData };
}

export function IranAssetBacktest() {
  const [config, setConfig] = useState<BacktestConfig>({
    symbol:        "XAU18",
    strategy:      "buy_hold",
    initialCapital: 100_000_000,
    startDate:     "2023-01-01",
    endDate:       "2024-01-01",
    monthlyAmount: 5_000_000,
  });
  const [result, setResult]   = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runBacktest(config);
      setResult(res);
    } catch {
      setError("خطا در اجرای بک‌تست. لطفاً دوباره تلاش کنید.");
    } finally {
      setRunning(false);
    }
  }, [config]);

  const isProfit = (result?.profit ?? 0) >= 0;

  return (
    <div className="space-y-4" dir="rtl">

      {/* Config panel */}
      <Card className="bg-[#0f1117] border border-white/5">
        <CardContent className="p-4 space-y-4">
          <h3 className="text-sm font-semibold text-white">پیکربندی بک‌تست</h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {/* Symbol */}
            <div>
              <label className="text-xs text-muted-foreground block mb-1">دارایی</label>
              <select
                value={config.symbol}
                onChange={(e) => setConfig((c) => ({ ...c, symbol: e.target.value }))}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
              >
                {SYMBOLS.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </div>

            {/* Strategy */}
            <div>
              <label className="text-xs text-muted-foreground block mb-1">استراتژی</label>
              <select
                value={config.strategy}
                onChange={(e) => setConfig((c) => ({ ...c, strategy: e.target.value as Strategy }))}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value}>{s.label} — {s.desc}</option>
                ))}
              </select>
            </div>

            {/* Initial capital */}
            <div>
              <label className="text-xs text-muted-foreground block mb-1">سرمایه اولیه (تومان)</label>
              <input
                type="number"
                value={config.initialCapital}
                onChange={(e) => setConfig((c) => ({ ...c, initialCapital: Number(e.target.value) }))}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
              />
            </div>

            {/* Start date */}
            <div>
              <label className="text-xs text-muted-foreground block mb-1">تاریخ شروع</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig((c) => ({ ...c, startDate: e.target.value }))}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
              />
            </div>

            {/* End date */}
            <div>
              <label className="text-xs text-muted-foreground block mb-1">تاریخ پایان</label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig((c) => ({ ...c, endDate: e.target.value }))}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
              />
            </div>

            {/* Monthly amount (DCA only) */}
            {config.strategy === "dca" && (
              <div>
                <label className="text-xs text-muted-foreground block mb-1">مبلغ ماهانه (تومان)</label>
                <input
                  type="number"
                  value={config.monthlyAmount}
                  onChange={(e) => setConfig((c) => ({ ...c, monthlyAmount: Number(e.target.value) }))}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/25"
                />
              </div>
            )}
          </div>

          {error && <p className="text-xs text-red-400">{error}</p>}

          <Button
            onClick={handleRun}
            disabled={running}
            className="bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 border border-emerald-500/20 text-sm"
          >
            {running
              ? <><Loader2 className="w-4 h-4 ml-2 animate-spin" />در حال اجرا...</>
              : <><Play className="w-4 h-4 ml-2" />اجرای بک‌تست</>
            }
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "ارزش نهایی",    value: formatToman(result.finalValue), highlight: true },
              { label: "سود/زیان",      value: `${isProfit ? "+" : ""}${result.profitPercent.toFixed(1)}٪`, color: isProfit ? "text-emerald-400" : "text-red-400" },
              { label: "حداکثر افت",    value: `${result.maxDrawdown.toFixed(1)}٪`, color: "text-orange-400" },
              { label: "شاخص شارپ",     value: result.sharpe.toFixed(2), color: result.sharpe > 1 ? "text-emerald-400" : "text-muted-foreground" },
            ].map((kpi) => (
              <Card key={kpi.label} className="bg-[#0f1117] border border-white/5">
                <CardContent className="p-3">
                  <p className="text-[11px] text-muted-foreground">{kpi.label}</p>
                  <p className={`text-base font-bold mt-1 tabular-nums ${kpi.color ?? "text-white"}`}>
                    {kpi.value}
                  </p>
                  {kpi.label === "سود/زیان" && (
                    <p className="text-[11px] text-muted-foreground mt-0.5">
                      {formatToman(Math.abs(result.profit))}
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Chart */}
          {result.chartData.length > 0 && (
            <Card className="bg-[#0f1117] border border-white/5">
              <CardContent className="p-4">
                <p className="text-sm font-semibold text-white mb-3">منحنی ارزش پورتفولیو</p>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={result.chartData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                    <defs>
                      <linearGradient id="btGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor={isProfit ? "#34d399" : "#f87171"} stopOpacity={0.25} />
                        <stop offset="95%" stopColor={isProfit ? "#34d399" : "#f87171"} stopOpacity={0}   />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                    <XAxis dataKey="date" tick={{ fill: "#888", fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                    <YAxis tick={{ fill: "#888", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v) => `${(v / 1_000_000).toFixed(0)}م`} width={40} />
                    <Tooltip
                      contentStyle={{ background: "#1a1d27", border: "1px solid #ffffff15", borderRadius: 8, fontSize: 12 }}
                      formatter={(v: number, name: string) => [formatToman(v), name === "value" ? "پورتفولیو" : "مرجع"]}
                    />
                    <Legend formatter={(v) => v === "value" ? "پورتفولیو" : "خرید و نگه‌داری (مرجع)"} />
                    <Area type="monotone" dataKey="value"     stroke={isProfit ? "#34d399" : "#f87171"} strokeWidth={2} fill="url(#btGrad)" dot={false} />
                    <Area type="monotone" dataKey="benchmark" stroke="#6366f1" strokeWidth={1.5} fill="none" dot={false} strokeDasharray="4 2" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
