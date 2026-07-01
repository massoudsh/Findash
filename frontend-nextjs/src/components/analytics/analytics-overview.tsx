'use client';

import { TrendingUp, BarChart2, Activity, PieChart } from 'lucide-react';
import { useEffect, useState } from 'react';
import { formatJalali } from '@/lib/jalali';

// Simple bar chart using divs (no extra dependency)
function MiniBarChart({ data }: { data: { label: string; value: number; color: string }[] }) {
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="flex items-end gap-1.5 h-24 w-full">
      {data.map((d) => (
        <div key={d.label} className="flex flex-col items-center gap-1 flex-1">
          <div
            className="w-full rounded-t-md transition-all duration-700"
            style={{
              height: `${(d.value / max) * 80}px`,
              background: d.color,
              opacity: 0.85,
            }}
          />
          <span className="text-[9px] text-muted-foreground leading-none">{d.label}</span>
        </div>
      ))}
    </div>
  );
}

const PORTFOLIO_ALLOCATION = [
  { label: 'کریپتو', value: 45, color: '#22C55E' },
  { label: 'طلا', value: 30, color: '#d4a017' },
  { label: 'ارز', value: 15, color: '#3b82f6' },
  { label: 'سهام', value: 10, color: '#8b5cf6' },
];

const MONTHLY_RETURN = [
  { label: 'فر', value: 8.2, color: '#22C55E' },
  { label: 'ار', value: 12.1, color: '#22C55E' },
  { label: 'خر', value: -3.4, color: '#ef4444' },
  { label: 'تیر', value: 6.7, color: '#22C55E' },
  { label: 'مر', value: 15.3, color: '#22C55E' },
  { label: 'شه', value: 2.1, color: '#22C55E' },
];

const RISK_METRICS = [
  { label: 'VaR روزانه (۹۵٪)', value: '۲.۳٪', icon: Activity, trend: 'down' },
  { label: 'حداکثر افت', value: '۱۲.۷٪', icon: TrendingUp, trend: 'up' },
  { label: 'نسبت شارپ', value: '۱.۸۴', icon: BarChart2, trend: 'up' },
  { label: 'بتا (نسبت به بازار)', value: '۰.۷۲', icon: PieChart, trend: 'neutral' },
];

export function AnalyticsOverview() {
  const [today, setToday] = useState('');

  useEffect(() => {
    setToday(formatJalali(new Date()));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-black">تحلیل پیشرفته</h2>
          <p className="text-xs text-muted-foreground mt-0.5">{today}</p>
        </div>
        <span className="persian-badge text-xs px-2 py-0.5 rounded-full">نسخه آزمایشی</span>
      </div>

      {/* Risk metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {RISK_METRICS.map((m) => (
          <div key={m.label} className="persian-card rounded-2xl p-4 border border-border/40">
            <div className="flex items-center gap-2 mb-2">
              <m.icon className="h-4 w-4 text-green-400" />
              <span className="text-[10px] text-muted-foreground">{m.label}</span>
            </div>
            <div className={`text-lg font-black ${m.trend === 'down' ? 'text-green-400' : m.trend === 'up' ? 'text-red-400' : 'text-foreground'}`}>
              {m.value}
            </div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {/* Monthly returns */}
        <div className="persian-card rounded-2xl p-5 border border-border/40">
          <h3 className="text-sm font-bold mb-4">بازده ماهانه پرتفولیو</h3>
          <MiniBarChart
            data={MONTHLY_RETURN.map((d) => ({
              label: d.label,
              value: Math.abs(d.value),
              color: d.value >= 0 ? '#22C55E' : '#ef4444',
            }))}
          />
          <p className="text-xs text-muted-foreground mt-3">
            بهترین ماه: <span className="text-green-400 font-semibold">مرداد (+۱۵.۳٪)</span>
          </p>
        </div>

        {/* Allocation */}
        <div className="persian-card rounded-2xl p-5 border border-border/40">
          <h3 className="text-sm font-bold mb-4">ترکیب دارایی‌ها</h3>
          <div className="space-y-3">
            {PORTFOLIO_ALLOCATION.map((a) => (
              <div key={a.label} className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground w-12">{a.label}</span>
                <div className="flex-1 h-2 bg-muted/30 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${a.value}%`, background: a.color }}
                  />
                </div>
                <span className="text-xs font-semibold w-8 text-left" style={{ color: a.color }}>
                  {a.value}٪
                </span>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            بیشترین وزن: <span className="text-green-400 font-semibold">کریپتو (۴۵٪)</span>
          </p>
        </div>
      </div>

      {/* Note */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-xs text-amber-400 text-center">
        داده‌های نمایش داده شده آزمایشی هستند. اتصال به پرتفولیو واقعی در نسخه بعدی فعال می‌شود.
      </div>
    </div>
  );
}
