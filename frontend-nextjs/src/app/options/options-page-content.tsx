'use client';

import { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Star, Download, FileSpreadsheet, FileJson } from 'lucide-react';
import { toPersian, formatToman } from '@/lib/fa-utils';

// ─── Mock data ────────────────────────────────────────────────────────────────

const INFLOW_DATA = [
  { time: '۹:۱۹', value: -95 }, { time: '۱۰:۰۱', value: -60 },
  { time: '۱۰:۴۶', value: 20 },  { time: '۱۱:۳۱', value: 80 },
  { time: '۱۲:۱۶', value: 140 }, { time: '۱۳:۰۲', value: 200 },
  { time: '۱۳:۵۱', value: 240 }, { time: '۱۴:۴۱', value: 270 },
  { time: '۱۵:۱۹', value: 285 },
];

const CALL_OPTIONS = [
  { symbol: 'ضهرم۵۰۳۵', lastPrice: '۳٫۳۵۰', tradeValue: '۲٫۳۳ T', daysLeft: 51, starred: false },
  { symbol: 'ضستا۴۰۴۵', lastPrice: '۲۳۵',    tradeValue: '۸۶۹٫۰۹ B', daysLeft: 2,  starred: false },
  { symbol: 'ضهرم۵۰۳۴', lastPrice: '۵٫۱۲۴',  tradeValue: '۷۸۴٫۶۸ B', daysLeft: 51, starred: false },
  { symbol: 'ضهرم۵۰۳۳', lastPrice: '۶٫۶۰۹',  tradeValue: '۵۹۴٫۵۹ B', daysLeft: 51, starred: false },
  { symbol: 'ضهرم۵۰۳۲', lastPrice: '۸٫۳۳۷',  tradeValue: '۵۶۸٫۹۳ B', daysLeft: 51, starred: false },
  { symbol: 'ضستا۴۰۴۶', lastPrice: '۶۱',     tradeValue: '۴۵۱٫۴۹ B', daysLeft: 2,  starred: false },
];

const PUT_OPTIONS = [
  { symbol: 'طهرم۵۰۳۵', lastPrice: '۵٫۵۲۵', tradeValue: '۳۲۰٫۵ B', daysLeft: 51, starred: false },
  { symbol: 'طبستا۴۰۴۶', lastPrice: '۱۹',   tradeValue: '۲۶۳٫۵۱ B', daysLeft: 2,  starred: false },
  { symbol: 'طهرم۵۰۳۲', lastPrice: '۳٫۶۰۰', tradeValue: '۱۷۹٫۹۳ B', daysLeft: 51, starred: false },
  { symbol: 'طهرم۵۰۳۵', lastPrice: '۱۱٫۸۳۱', tradeValue: '۱۷۳٫۰۱ B', daysLeft: 51, starred: false },
  { symbol: 'طهرم۵۰۳۱', lastPrice: '۱٫۹۰۰', tradeValue: '۱۴۰٫۲۸ B', daysLeft: 51, starred: false },
  { symbol: 'طبستا۴۰۲۲', lastPrice: '۵۸',   tradeValue: '۷۵٫۳ B',   daysLeft: 16, starred: false },
];

const EXPIRY_BASE = [
  { symbol: 'خودرو', expiry: '۱۴۰۵/۰۴/۱۰', daysLeft: 2,  position: '۷۵٬۸۸۸٬۸۲۹' },
  { symbol: 'فولاد', expiry: '۱۴۰۵/۰۴/۱۰', daysLeft: 2,  position: '۱۲٬۴۵۰٬۰۰۰' },
  { symbol: 'شستا',  expiry: '۱۴۰۵/۰۵/۱۴', daysLeft: 45, position: '۸٬۷۶۰٬۰۰۰'  },
  { symbol: 'ذوب',  expiry: '۱۴۰۵/۰۶/۱۸', daysLeft: 79, position: '۵٬۲۰۰٬۰۰۰'  },
];

const BALANCE_DATA = [
  { name: 'مثبت', value: 388, color: '#22c55e' },
  { name: 'خنثی', value: 120, color: '#94a3b8' },
  { name: 'منفی', value: 210, color: '#ef4444' },
];

const VALUE_DATA = [
  { day: '۱', buy: 120 }, { day: '۲', buy: 200 }, { day: '۳', buy: 150 },
  { day: '۴', buy: 300 }, { day: '۵', buy: 180 }, { day: '۶', buy: 420 },
  { day: '۷', buy: 260 }, { day: '۸', buy: 340 }, { day: '۹', buy: 190 },
  { day: '۱۰', buy: 280 },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function IconBtn({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <button title={title} className="p-1.5 rounded hover:bg-white/10 text-muted-foreground hover:text-foreground transition">
      {children}
    </button>
  );
}

function FilterTabs({ items, active, onSelect }: { items: string[]; active: string; onSelect: (v: string) => void }) {
  return (
    <div className="flex gap-1 bg-muted/30 rounded-lg p-0.5">
      {items.map(item => (
        <button
          key={item}
          onClick={() => onSelect(item)}
          className={`px-3 py-1 rounded-md text-xs font-medium transition ${
            active === item ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

// ─── Panel: ورود حقیقی ───────────────────────────────────────────────────────

function InflowPanel() {
  const [filter, setFilter] = useState('همه');
  const [view, setView] = useState<'گراف' | 'جدول'>('گراف');

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <span className="font-bold text-base">ورود حقیقی</span>
        <FilterTabs items={['گراف', 'جدول']} active={view} onSelect={v => setView(v as 'گراف' | 'جدول')} />
      </div>
      <div className="flex items-center justify-between">
        <IconBtn title="دانلود"><Download className="h-4 w-4" /></IconBtn>
        <FilterTabs items={['همه', 'اختیار خرید', 'اختیار فروش']} active={filter} onSelect={setFilter} />
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={INFLOW_DATA} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="inflowGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false}
              tickFormatter={v => v === 0 ? '۰' : `${toPersian(Math.abs(v))} M`} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [`${toPersian(v)} M`, 'ورود حقیقی']}
            />
            <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2}
              fill="url(#inflowGrad)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Panel: جدول قراردادها ────────────────────────────────────────────────────

type OptionRow = { symbol: string; lastPrice: string; tradeValue: string; daysLeft: number; starred: boolean };

function OptionsTablePanel() {
  const [calls, setCalls] = useState<OptionRow[]>(CALL_OPTIONS);
  const [puts, setPuts] = useState<OptionRow[]>(PUT_OPTIONS);

  const toggleStar = (list: OptionRow[], setList: (r: OptionRow[]) => void, idx: number) => {
    setList(list.map((r, i) => i === idx ? { ...r, starred: !r.starred } : r));
  };

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          <IconBtn title="اکسل"><FileSpreadsheet className="h-4 w-4 text-green-500" /></IconBtn>
          <IconBtn title="JSON"><FileJson className="h-4 w-4" /></IconBtn>
          <IconBtn title="دانلود"><Download className="h-4 w-4" /></IconBtn>
        </div>
        <span className="text-sm font-semibold text-muted-foreground">بیشترین ارزش معاملات</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              {/* Call header */}
              <th colSpan={4} className="text-center py-2 text-green-400 font-bold border-b border-green-500/30 bg-green-500/5 rounded-t">
                اختیار خرید
              </th>
              {/* divider */}
              <td className="w-2" />
              {/* Put header */}
              <th colSpan={4} className="text-center py-2 text-red-400 font-bold border-b border-red-500/30 bg-red-500/5 rounded-t">
                اختیار فروش
              </th>
            </tr>
            <tr className="text-muted-foreground border-b border-border/50">
              <th className="py-2 px-1 text-end font-medium">نماد</th>
              <th className="py-2 px-1 text-center font-medium">آخرین قیمت</th>
              <th className="py-2 px-1 text-center font-medium">ارزش معامله</th>
              <th className="py-2 px-1 text-center font-medium">روز مانده</th>
              <td className="w-2 border-r border-border/40" />
              <th className="py-2 px-1 text-end font-medium">نماد</th>
              <th className="py-2 px-1 text-center font-medium">آخرین قیمت</th>
              <th className="py-2 px-1 text-center font-medium">ارزش معامله</th>
              <th className="py-2 px-1 text-center font-medium">روز مانده</th>
            </tr>
          </thead>
          <tbody>
            {calls.map((call, i) => {
              const put = puts[i];
              return (
                <tr key={i} className="border-b border-border/30 hover:bg-muted/20 transition">
                  {/* Call side */}
                  <td className="py-2 px-1 text-end">
                    <button onClick={() => toggleStar(calls, setCalls, i)} className="me-1 opacity-50 hover:opacity-100">
                      <Star className={`h-3 w-3 ${call.starred ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                    </button>
                    <span className="text-green-400 font-medium">{call.symbol}</span>
                  </td>
                  <td className="py-2 px-1 text-center tabular-nums">{call.lastPrice}</td>
                  <td className="py-2 px-1 text-center tabular-nums text-muted-foreground">{call.tradeValue}</td>
                  <td className="py-2 px-1 text-center">{toPersian(call.daysLeft)}</td>
                  <td className="border-r border-border/40" />
                  {/* Put side */}
                  <td className="py-2 px-1 text-end">
                    <button onClick={() => toggleStar(puts, setPuts, i)} className="me-1 opacity-50 hover:opacity-100">
                      <Star className={`h-3 w-3 ${put.starred ? 'fill-yellow-400 text-yellow-400' : ''}`} />
                    </button>
                    <span className="text-red-400 font-medium">{put.symbol}</span>
                  </td>
                  <td className="py-2 px-1 text-center tabular-nums">{put.lastPrice}</td>
                  <td className="py-2 px-1 text-center tabular-nums text-muted-foreground">{put.tradeValue}</td>
                  <td className="py-2 px-1 text-center">{toPersian(put.daysLeft)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Panel: سررسیدها ──────────────────────────────────────────────────────────

function ExpiryPanel() {
  const [tab, setTab] = useState('نماد پایه');

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <span className="font-bold text-base">سررسیدها</span>
        <FilterTabs items={['نماد پایه', 'قراردادها']} active={tab} onSelect={setTab} />
      </div>
      <div className="flex gap-1">
        <IconBtn title="اکسل"><FileSpreadsheet className="h-4 w-4 text-green-500" /></IconBtn>
        <IconBtn title="JSON"><FileJson className="h-4 w-4" /></IconBtn>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground border-b border-border/50">
              <th className="py-2 text-end font-medium">نماد</th>
              <th className="py-2 text-center font-medium">تاریخ سررسید</th>
              <th className="py-2 text-center font-medium">روز مانده</th>
              <th className="py-2 text-start font-medium">موقعیت باز</th>
            </tr>
          </thead>
          <tbody>
            {EXPIRY_BASE.map((row, i) => (
              <tr key={i} className="border-b border-border/30 hover:bg-muted/20 transition">
                <td className="py-2 text-end">
                  <span className="me-1 opacity-40 cursor-pointer"><Star className="h-3 w-3 inline" /></span>
                  <span className="text-primary font-medium">{row.symbol}</span>
                </td>
                <td className="py-2 text-center tabular-nums">{row.expiry}</td>
                <td className="py-2 text-center">{toPersian(row.daysLeft)}</td>
                <td className="py-2 text-start tabular-nums">{row.position}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Panel: توازن بازار ───────────────────────────────────────────────────────

function BalancePanel() {
  const [tab, setTab] = useState('اختیار');
  const total = BALANCE_DATA.reduce((s, d) => s + d.value, 0);

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <span className="font-bold text-base">توازن بازار</span>
        <FilterTabs items={['اختیار', 'دارایی پایه']} active={tab} onSelect={setTab} />
      </div>
      <div className="flex-1 flex flex-col items-center justify-center gap-3">
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie data={BALANCE_DATA} cx="50%" cy="100%" startAngle={180} endAngle={0}
              innerRadius={60} outerRadius={90} dataKey="value" paddingAngle={2}>
              {BALANCE_DATA.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="flex gap-4 text-xs mt-[-40px]">
          {BALANCE_DATA.map((d, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
              <span className="text-muted-foreground">{d.name}</span>
              <span className="font-semibold">{toPersian(d.value)}</span>
            </div>
          ))}
        </div>
        <div className="text-xs text-muted-foreground">
          تعداد نمادهای مثبت: <span className="text-green-400 font-bold">{toPersian(BALANCE_DATA[0].value)}</span>
        </div>
      </div>
    </div>
  );
}

// ─── Panel: ارزش اختیار خریدها ────────────────────────────────────────────────

function CallValuePanel() {
  const [period, setPeriod] = useState('روز');
  const [relative, setRelative] = useState('نسبت به اختیار فروش‌ها');

  return (
    <div className="bg-card border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
      <div className="flex items-center justify-between">
        <span className="font-bold text-base">ارزش اختیار خریدها</span>
        <FilterTabs items={['روز', 'هفته', 'ماه']} active={period} onSelect={setPeriod} />
      </div>
      <div className="flex justify-end">
        <FilterTabs
          items={['نسبت به اختیار فروش‌ها', 'نسبت به کل اختیارها']}
          active={relative}
          onSelect={setRelative}
        />
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={VALUE_DATA} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
            <XAxis dataKey="day" tick={{ fontSize: 9, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
            <YAxis hide />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [toPersian(v), 'ارزش']}
            />
            <Bar dataKey="buy" fill="#3b82f6" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

const TOP_TABS = ['نمای کلی', 'زنجیره قراردادها', 'لیست قراردادها', 'نقشه بازار', 'استراتژی', 'بورس کالا'];

export function OptionsPageContent() {
  const [topTab, setTopTab] = useState('نمای کلی');

  return (
    <div className="flex flex-col gap-0 min-h-screen bg-background">
      {/* Top tab bar */}
      <div className="border-b border-border bg-card/50 sticky top-0 z-10 backdrop-blur">
        <div className="container mx-auto px-4">
          <div className="flex gap-0 overflow-x-auto">
            {TOP_TABS.map(tab => (
              <button
                key={tab}
                onClick={() => setTopTab(tab)}
                className={`px-5 py-3.5 text-sm font-medium whitespace-nowrap border-b-2 transition ${
                  topTab === tab
                    ? 'border-primary text-primary'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="container mx-auto px-4 py-4 flex flex-col gap-4">
        {topTab === 'نمای کلی' && (
          <>
            {/* Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              <div className="lg:col-span-2">
                <InflowPanel />
              </div>
              <div className="lg:col-span-3">
                <OptionsTablePanel />
              </div>
            </div>

            {/* Row 2 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ExpiryPanel />
              <BalancePanel />
              <CallValuePanel />
            </div>
          </>
        )}

        {topTab !== 'نمای کلی' && (
          <div className="flex flex-col items-center justify-center min-h-[400px] text-muted-foreground gap-2">
            <span className="text-5xl">📊</span>
            <span className="text-lg font-medium">{topTab}</span>
            <span className="text-sm">به‌زودی در دسترس خواهد بود</span>
          </div>
        )}
      </div>
    </div>
  );
}
