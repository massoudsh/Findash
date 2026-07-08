'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { toast } from '@/components/ui/toast';
import {
  Target, TrendingUp, Sparkles, X, AlertTriangle,
  CheckCircle, Clock, BarChart3, Shield, Zap, Info,
} from 'lucide-react';
import { strategyTemplates } from './options-content';
import { cn } from '@/lib/utils';

export interface OptionsStrategyForTerminal {
  name: string;
  description?: string;
  category?: string;
}

interface OptionsStrategiesTabProps {
  onTradeInTerminal?: (strategy?: OptionsStrategyForTerminal) => void;
}

type Strategy = typeof strategyTemplates[0] & { warning?: string };

/* ── Persian labels ── */
const CATEGORY_FA: Record<string, string> = {
  Directional: 'جهت‌دار',
  Income: 'درآمدزا',
  Spread: 'اسپرد',
  Neutral: 'خنثی',
  Volatility: 'نوسانی',
  Protective: 'محافظتی',
  Algorithmic: 'الگوریتمی',
  all: 'همه',
};

const COMPLEXITY_FA: Record<string, string> = {
  Beginner: 'مبتدی',
  Intermediate: 'متوسط',
  Advanced: 'پیشرفته',
  Expert: 'حرفه‌ای',
};

const RISK_FA: Record<string, string> = {
  Low: 'کم',
  Medium: 'متوسط',
  High: 'زیاد',
  'Very High': 'خیلی زیاد',
};

const OUTLOOK_FA: Record<string, string> = {
  Bullish: 'صعودی',
  Bearish: 'نزولی',
  Neutral: 'خنثی',
  'Moderately Bullish': 'نسبتاً صعودی',
  'Moderately Bearish': 'نسبتاً نزولی',
  'Neutral to Bullish': 'خنثی تا صعودی',
  'Neutral to Bearish': 'خنثی تا نزولی',
  Directional: 'جهت‌دار',
  Any: 'هر شرایطی',
};

const VOL_FA: Record<string, string> = {
  High: 'نوسان بالا',
  Low: 'نوسان پایین',
  'Low to Medium': 'پایین تا متوسط',
  Medium: 'متوسط',
  Any: 'هر نوسانی',
};

const DECAY_FA: Record<string, string> = {
  Positive: 'مثبت (به نفع)',
  Negative: 'منفی (به ضرر)',
  Neutral: 'خنثی',
};

/* ── When to use / avoid ── */
const STRATEGY_TIPS: Record<string, { use: string[]; avoid: string[]; steps: string[]; greeks: string }> = {
  'Long Call': {
    use: ['انتظار رشد قوی قیمت داری', 'می‌خواهی ریسکت را به پریمیوم محدود کنی', 'اهرم می‌خواهی بدون خرید سهام'],
    avoid: ['بازار رنج یا نزولی است', 'نوسانات ضمنی (IV) خیلی بالا باشد (گران است)'],
    steps: ['نماد و تاریخ انقضا را انتخاب کن', 'استرایک ATM یا کمی OTM را بخر', 'پریمیوم پرداخت می‌کنی', 'سود در صورت رشد قیمت بالای سطح سربه‌سر'],
    greeks: 'دلتا مثبت · گاما مثبت · تتا منفی · وگا مثبت',
  },
  'Long Put': {
    use: ['انتظار افت قیمت داری', 'می‌خواهی از سبد محافظت کنی', 'موقعیت short بدون ریسک نامحدود'],
    avoid: ['بازار صعودی قوی است', 'IV خیلی بالا باشد'],
    steps: ['نماد را انتخاب کن', 'استرایک ATM یا کمی OTM را بخر', 'پریمیوم پرداخت می‌کنی', 'سود در صورت افت قیمت زیر سطح سربه‌سر'],
    greeks: 'دلتا منفی · گاما مثبت · تتا منفی · وگا مثبت',
  },
  'Covered Call': {
    use: ['سهام داری و بازار رنج یا کمی صعودی است', 'می‌خواهی از سهامت درآمد پریمیوم کسب کنی', 'حاضری سهام را در قیمت استرایک بفروشی'],
    avoid: ['انتظار رشد شارپ داری (سود را محدود می‌کند)', 'سهام می‌تواند سقوط کند'],
    steps: ['۱۰۰ سهم داشته باش', 'کال OTM با تاریخ کوتاه بفروش', 'پریمیوم دریافت می‌کنی', 'سود محدود به استرایک + پریمیوم'],
    greeks: 'دلتا پایین مثبت · تتا مثبت · وگا منفی',
  },
  'Cash-Secured Put': {
    use: ['می‌خواهی سهام را زیر قیمت فعلی بخری', 'درآمد پریمیوم کسب کنی', 'IV بالاست (پریمیوم گران است)'],
    avoid: ['سهام سقوط تند داشته باشد', 'حاضر به خرید سهام نیستی'],
    steps: ['پول کافی در حساب داشته باش (strike × 100)', 'پوت OTM بفروش', 'پریمیوم دریافت می‌کنی', 'اگر تا انقضا OTM ماند سود کامل'],
    greeks: 'دلتا منفی کوچک · تتا مثبت · وگا منفی',
  },
  'Bull Call Spread': {
    use: ['انتظار رشد ملایم داری', 'می‌خواهی هزینه را کاهش دهی', 'IV بالاست و کال گران است'],
    avoid: ['رشد خیلی شارپ انتظار داری (سود محدود می‌شود)', 'بازار نزولی است'],
    steps: ['کال با استرایک پایین‌تر بخر', 'کال با استرایک بالاتر بفروش', 'هر دو یک تاریخ انقضا', 'سود محدود به اختلاف استرایک‌ها'],
    greeks: 'دلتا مثبت کوچک · تتا خنثی · وگا خنثی',
  },
  'Bear Put Spread': {
    use: ['انتظار افت ملایم داری', 'می‌خواهی هزینه پوت را کاهش دهی'],
    avoid: ['افت خیلی شارپ انتظار داری', 'بازار صعودی است'],
    steps: ['پوت با استرایک بالاتر بخر', 'پوت با استرایک پایین‌تر بفروش', 'هر دو یک تاریخ انقضا', 'سود محدود به اختلاف استرایک‌ها'],
    greeks: 'دلتا منفی کوچک · تتا خنثی · وگا خنثی',
  },
  'Iron Condor': {
    use: ['بازار رنج است و نوسان پایین انتظار داری', 'IV بالاست (پریمیوم گران، بفروش)', 'درآمد ثابت در شرایط خنثی'],
    avoid: ['خبر مهم یا رویداد بزرگ نزدیک است', 'انتظار حرکت بزرگ قیمت داری'],
    steps: ['پوت OTM بفروش', 'پوت با استرایک پایین‌تر بخر', 'کال OTM بفروش', 'کال با استرایک بالاتر بخر', 'پریمیوم خالص دریافت می‌کنی'],
    greeks: 'دلتا نزدیک صفر · تتا مثبت · وگا منفی · گاما منفی',
  },
  'Iron Butterfly': {
    use: ['بازار کاملاً رنج و نوسان پایین انتظار داری', 'IV خیلی بالاست', 'قیمت دقیقاً نزدیک ATM بماند'],
    avoid: ['انتظار هرگونه حرکت قوی قیمت داری'],
    steps: ['کال ATM بفروش', 'پوت ATM بفروش', 'کال OTM بخر', 'پوت OTM بخر'],
    greeks: 'دلتا صفر · تتا مثبت بالا · وگا منفی بالا · گاما منفی بالا',
  },
  'Long Straddle': {
    use: ['رویداد مهم (earnings, خبر مهم) نزدیک است', 'انتظار نوسان بزرگ داری اما نمی‌دانی کدام جهت', 'IV نسبتاً پایین است'],
    avoid: ['بازار رنج خواهد بود', 'IV خیلی بالاست (گران می‌خری)'],
    steps: ['کال ATM بخر', 'پوت ATM بخر', 'هر دو یک استرایک و تاریخ', 'سود در صورت حرکت بزرگ در هر جهت'],
    greeks: 'دلتا صفر · گاما مثبت · تتا منفی · وگا مثبت',
  },
  'Long Strangle': {
    use: ['رویداد بزرگ نزدیک است', 'هزینه کمتر از استرادل می‌خواهی', 'انتظار حرکت خیلی بزرگ داری'],
    avoid: ['بازار رنج خواهد بود', 'هزینه بالا نسبت به حرکت احتمالی'],
    steps: ['کال OTM بخر', 'پوت OTM بخر', 'تاریخ یکسان', 'سربه‌سر دورتر از استرادل است'],
    greeks: 'دلتا صفر · گاما مثبت · تتا منفی بالا · وگا مثبت',
  },
  'Short Straddle': {
    use: ['بازار کاملاً رنج و IV خیلی بالاست', 'درآمد بالا در شرایط آرام بازار'],
    avoid: ['رویداد یا خبر مهم نزدیک است', 'ریسک نامحدود در هر دو جهت'],
    steps: ['کال ATM بفروش', 'پوت ATM بفروش', 'پریمیوم بالا دریافت می‌کنی', 'ریسک: حرکت بزرگ در هر جهت زیان‌ده است'],
    greeks: 'دلتا صفر · گاما منفی · تتا مثبت · وگا منفی',
  },
  'Collar': {
    use: ['سهام داری و می‌خواهی از افت محافظت کنی', 'پریمیوم محافظت را با فروش کال تأمین می‌کنی'],
    avoid: ['سهام را می‌خواهی بفروشی (بهتر است مستقیم بفروشی)'],
    steps: ['سهام داشته باش', 'پوت OTM محافظ بخر', 'کال OTM بفروش تا هزینه را جبران کند'],
    greeks: 'دلتا مثبت کم · تتا خنثی · وگا خنثی',
  },
  'Protective Put': {
    use: ['سهام داری و نگران افت هستی', 'رویداد مهم نزدیک است اما سهام را نمی‌خواهی بفروشی'],
    avoid: ['IV خیلی بالاست (بیمه گران می‌شود)', 'برای مدت طولانی (تتا می‌خورد)'],
    steps: ['سهام داشته باش', 'پوت ATM یا کمی OTM بخر', 'پریمیوم پرداخت می‌کنی', 'سقف زیان: قیمت سهام - استرایک + پریمیوم'],
    greeks: 'دلتا مثبت بالا · گاما مثبت · تتا منفی · وگا مثبت',
  },
};

const DEFAULT_TIPS = {
  use: ['شرایط بازار را بررسی کن', 'IV را با میانگین تاریخی مقایسه کن', 'حجم و ریسک مناسب انتخاب کن'],
  avoid: ['بدون تحقیق کافی وارد نشو', 'بیش از ریسک قابل تحمل وارد نشو'],
  steps: ['استراتژی را در ترمینال باز کن', 'نماد و تاریخ انقضا را انتخاب کن', 'حجم مناسب تعیین کن', 'سفارش ارسال کن'],
  greeks: 'به ترمینال مراجعه کن',
};

/* ── Payoff SVG ── */
function PayoffDiagram({ strategy }: { strategy: Strategy }) {
  const w = 280;
  const h = 120;
  const mid = h / 2;

  type Point = [number, number];

  function buildPath(pts: Point[]) {
    return pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x},${y}`).join(' ');
  }

  let pts: Point[] = [];
  const name = strategy.name;

  if (name === 'Long Call') {
    pts = [[0, mid + 40], [140, mid + 40], [220, mid - 40], [w, mid - 60]];
  } else if (name === 'Long Put') {
    pts = [[0, mid - 60], [60, mid - 40], [140, mid + 40], [w, mid + 40]];
  } else if (name === 'Covered Call') {
    pts = [[0, mid + 30], [100, mid - 10], [160, mid - 30], [w, mid - 30]];
  } else if (name === 'Cash-Secured Put') {
    pts = [[0, mid - 30], [60, mid - 30], [160, mid + 10], [w, mid + 30]];
  } else if (name === 'Bull Call Spread') {
    pts = [[0, mid + 30], [100, mid + 30], [180, mid - 30], [w, mid - 30]];
  } else if (name === 'Bear Put Spread') {
    pts = [[0, mid - 30], [100, mid - 30], [180, mid + 30], [w, mid + 30]];
  } else if (name === 'Iron Condor' || name === 'Iron Butterfly') {
    pts = [[0, mid + 35], [60, mid + 35], [100, mid - 35], [180, mid - 35], [220, mid + 35], [w, mid + 35]];
  } else if (name === 'Long Straddle' || name === 'Long Strangle') {
    pts = [[0, mid - 30], [80, mid + 40], [140, mid + 40], [200, mid - 30], [w, mid - 50]];
  } else if (name === 'Short Straddle') {
    pts = [[0, mid + 30], [80, mid - 40], [140, mid - 40], [200, mid + 30], [w, mid + 50]];
  } else if (name === 'Collar') {
    pts = [[0, mid - 30], [60, mid - 30], [130, mid + 10], [180, mid - 20], [w, mid - 20]];
  } else if (name === 'Protective Put') {
    pts = [[0, mid - 20], [80, mid - 20], [140, mid + 10], [w, mid - 40]];
  } else {
    // generic diagonal
    pts = [[0, mid + 20], [w / 2, mid], [w, mid - 20]];
  }

  const pathD = buildPath(pts);
  const isProfit = strategy.timeDecay === 'Positive';

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full h-28 rounded-lg bg-muted/30"
      aria-label={`نمودار سود/زیان ${strategy.name}`}
    >
      {/* zero line */}
      <line x1="0" y1={mid} x2={w} y2={mid} stroke="currentColor" strokeOpacity="0.2" strokeWidth="1" strokeDasharray="4 3" />
      {/* profit area fill */}
      <path
        d={`${pathD} L${w},${mid} L0,${mid} Z`}
        fill={isProfit ? 'rgba(34,197,94,0.08)' : 'rgba(59,130,246,0.08)'}
      />
      {/* main line */}
      <path
        d={pathD}
        fill="none"
        stroke={isProfit ? '#22c55e' : '#3b82f6'}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* labels */}
      <text x="6" y={mid - 6} fontSize="9" fill="rgba(255,255,255,0.4)">سود</text>
      <text x="6" y={mid + 14} fontSize="9" fill="rgba(255,255,255,0.4)">زیان</text>
      <text x={w / 2} y={h - 4} fontSize="8" fill="rgba(255,255,255,0.3)" textAnchor="middle">قیمت دارایی</text>
    </svg>
  );
}

/* ── complexity/risk colors ── */
function complexityColor(c: string) {
  return c === 'Beginner' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20'
    : c === 'Intermediate' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20'
    : c === 'Advanced' ? 'bg-orange-500/15 text-orange-400 border-orange-500/20'
    : 'bg-red-500/15 text-red-400 border-red-500/20';
}

function riskColor(r: string) {
  return r === 'Low' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20'
    : r === 'Medium' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20'
    : r === 'High' ? 'bg-orange-500/15 text-orange-400 border-orange-500/20'
    : 'bg-red-500/15 text-red-400 border-red-500/20';
}

function categoryColor(cat: string) {
  const map: Record<string, string> = {
    Directional: 'bg-blue-500/15 text-blue-400',
    Income: 'bg-emerald-500/15 text-emerald-400',
    Spread: 'bg-violet-500/15 text-violet-400',
    Neutral: 'bg-slate-500/15 text-slate-400',
    Volatility: 'bg-yellow-500/15 text-yellow-400',
    Protective: 'bg-cyan-500/15 text-cyan-400',
    Algorithmic: 'bg-fuchsia-500/15 text-fuchsia-400',
  };
  return map[cat] ?? 'bg-muted text-muted-foreground';
}

/* ── Detail Dialog ── */
function StrategyDetailDialog({
  strategy,
  open,
  onClose,
  onTrade,
}: {
  strategy: Strategy | null;
  open: boolean;
  onClose: () => void;
  onTrade: (s: Strategy) => void;
}) {
  if (!strategy) return null;
  const tips = STRATEGY_TIPS[strategy.name] ?? DEFAULT_TIPS;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto p-0">
        <DialogHeader className="px-6 pt-5 pb-3 border-b sticky top-0 bg-background z-10">
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-3 text-lg">
              <span className="text-3xl">{strategy.icon}</span>
              <div>
                <div className="font-bold">{strategy.name}</div>
                <div className="text-xs font-normal text-muted-foreground mt-0.5">{strategy.description}</div>
              </div>
            </DialogTitle>
            <button onClick={onClose} className="p-1.5 rounded-md hover:bg-muted transition-colors">
              <X className="h-4 w-4" />
            </button>
          </div>
        </DialogHeader>

        <div className="px-6 py-4 space-y-5">

          {/* Badges row */}
          <div className="flex flex-wrap gap-2">
            <Badge className={cn('border', complexityColor(strategy.complexity))}>
              {COMPLEXITY_FA[strategy.complexity] ?? strategy.complexity}
            </Badge>
            <Badge className={cn('border', riskColor(strategy.riskLevel))}>
              ریسک: {RISK_FA[strategy.riskLevel] ?? strategy.riskLevel}
            </Badge>
            <Badge className={categoryColor(strategy.category)}>
              {CATEGORY_FA[strategy.category] ?? strategy.category}
            </Badge>
            {strategy.marginRequired && (
              <Badge className="bg-red-500/10 text-red-400 border border-red-500/20">نیاز به مارجین</Badge>
            )}
            {(strategy as Strategy & { warning?: string }).warning && (
              <Badge className="bg-red-600/15 text-red-400 border border-red-500/20 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                {(strategy as Strategy & { warning?: string }).warning}
              </Badge>
            )}
          </div>

          {/* Payoff diagram */}
          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1">
              <BarChart3 className="h-3.5 w-3.5" />
              نمودار سود/زیان (تقریبی)
            </p>
            <PayoffDiagram strategy={strategy} />
          </div>

          {/* Key metrics grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              { label: 'حداکثر سود', value: strategy.maxProfit, icon: TrendingUp, good: true },
              { label: 'حداکثر ریسک', value: strategy.maxRisk, icon: Shield, good: false },
              { label: 'سربه‌سر', value: strategy.breakeven, icon: Target, good: null },
              { label: 'چشم‌انداز بازار', value: OUTLOOK_FA[strategy.marketOutlook] ?? strategy.marketOutlook, icon: BarChart3, good: null },
              { label: 'نوسان مطلوب', value: VOL_FA[strategy.volatilityOutlook] ?? strategy.volatilityOutlook, icon: Zap, good: null },
              { label: 'تتا (گذر زمان)', value: DECAY_FA[strategy.timeDecay] ?? strategy.timeDecay, icon: Clock, good: strategy.timeDecay === 'Positive' },
            ].map((m) => {
              const Icon = m.icon;
              return (
                <div key={m.label} className="rounded-lg border bg-muted/20 p-3">
                  <div className="flex items-center gap-1.5 mb-1">
                    <Icon className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-[11px] text-muted-foreground">{m.label}</span>
                  </div>
                  <p className={cn(
                    'text-sm font-semibold',
                    m.good === true ? 'text-emerald-400' : m.good === false ? 'text-red-400' : 'text-foreground'
                  )}>
                    {m.value}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Greeks */}
          <div className="rounded-lg border bg-muted/20 p-3">
            <p className="text-xs font-semibold text-muted-foreground mb-1 flex items-center gap-1">
              <Info className="h-3.5 w-3.5" />
              یونانی‌ها (Greeks)
            </p>
            <p className="text-sm text-foreground/80">{tips.greeks}</p>
          </div>

          {/* Setup steps */}
          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-2">مراحل پیاده‌سازی</p>
            <ol className="space-y-1.5">
              {tips.steps.map((step, i) => (
                <li key={i} className="flex gap-2.5 text-sm text-muted-foreground">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-primary/15 text-primary flex items-center justify-center text-[10px] font-bold mt-0.5">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </div>

          {/* When to use / avoid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
              <p className="text-xs font-semibold text-emerald-400 mb-2 flex items-center gap-1">
                <CheckCircle className="h-3.5 w-3.5" />
                چه زمانی استفاده کنی
              </p>
              <ul className="space-y-1">
                {tips.use.map((u, i) => (
                  <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                    <span className="text-emerald-400 shrink-0">•</span>{u}
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
              <p className="text-xs font-semibold text-red-400 mb-2 flex items-center gap-1">
                <AlertTriangle className="h-3.5 w-3.5" />
                چه زمانی استفاده نکنی
              </p>
              <ul className="space-y-1">
                {tips.avoid.map((a, i) => (
                  <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                    <span className="text-red-400 shrink-0">•</span>{a}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Popularity + legs */}
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span>محبوبیت: {strategy.popularity}%</span>
            <Progress value={strategy.popularity} className="h-1.5 flex-1" />
            <span>پاها: {strategy.legs}</span>
          </div>

          {/* CTA */}
          <Button
            className="w-full"
            onClick={() => { onTrade(strategy); onClose(); }}
          >
            <TrendingUp className="h-4 w-4 mr-2" />
            معامله در ترمینال
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* ── Main Tab ── */
export function OptionsStrategiesTab({ onTradeInTerminal }: OptionsStrategiesTabProps) {
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [detailStrategy, setDetailStrategy] = useState<Strategy | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const categories = ['all', ...Array.from(new Set(strategyTemplates.map((s) => s.category)))];
  const filtered = categoryFilter === 'all'
    ? strategyTemplates
    : strategyTemplates.filter((s) => s.category === categoryFilter);

  const openDetail = (s: Strategy) => { setDetailStrategy(s); setDetailOpen(true); };
  const handleTrade = (s: Strategy) => {
    onTradeInTerminal?.({ name: s.name, description: s.description, category: s.category });
  };

  return (
    <div className="space-y-5 p-4 md:p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Target className="h-5 w-5 text-emerald-400" />
            استراتژی‌های اختیار معامله
          </h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            برای مشاهده جزئیات کامل روی هر کارت کلیک کن
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="دسته‌بندی" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">همه</SelectItem>
              {categories.filter((c) => c !== 'all').map((cat) => (
                <SelectItem key={cat} value={cat}>{CATEGORY_FA[cat] ?? cat}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const suggested = strategyTemplates[Math.floor(Math.random() * strategyTemplates.length)];
              setCategoryFilter(suggested.category);
              openDetail(suggested as Strategy);
              toast({
                title: 'پیشنهاد هوش مصنوعی',
                description: `«${suggested.name}» را امتحان کن — ${suggested.description}`,
                type: 'info',
              });
            }}
          >
            <Sparkles className="h-4 w-4 mr-1" />
            پیشنهاد هوش مصنوعی
          </Button>
        </div>
      </div>

      {/* Strategy cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {filtered.map((strategy) => {
          const s = strategy as Strategy;
          return (
            <Card
              key={s.name}
              className="bg-card/80 hover:border-primary/40 transition-all cursor-pointer overflow-hidden hover:shadow-md group"
              onClick={() => openDetail(s)}
            >
              <CardContent className="p-4 space-y-3">
                {/* top row */}
                <div className="flex items-start justify-between gap-2">
                  <span className="text-2xl group-hover:scale-110 transition-transform">{s.icon}</span>
                  <Badge className={cn('text-[10px] shrink-0', categoryColor(s.category))}>
                    {CATEGORY_FA[s.category] ?? s.category}
                  </Badge>
                </div>

                {/* name + desc */}
                <div>
                  <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">{s.name}</h3>
                  <p className="text-[11px] text-muted-foreground line-clamp-2 mt-0.5">{s.description}</p>
                </div>

                {/* badges */}
                <div className="flex gap-1.5 flex-wrap">
                  <Badge variant="outline" className={cn('text-[10px] border', complexityColor(s.complexity))}>
                    {COMPLEXITY_FA[s.complexity] ?? s.complexity}
                  </Badge>
                  <Badge variant="outline" className={cn('text-[10px] border', riskColor(s.riskLevel))}>
                    ریسک {RISK_FA[s.riskLevel] ?? s.riskLevel}
                  </Badge>
                </div>

                {/* quick metrics */}
                <div className="grid grid-cols-2 gap-1.5 text-[10px] text-muted-foreground">
                  <div className="flex flex-col gap-0.5">
                    <span>حداکثر سود</span>
                    <span className="text-emerald-400 font-medium truncate">{s.maxProfit}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span>حداکثر ریسک</span>
                    <span className="text-red-400 font-medium truncate">{s.maxRisk}</span>
                  </div>
                </div>

                {/* popularity */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>محبوبیت</span>
                    <span>{s.popularity}%</span>
                  </div>
                  <Progress value={s.popularity} className="h-1" />
                </div>

                {/* mini payoff preview */}
                <PayoffDiagram strategy={s} />

                {/* legs + theta quick row */}
                <div className="flex justify-between text-[10px] text-muted-foreground border-t pt-2">
                  <span>پاها: {s.legs}</span>
                  <span className={cn(
                    'font-medium',
                    s.timeDecay === 'Positive' ? 'text-emerald-400' :
                    s.timeDecay === 'Negative' ? 'text-red-400' : 'text-muted-foreground'
                  )}>
                    تتا: {DECAY_FA[s.timeDecay] ?? s.timeDecay}
                  </span>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Strategy Detail Dialog */}
      <StrategyDetailDialog
        strategy={detailStrategy}
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        onTrade={handleTrade}
      />
    </div>
  );
}
