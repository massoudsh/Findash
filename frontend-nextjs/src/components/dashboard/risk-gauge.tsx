'use client';

import { useEffect, useRef, useState } from 'react';
import { Shield, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

type RiskLevel = 'safe' | 'moderate' | 'high' | 'critical';

interface RiskGaugeProps {
  /** 0–100 */
  value?: number;
  className?: string;
  /** If true, simulate live updates */
  live?: boolean;
}

const LEVELS: { level: RiskLevel; label: string; min: number; max: number; color: string; bg: string; icon: typeof Shield }[] = [
  { level: 'safe',     label: 'ایمن',     min: 0,  max: 25, color: '#22C55E', bg: 'bg-green-500/10 border-green-500/20',   icon: Shield },
  { level: 'moderate', label: 'متوسط',   min: 25, max: 50, color: '#F59E0B', bg: 'bg-amber-500/10 border-amber-500/20',   icon: Activity },
  { level: 'high',     label: 'بالا',     min: 50, max: 75, color: '#F97316', bg: 'bg-orange-500/10 border-orange-500/20', icon: TrendingDown },
  { level: 'critical', label: 'بحرانی',  min: 75, max: 100, color: '#EF4444', bg: 'bg-red-500/10 border-red-500/20',       icon: AlertTriangle },
];

function getLevel(value: number) {
  return LEVELS.find((l) => value >= l.min && value < l.max) ?? LEVELS[LEVELS.length - 1];
}

// Convert 0-100 value to SVG arc path
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startAngle));
  const y1 = cy + r * Math.sin(toRad(startAngle));
  const x2 = cx + r * Math.cos(toRad(endAngle));
  const y2 = cy + r * Math.sin(toRad(endAngle));
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

export function RiskGauge({ value: propValue, className, live = false }: RiskGaugeProps) {
  const [value, setValue] = useState(propValue ?? 34);
  const [animated, setAnimated] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setAnimated(true);
    if (live) {
      intervalRef.current = setInterval(() => {
        setValue((v) => {
          const delta = (Math.random() - 0.48) * 4;
          return Math.min(99, Math.max(1, v + delta));
        });
      }, 2500);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [live]);

  useEffect(() => { if (propValue !== undefined) setValue(propValue); }, [propValue]);

  const level = getLevel(value);
  const Icon  = level.icon;

  // Gauge: 210° arc from 210° to 330° (spans 300° total but let's use 180° semicircle)
  // Semicircle: start=-180, end=0 in standard → we'll use 210° start, 330° end = 300° sweep
  const START = 210;
  const SWEEP = 180;
  const cx = 80, cy = 75, r = 60;

  // Track background
  const trackPath = describeArc(cx, cy, r, START, START + SWEEP);
  // Value arc
  const valueDeg = (value / 100) * SWEEP;
  const valuePath = animated
    ? describeArc(cx, cy, r, START, START + valueDeg)
    : describeArc(cx, cy, r, START, START);

  // Needle
  const needleDeg = START + (value / 100) * SWEEP;
  const needleRad = (needleDeg * Math.PI) / 180;
  const needleX = cx + (r - 8) * Math.cos(needleRad);
  const needleY = cy + (r - 8) * Math.sin(needleRad);

  return (
    <div className={cn('rounded-2xl border bg-card/80 backdrop-blur-sm p-5', level.bg, className)}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg p-1.5" style={{ backgroundColor: `${level.color}18` }}>
            <Icon className="h-4 w-4" style={{ color: level.color }} />
          </div>
          <span className="text-sm font-semibold text-foreground">ریسک ریل‌تایم</span>
        </div>
        {live && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ backgroundColor: level.color }} />
            زنده
          </span>
        )}
      </div>

      {/* SVG Gauge */}
      <div className="flex justify-center">
        <svg viewBox="0 0 160 100" width="160" height="100" aria-label={`ریسک: ${Math.round(value)} از ۱۰۰`}>
          {/* Gradient defs */}
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stopColor="#22C55E" />
              <stop offset="33%"  stopColor="#F59E0B" />
              <stop offset="66%"  stopColor="#F97316" />
              <stop offset="100%" stopColor="#EF4444" />
            </linearGradient>
          </defs>

          {/* Track */}
          <path
            d={trackPath}
            fill="none"
            stroke="rgba(148,163,184,0.15)"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Colored track */}
          <path
            d={describeArc(cx, cy, r, START, START + SWEEP)}
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth="12"
            strokeLinecap="round"
            opacity="0.25"
          />

          {/* Value arc */}
          <path
            d={valuePath}
            fill="none"
            stroke={level.color}
            strokeWidth="12"
            strokeLinecap="round"
            style={{ transition: 'all 0.8s cubic-bezier(0.4,0,0.2,1)' }}
          />

          {/* Needle dot */}
          <circle
            cx={needleX}
            cy={needleY}
            r="5"
            fill={level.color}
            style={{ transition: 'all 0.8s cubic-bezier(0.4,0,0.2,1)' }}
          />
          <circle cx={cx} cy={cy} r="4" fill="rgba(148,163,184,0.3)" />

          {/* Center value */}
          <text x={cx} y={cy + 18} textAnchor="middle" fill={level.color} fontSize="20" fontWeight="700" fontFamily="Dana,Vazirmatn,sans-serif">
            {Math.round(value)}
          </text>
          <text x={cx} y={cy + 30} textAnchor="middle" fill="rgba(148,163,184,0.7)" fontSize="7" fontFamily="Dana,Vazirmatn,sans-serif">
            از ۱۰۰
          </text>
        </svg>
      </div>

      {/* Level badge */}
      <div className="flex justify-center mt-1 mb-3">
        <span className="text-sm font-bold px-3 py-1 rounded-full border" style={{ color: level.color, borderColor: `${level.color}40`, backgroundColor: `${level.color}10` }}>
          {level.label}
        </span>
      </div>

      {/* Risk Factors */}
      <div className="space-y-2">
        {[
          { label: 'ارزش در معرض ریسک (VaR)', value: `${(value * 0.28).toFixed(1)}%` },
          { label: 'افت حداکثری', value: `${(value * 0.35).toFixed(1)}%` },
          { label: 'بتا پرتفولیو', value: (0.6 + value / 200).toFixed(2) },
        ].map((f) => (
          <div key={f.label} className="flex justify-between text-xs">
            <span className="text-muted-foreground">{f.label}</span>
            <span className="font-medium tabular-nums" style={{ color: level.color }}>{f.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
