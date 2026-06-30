'use client';

import { useEffect, useState } from 'react';
import { Star, TrendingUp, Clock, Shield, Zap, Award } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CreditScoreProps {
  /** 300–850 */
  score?: number;
  className?: string;
}

type Grade = 'عالی' | 'خوب' | 'متوسط' | 'ضعیف' | 'بد';

function getGrade(score: number): { grade: Grade; color: string; bg: string; stars: number } {
  if (score >= 750) return { grade: 'عالی',   color: '#22C55E', bg: 'bg-green-500/10 border-green-500/20',   stars: 5 };
  if (score >= 650) return { grade: 'خوب',    color: '#4ade80', bg: 'bg-green-400/10 border-green-400/20',   stars: 4 };
  if (score >= 550) return { grade: 'متوسط', color: '#F59E0B', bg: 'bg-amber-500/10 border-amber-500/20',   stars: 3 };
  if (score >= 450) return { grade: 'ضعیف',  color: '#F97316', bg: 'bg-orange-500/10 border-orange-500/20', stars: 2 };
  return                    { grade: 'بد',    color: '#EF4444', bg: 'bg-red-500/10 border-red-500/20',       stars: 1 };
}

const FACTORS = [
  { key: 'winRate',   icon: TrendingUp, label: 'نرخ موفقیت',      weight: 35 },
  { key: 'activity',  icon: Clock,      label: 'فعالیت معاملاتی', weight: 20 },
  { key: 'risk',      icon: Shield,     label: 'مدیریت ریسک',      weight: 25 },
  { key: 'diversity', icon: Zap,        label: 'تنوع پرتفولیو',    weight: 20 },
];

// Normalise a score factor to 0-100 for the bar
function pct(score: number, weight: number): number {
  return Math.min(100, Math.max(0, ((score - 300) / 550) * weight * (100 / weight)));
}

export function CreditScore({ score: propScore = 712, className }: CreditScoreProps) {
  const [score, setScore] = useState(300); // start low for animation
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Animate score up on mount
    let raf: number;
    const target = propScore;
    const start = performance.now();
    const duration = 1200;

    function step(now: number) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out-cubic
      setScore(Math.round(300 + (target - 300) * eased));
      if (t < 1) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [propScore]);

  const { grade, color, bg, stars } = getGrade(score);

  // Progress bar for 300–850 range (550 span)
  const progressPct = ((score - 300) / 550) * 100;

  // Simulated factor scores
  const factorScores: Record<string, number> = {
    winRate:   Math.min(100, (score / 850) * 110),
    activity:  Math.min(100, (score / 850) * 95),
    risk:      Math.min(100, (score / 850) * 105),
    diversity: Math.min(100, (score / 850) * 88),
  };

  return (
    <div className={cn('rounded-2xl border bg-card/80 backdrop-blur-sm p-5', bg, className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="rounded-lg p-1.5" style={{ backgroundColor: `${color}18` }}>
            <Award className="h-4 w-4" style={{ color }} />
          </div>
          <span className="text-sm font-semibold text-foreground">امتیاز اعتباری</span>
        </div>
        <div className="flex gap-0.5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Star
              key={i}
              className="h-3.5 w-3.5"
              fill={i < stars ? color : 'transparent'}
              stroke={i < stars ? color : 'rgba(148,163,184,0.4)'}
              style={{ transition: 'all 0.3s ease' }}
            />
          ))}
        </div>
      </div>

      {/* Score display */}
      <div className="text-center mb-4">
        <div
          className="text-5xl font-bold tabular-nums leading-none mb-1"
          style={{ color, transition: 'color 0.5s ease', fontVariantNumeric: 'tabular-nums' }}
        >
          {score}
        </div>
        <div className="text-xs text-muted-foreground mb-2">از ۸۵۰ — {grade}</div>

        {/* Range bar */}
        <div className="relative h-2 rounded-full bg-muted overflow-hidden">
          <div
            className="absolute inset-y-0 right-0 rounded-full transition-all duration-[1200ms] ease-out"
            style={{
              width: mounted ? `${progressPct}%` : '0%',
              background: `linear-gradient(90deg, #EF4444, #F97316, #F59E0B, #22C55E)`,
            }}
          />
        </div>
        <div className="flex justify-between text-[9px] text-muted-foreground mt-1 px-0.5">
          <span>۳۰۰</span>
          <span>۵۵۰</span>
          <span>۸۵۰</span>
        </div>
      </div>

      {/* Grade badge */}
      <div className="flex justify-center mb-4">
        <span
          className="text-sm font-bold px-3 py-1 rounded-full border"
          style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}
        >
          {grade}
        </span>
      </div>

      {/* Factor bars */}
      <div className="space-y-2.5">
        {FACTORS.map((f) => {
          const Icon = f.icon;
          const pctVal = factorScores[f.key] ?? 0;
          return (
            <div key={f.key}>
              <div className="flex justify-between items-center mb-1">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Icon className="h-3 w-3" />
                  {f.label}
                </div>
                <span className="text-xs font-medium tabular-nums" style={{ color }}>
                  {Math.round(pctVal)}٪
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-[1000ms] ease-out"
                  style={{ width: mounted ? `${pctVal}%` : '0%', backgroundColor: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
