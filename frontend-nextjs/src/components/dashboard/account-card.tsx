'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { formatCurrency } from '@/lib/utils';
import { Eye, MoreHorizontal, Plus, Minus, CreditCard, Wallet, Landmark, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface AccountCardData {
  id: string;
  name: string;
  number: string;
  balance: number;
  type: 'visa' | 'crypto' | 'savings';
  gradient: string;
}

const typeConfig = {
  visa: {
    icon: CreditCard,
    label: 'Trading',
    brand: 'VISA',
    glassTint: 'from-emerald-500/20 to-teal-600/15',
    iconBg: 'from-emerald-500/25 via-emerald-500/15 to-teal-600/20 dark:from-emerald-400/20 dark:via-emerald-500/10 dark:to-teal-500/25',
  },
  crypto: {
    icon: Wallet,
    label: 'Crypto',
    brand: '••••',
    glassTint: 'from-violet-500/20 to-indigo-600/15',
    iconBg: 'from-violet-500/25 via-indigo-500/15 to-purple-600/20 dark:from-violet-400/20 dark:via-indigo-500/10 dark:to-purple-500/25',
  },
  savings: {
    icon: Landmark,
    label: 'Savings',
    brand: '••••',
    glassTint: 'from-amber-500/20 to-orange-600/15',
    iconBg: 'from-amber-500/25 via-amber-500/15 to-orange-600/20 dark:from-amber-400/20 dark:via-amber-500/10 dark:to-orange-500/25',
  },
};

function CardChip() {
  return (
    <div
      className="absolute left-5 top-9 w-9 h-7 rounded bg-gradient-to-br from-amber-200/80 to-amber-400/70 shadow-inner border border-amber-500/20"
      aria-hidden
    >
      <div className="absolute inset-0.5 flex flex-wrap gap-px">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-1 w-1.5 bg-amber-600/40 rounded-sm" />
        ))}
      </div>
    </div>
  );
}

export function AccountCard({
  account,
  className,
  isLoading,
  error,
}: {
  account: AccountCardData;
  className?: string;
  /** When true, shows skeleton placeholders for balance and number (for API wiring). */
  isLoading?: boolean;
  /** When set, shows error state in card (for API wiring). */
  error?: string;
}) {
  const config = typeConfig[account.type];
  const Icon = config.icon;

  if (error) {
    return (
      <Card
        variant="default"
        hover={false}
        className={cn(
          'group/card overflow-hidden rounded-2xl flex flex-col w-full min-h-[180px] sm:min-h-[200px]',
          'border border-amber-200/50 dark:border-amber-500/30',
          'bg-amber-50/80 dark:bg-amber-950/30 backdrop-blur-xl',
          className
        )}
      >
        <CardContent className="relative flex flex-1 flex-col items-center justify-center gap-3 p-5 text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-500/20 ring-1 ring-amber-500/30">
            <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
          </div>
          <p className="text-sm font-medium text-amber-800 dark:text-amber-200">{error}</p>
          <p className="text-xs text-muted-foreground">{config.label}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      variant="default"
      hover={true}
      className={cn(
        'group/card overflow-hidden rounded-2xl transition-all duration-300 flex flex-col w-full',
        'min-h-[180px] sm:min-h-[200px]',
        'border border-white/30 dark:border-white/20',
        'bg-white/15 dark:bg-white/10 backdrop-blur-xl',
        'shadow-[0_8px_32px_rgba(0,0,0,0.08)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.2)]',
        'hover:bg-white/20 dark:hover:bg-white/15',
        'focus-within:ring-2 focus-within:ring-emerald-500/30 focus-within:ring-offset-2 focus-within:ring-offset-background',
        account.type === 'visa' && 'hover:border-emerald-400/30 dark:hover:border-emerald-500/20 hover:shadow-[0_12px_40px_rgba(16,185,129,0.15)]',
        account.type === 'crypto' && 'hover:border-violet-400/30 dark:hover:border-violet-500/20 hover:shadow-[0_12px_40px_rgba(139,92,246,0.15)]',
        account.type === 'savings' && 'hover:border-amber-400/30 dark:hover:border-amber-500/20 hover:shadow-[0_12px_40px_rgba(245,158,11,0.15)]',
        className
      )}
    >
      {/* Glass tint overlay (greeny fintech) */}
      <div
        className={cn(
          'absolute inset-0 pointer-events-none rounded-2xl bg-gradient-to-br opacity-90',
          config.glassTint
        )}
        aria-hidden
      />

      {/* Shine on hover */}
      <div
        className="absolute inset-0 opacity-0 group-hover/card:opacity-100 transition-opacity duration-500 pointer-events-none rounded-2xl"
        style={{
          background:
            'linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 50%)',
        }}
        aria-hidden
      />

      <CardContent className="relative flex flex-1 flex-col p-4 sm:p-5 text-foreground overflow-hidden">
        {/* Top row: icon + label + actions */}
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <div
              className={cn(
                'flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl',
                'bg-gradient-to-br',
                config.iconBg,
                'ring-1 ring-white/30 dark:ring-white/20 backdrop-blur-md',
                account.type === 'visa' && 'shadow-[0_2px_12px_rgba(16,185,129,0.2)]',
                account.type === 'crypto' && 'shadow-[0_2px_12px_rgba(139,92,246,0.2)]',
                account.type === 'savings' && 'shadow-[0_2px_12px_rgba(245,158,11,0.2)]'
              )}
            >
              <Icon
                className={cn(
                  'h-5 w-5',
                  account.type === 'visa' && 'text-emerald-700 dark:text-emerald-200',
                  account.type === 'crypto' && 'text-violet-700 dark:text-violet-200',
                  account.type === 'savings' && 'text-amber-700 dark:text-amber-200'
                )}
              />
            </div>
            <span className="text-xs font-semibold tracking-widest uppercase text-muted-foreground">
              {config.label}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 min-h-[44px] min-w-[44px] sm:h-8 sm:w-8 sm:min-h-0 sm:min-w-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/20 border-0 focus-visible:ring-2 focus-visible:ring-emerald-500/50 focus-visible:ring-offset-2"
              aria-label="View details"
            >
              <Eye className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9 min-h-[44px] min-w-[44px] sm:h-8 sm:w-8 sm:min-h-0 sm:min-w-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/20 border-0 focus-visible:ring-2 focus-visible:ring-emerald-500/50 focus-visible:ring-offset-2"
              aria-label="More options"
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <CardChip />

        {/* Card number */}
        <div className="mt-6 sm:mt-8 mb-3">
          {isLoading ? (
            <div className="h-4 w-full max-w-[180px] rounded bg-white/30 dark:bg-white/20 animate-pulse" aria-hidden />
          ) : (
            <p
              className="font-mono text-sm tracking-[0.25em] text-foreground/90 tabular-nums"
              style={{ letterSpacing: '0.22em' }}
            >
              {account.number}
            </p>
          )}
        </div>

        {/* Bottom: name + balance + actions */}
        <div className="mt-auto flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-0.5">
              {account.name}
            </p>
            {isLoading ? (
              <div className="h-7 w-28 rounded bg-white/30 dark:bg-white/20 animate-pulse" aria-hidden />
            ) : (
              <p className="text-xl font-bold tracking-tight tabular-nums text-foreground">
                {formatCurrency(account.balance)}
              </p>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              className="h-9 min-h-[44px] sm:h-8 sm:min-h-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-emerald-500/20 text-xs font-medium border border-emerald-500/20 focus-visible:ring-2 focus-visible:ring-emerald-500/50 focus-visible:ring-offset-2"
            >
              <Plus className="h-3.5 w-3.5 mr-1" />
              Deposit
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-9 min-h-[44px] sm:h-8 sm:min-h-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-emerald-500/20 text-xs font-medium border border-emerald-500/20 focus-visible:ring-2 focus-visible:ring-emerald-500/50 focus-visible:ring-offset-2"
            >
              <Minus className="h-3.5 w-3.5 mr-1" />
              Withdraw
            </Button>
          </div>
        </div>

        {/* Brand */}
        <div className="absolute right-4 bottom-4">
          <span className="text-[10px] font-bold tracking-widest text-muted-foreground/80 uppercase">
            {config.brand}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
